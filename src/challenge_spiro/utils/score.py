"""
Challenge Evaluation, Benchmarking, and Leaderboard Pipeline
===========================================================

This module provides the evaluation and grading framework for the SPIRO Challenge.
It automates the scoring of submitted inverse models against two independent 
benchmarks (synthetic and experimental) and maintains a persistent leaderboard.

Key capabilities include:
1. **Accuracy Assessment**: Evaluating parameter reconstruction errors on synthetic 
   Digital Reference Objects (DROs) against known physiological ground truths.
2. **Generalizability Assessment**: Evaluating forward-signal consistency using 
   nested experimental or clinical imaging records.
3. **Normative Benchmarking**: Relative scoring where target performances are 
   computed as percentages of a baseline model (`inverse_normative`).
4. **Leaderboard Management**: Serializing scored metrics, generating console-based 
   ASCII scorecards, and appending/ranking submissions dynamically inside a CSV 
   league table.

Pipeline Workflow
-----------------
The batch scoring system handles multiple candidates registered under `SOLUTIONS`:

    [ Registered Solutions ] ──> Unique Submitter Iteration
                                        │
                                        ▼ ( Skip if exists in CSV )
                              ┌──────────────────┐
                              │   global_score   │
                              └─────────┬────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
           ┌──────────────────┐                  ┌──────────────────┐
           │  accuracy_loss   │                  │generalizability_ │
           └──────────────────┘                  │       loss       │
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ print_scorecard  │ ──> (ASCII Console Display)
                              └─────────┬────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │save_to_league_   │ ──> (Append & Sort CSV)
                              │      table       │
                              └──────────────────┘

Dependencies
------------
- numpy
- csv
- datetime
- tqdm
- pydmr (for nesting file syntax parsing)

Usage Example
-------------
To process all registered candidate entries over the reference assets:

>>> from spiro_challenge.utils.score import score_all_submissions
>>> # Define files
>>> dro = "data/dro_cohort_64.pkl"
>>> data = "data/clinical_trial.nest"
>>> table = "leaderboard/standings.csv"
>>> # Run grading suite
>>> score_all_submissions(dro, data, table)
Submitter team_alpha_nn: computing score
...
"""


import os
import csv
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pydmr
from tqdm import tqdm

from challenge_spiro.solutions.registered import SOLUTIONS
from challenge_spiro.utils.model import forward
from challenge_spiro.solutions.provided.normative import inverse as inverse_normative






def accuracy_loss(inverse_model, dro_file):
    """
    Calculate the 90th percentile of relative reconstruction errors across a DRO cohort.

    This function evaluates the performance of an inverse model 
    against a pre-computed Digital Reference 
    Object (DRO) cohort dataset. It reconstructs parameters for each subject, 
    computes the normalized Euclidean distance ($L_2$ norm) between the predicted 
    parameters and the true parameters, and aggregates these errors.

    Parameters
    ----------
    inverse_model : callable
        A function or model instance capable of mapping time-series data back into 
        physiological parameters. It must accept two arguments (`time`, `signal`) 
        and return a dictionary where the values represent estimated parameters.
    dro_file : str or pathlib.Path
        The file path to a pickled dictionary containing the DRO cohort dataset 
        (must contain 'data' and 'truth' high-level keys).

    Returns
    -------
    float
        The 90th percentile of the relative reconstruction error across the cohort. 
        A lower value indicates a more accurate and robust inverse model.

    Notes
    -----
    **Error Metric:**
    For each subject $i$, the relative error ($e_i$) is computed as the ratio of 
    the Euclidean norm ($L_2$ norm) of the absolute error vector to the norm 
    of the ground-truth vector:

    .. math:: e_i = \\frac{\\|\\mathbf{p}_{recon} - \\mathbf{p}_{truth}\\|_2}{\\|\\mathbf{p}_{truth}\\|_2}

    **Percentile Aggregation:**
    Returning the 90th percentile instead of the mean or median ensures that the 
    metric accounts for the "worst-case" edge performance, acting as a robust indicator 
    of stability against outliers or poorly converged fits across the cohort.

    Examples
    --------
    >>> from challenge_spiro.utils.score import accuracy_loss
    >>> loss = accuracy_loss(perfect_inverse, "cohort_64_subjects.pkl")
    Computing accuracy loss..: 100%|██████████████████████████████| 64/64 [00:02<00:00, 24.11it/s]
    >>> print(f"90th Percentile Relative Error: {loss:.4f}")
    """
    with open(dro_file, "rb") as f:
        dro = pickle.load(f)

    npix = len(dro['truth'])
    err = []

    for i in tqdm(range(npix), desc='Computing accuracy loss..'):

        time = dro['data'][i][0]
        signal = dro['data'][i][1]

        recon = inverse_model(time, signal)

        # Extract values from both dicts using a shared, strict key ordering
        keys = list(dro['truth'][i].keys())
        
        truth = np.array([dro['truth'][i][k] for k in keys])
        recon = np.array([recon[k] for k in keys])

        err_i = np.linalg.norm(recon - truth) / np.linalg.norm(truth)
        err.append(err_i)

    return np.percentile(err, 90)


def generalizability_loss(inverse_model, data_file):
    """
    Calculate the 90th percentile of relative signal reconstruction errors on experimental data.

    Unlike accuracy loss (which evaluates against a synthetic ground truth), this function 
    measures generalizability by checking for *self-consistency* on clinical or experimental 
    datasets where the underlying true parameters are unknown. 
    
    It extracts data across all subjects and longitudinal visits, uses the `inverse_model` 
    to estimate physiological parameters, passes those estimates back through the `forward` 
    simulation model to synthesize a reconstructed signal, and compares it directly 
    against the original observed signal.

    Parameters
    ----------
    inverse_model : callable
        A function or model instance that maps a time-series curve to physiological 
        parameters. It must accept three arguments: `(time, signal, **settings)` 
        and return a parameter dictionary compatible with the `forward` function.
    data_file : str or pathlib.Path
        The path to the experimental dataset file, readable by `pydmr.read` using 
        the 'nest' format layout.

    Returns
    -------
    float
        The 90th percentile of relative signal reconstruction errors aggregated across 
        all subjects, visits, and multi-phase signals (4 phases per visit). Lower values 
        indicate the model generalizes well to real-world data variations.

    Notes
    -----
    **Evaluation Mechanism:**
    For each subject, visit, and each of the 4 acquisition phases ($i$), the relative 
    discrepancy between the observed signal and the re-simulated forward signal is 
    calculated as:

    .. math:: e_{subj, visit, i} = \\frac{\\|\\mathbf{s}_{recon, i} - \\mathbf{s}_{observed, i}\\|_2}{\\|\\mathbf{s}_{observed, i}\\|_2}

    **Nested Schema:**
    The file structure assumes a hierarchical nested tree parsing layout:
    ``Subject ID ──> Visit ID ──> Regions of Interest (ROIs)``
    Every single combination is parsed individually and generates 4 discrete error points 
    (one for each component of the multi-phase signal tuple).

    Examples
    --------
    >>> from challenge_spiro.utils.score import generalizability_loss
    >>> loss = generalizability_loss(my_inverse_model, "clinical_trial_data.nest")
    Computing generalizability loss..: 100%|██████████████████████████████| 20/20 [00:15<00:00,  1.33it/s]
    >>> print(f"Signal Generalizability Loss (90th percentile): {loss:.4f}")
    """
    
    data = pydmr.read(data_file, 'nest')
    rois = data['rois']

    err = []
    for subj in tqdm(rois.keys(), desc='Computing generalizability loss..'):
        for visit in rois[subj].keys():

            time, signal, settings = parse_data(data_file, subj, visit)

            params_recon = inverse_model(time, signal, **settings)
            # signal_recon = forward(params_recon, time) # BUG v0.0.0: Not passing settings!!!!
            signal_recon = forward(params_recon, time, **settings) # BUG fix v0.0.1

            for i in range(4):
                err_i = np.linalg.norm(signal_recon[i] - signal[i]) / np.linalg.norm(signal[i])
                err.append(err_i)

    return np.percentile(err, 90)


def parse_data(data_file, subj, visit):
    """
    Parse experimental data for a specific subject and visit into modeling inputs.

    Extracts regions of interest (ROIs) and experimental parameters from a nested 
    dataset, normalizes acquisition timelines relative to the start of Phase 1, 
    applies quality-control acceptance filters, and bundles MRI/dosing configurations.

    Parameters
    ----------
    data_file : str or pathlib.Path
        Path to the source data file readable by `pydmr.read` using 'nest' format.
    subj : str
        The unique identifier for the target subject.
    visit : str
        The identifier for the specific longitudinal session or visit.

    Returns
    -------
    time : tuple of numpy.ndarray
        A 4-element tuple containing time grids (in seconds) for the filtered aorta 
        (Phase 1 & 2) and liver (Phase 1 & 2), normalized to begin at `time_1[0]`.
    signal : tuple of numpy.ndarray
        A 4-element tuple containing the raw measurement signals corresponding to the 
        accepted time points for the aorta and liver across both phases.
    settings : dict
        A dictionary of experimental, sequence, and dosing parameters (e.g., patient 
        weight, flip angles, agent doses) used to configure the forward or inverse models.
    """
    data = pydmr.read(data_file, 'nest')

    roi = data['rois'][subj][visit]
    par = data['pars'][subj][visit]

    time = (
        roi['time_1'][roi['aorta_1_accept']] - roi['time_1'][0],
        roi['time_2'][roi['aorta_2_accept']] - roi['time_1'][0],
        roi['time_1'][roi['liver_1_accept']] - roi['time_1'][0],
        roi['time_2'][roi['liver_2_accept']] - roi['time_1'][0],
    )
    signal = (
        roi['aorta_1'][roi['aorta_1_accept']],
        roi['aorta_2'][roi['aorta_2_accept']],
        roi['liver_1'][roi['liver_1_accept']],
        roi['liver_2'][roi['liver_2_accept']],
    )
    settings = {
        'weight': par['weight'],
        'field_strength': 3,
        'TR': par['TR'],
        'FA': par['FA_1'],
        'FA2': par['FA_2'],
        'agent': 'gadoxetate',
        'dose': par['dose_1'],
        'dose2': par['dose_2'],
        'rate': 1,
    }
    return time, signal, settings

def global_score(inverse_model, dro_file, data_file):
    """
    Compute a normalized evaluation benchmark score relative to a normative model.

    This function calculates a composite evaluation score for a target `inverse_model` 
    by assessing both its parameter parameter estimation accuracy (via a synthetic DRO cohort) 
    and its signal estimation generalizability (via an experimental dataset). 

    To contextualize performance, the target model's losses are benchmarked against 
    a standard baseline implementation (`inverse_normative`). The final outputs are 
    expressed as percentages of this normative benchmark, where higher values indicate 
    better relative performance (lower loss) than the baseline.

    Parameters
    ----------
    inverse_model : callable
        The candidate inverse model/fitting routing undergoing evaluation. Must be 
        compatible with the signatures required by `accuracy_loss` and `generalizability_loss`.
    dro_file : str or pathlib.Path
        The file path to the pickled Digital Reference Object (DRO) data cohort used 
        for evaluating parameter mapping accuracy.
    data_file : str or pathlib.Path
        The file path to the nested experimental or clinical dataset used for 
        evaluating signal reconstruction generalizability.

    Returns
    -------
    scores : dict
        A dictionary containing the benchmarked evaluations:
        
        - 'global score' : float
            The evenly weighted arithmetic mean (50/50) of the accuracy and 
            generalizability benchmark scores.
        - 'accuracy score' : float
            Normalized parameter tracking metric. Calculated as: 
            :math:`100 \\times \\text{loss}_{normative} / \\text{loss}_{target}`.
        - 'generalizability score' : float
            Normalized signal generalizability metric. Calculated as: 
            :math:`100 \\times \\text{loss}_{normative} / \\text{loss}_{target}`.

    Notes
    -----
    **Interpretation of Percentages:**
    Because the scores invert the loss fractions, the scoring profile behaves as follows:
    
    * **Score > 100%**: The target model outperforms the normative model (achieves lower loss).
    * **Score == 100%**: The target model performs identically to the normative model.
    * **Score < 100%**: The target model underperforms relative to the normative model.

    *Warning*: This function relies on a globally scoped or predefined reference object 
    named `inverse_normative` to compute the baseline metrics.

    Examples
    --------
    >>> from challenge_spiro.utils.score import global_score
    >>> results = global_score(my_optimized_fitter, "simulated_cohort.pkl", "patient_registry.nest")
    Computing accuracy loss for normative..
    ...
    >>> print(f"Global Benchmark Rating: {results['global score']:.2f}%")
    Global Benchmark Rating: 114.52%
    """
    print('Computing accuracy loss for normative..')
    loss_1_normative = accuracy_loss(inverse_normative, dro_file)

    print('Computing generalizability loss for normative..')
    loss_2_normative = generalizability_loss(inverse_normative, data_file)

    print('Computing accuracy loss..')
    loss_1 = accuracy_loss(inverse_model, dro_file)

    print('Computing generalizability loss..')
    loss_2 = generalizability_loss(inverse_model, data_file)

    # Return scores as a percentage of benchmark score
    accuracy_score = 100 * loss_1_normative / loss_1
    generalizability_score = 100 * loss_2_normative / loss_2
    global_score = 0.5 * (accuracy_score + generalizability_score)

    return {
        'global score': global_score, 
        'accuracy score': accuracy_score, 
        'generalizability score': generalizability_score,
    }


def submitter_exists(submitter: str, league_table: str) -> bool:
    """
    Returns True if the submitter is already present in the league table CSV.
    """
    if not os.path.isfile(league_table):
        return False

    with open(league_table, mode="r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("submitter") == submitter:
                return True

    return False

def print_scorecard(scores: dict, submitter: str):
    """
    Print a formatted, double-bordered ASCII console scorecard for a submission.

    Centers a dynamic challenge header and tabularizes metric scores, automatically
    replacing underscores with title-cased labels and formatting floats to 8 decimals.

    Parameters
    ----------
    scores : dict
        Dictionary of metric names (keys) and numerical values or statuses (values).
    submitter : str
        The name or identifier of the challenge participant.
    """
    title = f"SPIRO CHALLENGE (Submitter: {submitter})"

    width = max(50, len(title) + 4)
    line = "═" * width

    def fmt(v):
        return f"{v:,.8f}" if isinstance(v, (int, float)) else str(v)

    print("\n" + line)
    print(title.center(width))
    print(line)

    for k, v in scores.items():
        label = k.replace("_", " ").title()
        print(f"   {label:<28} {fmt(v)}")

    print(line + "\n")


def save_to_league_table(scores: dict, submitter: str, league_table: str):
    """
    Append evaluation scores and metadata as a new row in a CSV league table.

    Flattens the submission details by bundling the submitter's identity, an ISO 
    timestamp, and the metric results into a single row. Dynamically initializes 
    the CSV file with appropriate headers if the file does not already exist.

    Parameters
    ----------
    scores : dict
        A dictionary containing metric names (keys) and their calculated values.
    submitter : str
        The name or identifier of the challenge participant.
    league_table : str or pathlib.Path
        The destination path to the CSV file representing the leaderboard.
    """
    # Flatten row
    row = {
        "submitter": submitter,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **scores
    }

    file_exists = os.path.isfile(league_table)

    with open(league_table, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Results for submitter {submitter} saved to league table")



def rank_league_table(league_table: str,
                      score_key: str = "global score"):
    """
    Sorts the league table by global score (descending)
    and overwrites the CSV in ranked order.
    """

    if not os.path.isfile(league_table):
        raise FileNotFoundError(f"No league table found at: {league_table}")

    df = pd.read_csv(league_table)

    if score_key not in df.columns:
        raise ValueError(f"'{score_key}' column not found in CSV.")

    # Ensure numeric sorting (handles accidental strings)
    df[score_key] = pd.to_numeric(df[score_key], errors="coerce")

    # Sort descending (higher score = better rank)
    df = df.sort_values(by=score_key, ascending=False).reset_index(drop=True)

    # Save back to CSV
    df.to_csv(league_table, index=False)

    print(f"League table - new ranking!")


def score_all_solutions(
    dro_file,
    data_file,
    league_table
):
    """
    Batch score all missing entries in the global solutions register.

    Iterates through registered solutions, skipping entries that already exist in 
    the leaderboard to prevent redundant evaluations. For new submissions, it 
    computes composite scores, prints a formatted console scorecard, appends the 
    results to the CSV table, and updates the leaderboard rankings.

    Parameters
    ----------
    dro_file : str or pathlib.Path
        Path to the synthetic Digital Reference Object file used for accuracy checks.
    data_file : str or pathlib.Path
        Path to the experimental nested data file used for generalizability checks.
    league_table : str or pathlib.Path
        Path to the destination CSV file maintaining the ranked standings.

    Examples
    --------
    >>> from spiro_challenge.utils.score import score_all_submissions
    >>> # Global dictionary containing candidate models
    >>> SOLUTIONS = {
    ...     "team_alpha_nn": alpha_inverse_model,
    ...     "team_beta_curvefit": beta_inverse_model
    ... }
    >>> # Running the batch evaluator
    >>> score_all_submissions(
    ...     dro_file="dro_cohort.pkl",
    ...     data_file="clinical_trial.nest",
    ...     league_table="leaderboard.csv"
    ... )
    Submitter team_alpha_nn: computing score
    ...
    """
    for submitter, inverse_model in SOLUTIONS.items():
        if submitter_exists(submitter, league_table):
            print(f"Submitter {submitter} has already been scored. If you want to compute the score again, remove the row manually from the league table and recompute scores.")
        else:
            print(f"Submitter {submitter}: computing score")
            scores = global_score(inverse_model, dro_file, data_file)
            print_scorecard(scores, submitter)
            save_to_league_table(scores, submitter, league_table)
            rank_league_table(league_table)