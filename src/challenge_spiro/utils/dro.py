"""
Digital Reference Object (DRO) Generation Pipeline
==================================================

This module provides a stochastic data generation framework to simulate 
synthetic subject populations and their corresponding measurement signals. 
It is designed to create Digital Reference Objects (DROs) for validating 
kinetic modeling, parameter estimation, and machine learning pipelines.

The simulation generates multi-phase time grids, randomizes complex physiological/
signal parameters from valid uniform boundaries, and passes them through a 
forward simulation model to output a decoupled 'data' vs. 'ground-truth' cohort.

Pipeline Architecture
---------------------
1. `generate_subject`: Randomly samples an acquisition timeline (two scan 
   sessions separated by an inter-scan wait period) and generates 18 distinct 
   baseline, kinetic, and physiological parameters.
2. `generate_dro`: Orchestrates the cohort creation by iteratively calling the 
   subject generator, pushing parameters through the `forward()` model, and 
   serializing a reproducible 64-subject cohort dictionary to a pickle file.

Data Structures
---------------
The generated DRO is saved as a serialized dictionary with the following schema:

    dro = {
        'data': [
            (time_tuple_subj1, signal_array_subj1),
            (time_tuple_subj2, signal_array_subj2),
            ...
        ],
        'truth': [
            {param_1: val, param_2: val, ...}, # Subj 1 ground truth
            {param_1: val, param_2: val, ...}, # Subj 2 ground truth
            ...
        ]
    }

Dependencies
------------
- numpy (>= 1.20.0)
- tqdm
- os
- pickle

Usage Example
-------------
To generate a new cohort file from the command line or an external script:

>>> from challenge_spiro.utils.dro import generate_dro
>>> generate_dro(seed=101, file="synthetic_cohort_v1.pkl")
"""



import os
import pickle

import numpy as np
from tqdm import tqdm

from challenge_spiro.utils.model import forward


def generate_subject(rng=None):
    """
    Generate randomized acquisition timelines and physiological parameters for a subject.

    This function simulates a two-phase data acquisition setup (dual-injection, 
    split-session imaging) separated by a resting/waiting period. It randomly 
    samples temporal configurations and kinetic parameters from uniform distributions
    to create a distinct synthetic subject dataset.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        A specific NumPy random number generator instance to ensure reproducible 
        sampling. If None, a new generator is initialized via `np.random.default_rng()`.

    Returns
    -------
    params : dict
        A dictionary containing the generated kinetic, signal, and physiological parameters:
        
        - 'S0a', 'S0l' : float
            Baseline signal intensity factors for phase 1 [0.0, 1000.0].
        - 'S02a', 'S02l' : float
            Phase 2 baseline signal scaling, varying ±25% relative to Phase 1.
        - 'BAT' : float
            Bolus Arrival Time for Phase 1 in seconds [60.0, 600.0].
        - 'BAT2' : float
            Bolus Arrival Time for Phase 2 in absolute timeline seconds 
            ([60.0, 600.0] + Phase 2 start time).
        - 'CO' : float
            Cardiac Output or baseline blood flow parameter [0.0, 300.0].
        - 'Thl', 'Dhl' : float
            Clearance/half-life constants ([0.0, 30.0] and [0.05, 0.95]).
        - 'To', 'Eo', 'Toe' : float
            Washout, extraction, and equilibrium constants.
        - 'Eb' : float
            Background blood volume fraction [0.01, 0.15].
        - 've' : float
            Extracellular extravascular space fraction [0.01, 0.6].
        - 'Te', 'De' : float
            Exchange time constant [0.1, 60.0] and diffusion coefficient [0.0, 1.0].
        - 'khe_i', 'khe_f' : float
            Initial and final elimination/transfer rates [0.0, 0.01].
        - 'Th_i', 'Th_f' : float
            Compartmental half-life parameters in seconds [600.0, 36000.0] (10m to 10h).

    time : tuple of numpy.ndarray
        A 4-element tuple `(tacq_1, tacq_2, tacq_1, tacq_2)` representing the 
        time grids (in seconds) for the respective measurement phases:
        
        - tacq_1 : Time grid for Phase 1 (starts at 0, spans `t_acq`).
        - tacq_2 : Time grid for Phase 2 (starts after `t_acq + t_wait`, spans `t_acq`).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> params, time_grids = generate_subject(rng)
    >>> print(params['BAT'])
    521.0531514757348
    """
    if rng is None:
        rng = np.random.default_rng()

    # Acquisition parameters
    dt = rng.uniform(low=1.5, high=2.0)
    t_acq = 60 * rng.uniform(low=40, high=50)
    t_wait = 60 * rng.uniform(low=60, high=120)

    # Time points
    tacq_1 = np.arange(0, t_acq, dt)
    tacq_2 = np.arange(t_acq + t_wait, t_acq + t_wait + t_acq, dt)
    time = (tacq_1, tacq_2, tacq_1, tacq_2)

    # Parameters
    params = {
        'S0a': rng.uniform(low=0.0, high=1e3), 
        'S0l': rng.uniform(low=0.0, high=1e3),
        'BAT': rng.uniform(low=60.0, high=600), 
        'BAT2': rng.uniform(low=60.0, high=600) + tacq_2[0], 
        'CO': rng.uniform(low=0.0, high=300), 
        'Thl': rng.uniform(low=0.0, high=30), 
        'Dhl': rng.uniform(low=0.05, high=0.95), 
        'To': rng.uniform(low=0.0, high=60), 
        'Eo': rng.uniform(low=0.0, high=0.5), 
        'Toe': rng.uniform(low=0.0, high=800), 
        'Eb': rng.uniform(low=0.01, high=0.15), 
        've': rng.uniform(low=0.01, high=0.6), 
        'Te': rng.uniform(low=0.1, high=60), 
        'De': rng.uniform(low=0.0, high=1), 
        'khe_i': rng.uniform(low=0.0, high=1e-2), 
        'khe_f': rng.uniform(low=0.0, high=1e-2), 
        'Th_i': rng.uniform(low=10 * 60, high=10 * 60 * 60), 
        'Th_f': rng.uniform(low=10 * 60, high=10 * 60 * 60), 
    }
    params['S02a'] = params['S0a'] * rng.uniform(low=0.75, high=1.25) 
    params['S02l'] = params['S0l'] * rng.uniform(low=0.75, high=1.25) 

    return params, time



def generate_dro(seed, file):
    """
    Generate and save a Digital Reference Object (DRO) for a cohort of synthetic subjects.

    This function simulates synthetic dataset pairs (time-series data and their corresponding 
    ground-truth kinetic parameters) for a cohort of 64 subjects. It uses a reproducible 
    random number generator seeded with the provided value. If the target file already exists, 
    the computation is skipped to avoid redundant processing.

    Parameters
    ----------
    seed : int
        The seed value used to initialize the NumPy default random number generator (`np.random.default_rng`). 
        Ensures strict reproducibility of the generated cohort.
    file : str or pathlib.Path
        The destination file path where the generated DRO dictionary will be saved as a pickled binary file.

    Returns
    -------
    None
        The function saves the output directly to disk and returns nothing.

    Side Effects
    ------------
    - Checks for the existence of `file` on disk.
    - Prints progress updates and an interactive progress bar (`tqdm`) to the console.
    - Serializes and writes a dictionary to `file` using `pickle.dump()`.

    Notes
    -----
    The resulting dictionary structure saved to the pickle file is:
    
    .. code-block:: python

        dro = {
            'data': [
                (time_subject_1, signal_subject_1),
                (time_subject_2, signal_subject_2),
                ...
            ],
            'truth': [
                params_subject_1_dict,
                params_subject_2_dict,
                ...
            ]
        }

    - `time_subject_i`: A 4-element tuple of numpy arrays specifying the acquisition timelines.
    - `signal_subject_i`: The simulated signal array generated by the `forward` model.
    - `params_subject_i_dict`: A dictionary containing the ground-truth physiological parameters.

    Examples
    --------
    >>> generate_dro(seed=42, file="cohort_64_subjects.pkl")
    Computing DRO: 100%|██████████████████████████████| 64/64 [00:04<00:00, 15.20it/s]
    Saving dro to cohort_64_subjects.pkl
    Finished saving dro to cohort_64_subjects.pkl
    
    >>> # Running it again skips execution:
    >>> generate_dro(seed=42, file="cohort_64_subjects.pkl")
    Skipping calculation of DRO (already exists)
    """
    if os.path.exists(file):
        print('Skipping calculation of DRO (already exists)')
        return
    
    nsubj = 64

    # Initializing RNG from seed so the DRO can be reproduced
    rng = np.random.default_rng(seed=seed)

    # Compute DRO separating data from ground truth
    dro = {'data': [], 'truth': []}
    for _ in tqdm(range(nsubj), desc='Computing DRO'):

        params_i, time_i = generate_subject(rng)
        signal_i = forward(params_i, time_i)

        dro['truth'].append(params_i)
        dro['data'].append((time_i, signal_i))

    print(f'Saving dro to {file}')

    with open(file, "wb") as f:
        pickle.dump(dro, f)

    print(f'Finished saving dro to {file}')