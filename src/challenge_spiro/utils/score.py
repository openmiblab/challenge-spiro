import os
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pydmr
from tqdm import tqdm

from challenge_spiro import SUBMISSIONS
from challenge_spiro.utils.model import forward
from challenge_spiro.utils.solution_benchmark import inverse as inverse_benchmark


def accuracy_loss(inverse_model, dro_file):

    dro = np.load(dro_file)

    npix = dro['khe_i'].size
    err = 0

    for i in tqdm(range(npix), desc='Computing accuracy loss..'):
        time = (
            dro['time_1'], 
            dro['time_2'],
            dro['time_1'], 
            dro['time_2'],
        )
        signal = (
            dro['aorta_1'][i,:], 
            dro['aorta_2'][i,:],
            dro['liver_1'][i,:], 
            dro['liver_2'][i,:],
        )

        khe_i, khe_f, kbh_i, kbh_f = inverse_model(time, signal)

        rec = np.array([khe_i, khe_f, kbh_i, kbh_f])
        orig = np.array([dro['khe_i'][i], dro['khe_f'][i], dro['kbh_i'][i], dro['kbh_f'][i]])

        err += np.linalg.norm(rec - orig) / np.linalg.norm(orig)

    return err / npix


def generalizability_loss(inverse_model, data_file):
    
    data = pydmr.read(data_file, 'nest')
    rois = data['rois']

    err = 0
    nsig = 0
    for subj in tqdm(rois.keys(), desc='Computing generalizability loss..'):
        for visit in rois[subj].keys():

            roi = rois[subj][visit]

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

            results = inverse_model(time, signal)
            signal_rec = forward(*results, time)

            for i in range(4):
                err += np.linalg.norm(signal_rec[i] - signal[i]) / np.linalg.norm(signal[i])
                nsig += 1

    return err / nsig



def global_score(inverse_model, dro_file, data_file):

    print('Computing accuracy loss..')
    loss_1 = accuracy_loss(inverse_model, dro_file)

    print('Computing generalizability loss..')
    loss_2 = generalizability_loss(inverse_model, data_file)

    print('Computing accuracy loss for benchmark..')
    loss_1_benchmark = accuracy_loss(inverse_benchmark, dro_file)

    print('Computing generalizability loss for benchmark..')
    loss_2_benchmark = generalizability_loss(inverse_benchmark, data_file)

    # Return scores as a percentage of benchmark score
    accuracy_score = 100 * loss_1_benchmark / loss_1
    generalizability_score = 100 * loss_2_benchmark / loss_2
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


def score_all_submissions(
    dro_file,
    data_file,
    league_table
):
    for submitter, inverse_model in SUBMISSIONS.items():
        if submitter_exists(submitter, league_table):
            print(f"Submitter {submitter} has already been scored. If you want to compute the score again, remove the row manually from the league table and recompute scores.")
        else:
            print(f"Submitter {submitter}: computing score")
            scores = global_score(inverse_model, dro_file, data_file)
            print_scorecard(scores, submitter)
            save_to_league_table(scores, submitter, league_table)
            rank_league_table(league_table)