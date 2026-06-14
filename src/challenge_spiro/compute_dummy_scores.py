import os
import dcmri as dc

from challenge_spiro.utils.score import score_all_submissions
from challenge_spiro.utils.dro import generate_dro

secrets_dir = os.path.join(os.getcwd(), "dummy_secrets")
os.makedirs(secrets_dir, exist_ok=True)

my_data_file = dc.fetch('tristan_humans_healthy_rifampicin')
my_dro_file = os.path.join(secrets_dir, "dro.npz")
my_league_table = os.path.join(secrets_dir, "league_table.csv")

my_seed = 1
generate_dro(my_seed, my_dro_file) 


# Compute scores
score_all_submissions(my_dro_file, my_data_file, my_league_table)