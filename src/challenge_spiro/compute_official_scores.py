import os

from challenge_spiro.utils.score import score_all_submissions
from challenge_spiro.utils.dro import generate_dro

secrets_dir = os.path.join(os.getcwd(), "secrets")
generated_secrets_dir = os.path.join(secrets_dir, 'generated')
os.makedirs(generated_secrets_dir, exist_ok=True)

data_file = os.path.join(secrets_dir, 'source', "data.dmr")
seed_file = os.path.join(secrets_dir, 'source', "seed.txt")
dro_file = os.path.join(generated_secrets_dir, "dro.pkl")
league_table = os.path.join(generated_secrets_dir, "league_table.csv")

# Compute DRO
with open(seed_file, "r") as f:
    dro_seed = int(f.read().strip())
generate_dro(dro_seed, dro_file) 

# Compute scores
score_all_submissions(dro_file, data_file, league_table)