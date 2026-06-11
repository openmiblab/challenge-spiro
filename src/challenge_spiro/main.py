import os

from challenge_spiro.utils.score import score_all_submissions
from challenge_spiro.utils.dro import generate_dro

secrets_dir = os.path.join(os.getcwd(), "secrets")

seed_file = os.path.join(secrets_dir, "seed.txt")
dro_file = os.path.join(secrets_dir, "dro.npz")
data_file = os.path.join(secrets_dir, "data.dmr")
league_table = os.path.join(secrets_dir, "league_table.csv")


# Compute DRO
def load_seed(seed_path: str) -> int:
    with open(seed_path, "r") as f:
        content = f.read().strip()
    return int(content)

dro_seed = load_seed(seed_file)
generate_dro(dro_seed, dro_file) 


# Compute scores
score_all_submissions(dro_file, data_file, league_table)