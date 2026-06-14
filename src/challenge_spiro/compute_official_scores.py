"""
Challenge Execution Engine & Automation Script
==============================================

This script serves as the centralized execution pipeline and entry point for 
the SPIRO Challenge evaluation workspace. It handles local file system pathing, 
reproduces the synthetic cohort generation, and triggers the batch scoring engine 
across all submitted models.

Execution Stages
----------------
1. **Directory Provisioning**: Maps the secure workspace path structure and creates 
   a transient directory (`secrets/generated/`) for staging ephemeral output assets.
2. **Deterministic DRO Assembly**: Extracts a random seed from an isolated text file 
   and generates a reproducible 64-subject Digital Reference Object (`dro.pkl`).
3. **Batch Evaluation**: Feeds the generated DRO and clinical baseline datasets into 
   the challenge grading suite to compute, print, and save leaderboard rankings.

Directory Structure Assumptions
-------------------------------
The module expects the following directory structure relative to the active working directory:

    current_working_directory/
    ├── secrets/
    │   ├── source/
    │   │   ├── data.dmr         <-- Clinical reference repository
    │   │   └── seed.txt         <-- Seed for reproducible cohort creation
    │   └── generated/           <-- (Auto-created if missing)
    │       ├── dro.pkl          <-- Produced by run
    │       └── league_table.csv <-- Ranked leaderboard log

Dependencies
------------
- os
- challenge_spiro.utils.score.score_all_submissions
- challenge_spiro.utils.dro.generate_dro

Usage
-----
Execute this module directly from your terminal inside the challenge workspace root:

.. code-block:: bash

    $ python -m challenge_spiro.compute_official_scores
"""

import os

from challenge_spiro.utils.score import score_all_submissions
from challenge_spiro.utils.dro import generate_dro

# ==============================================================================
# 1. PATH CONFIGURATION & WORKSPACE SETUP
# ==============================================================================
# Resolve standard environment directory paths
secrets_dir = os.path.join(os.getcwd(), "secrets")
generated_secrets_dir = os.path.join(secrets_dir, 'generated')

# Guard layout execution by ensuring generation outputs have a destination folder
os.makedirs(generated_secrets_dir, exist_ok=True)

# Map file locations for inputs and final metrics
data_file = os.path.join(secrets_dir, 'source', "data.dmr")
seed_file = os.path.join(secrets_dir, 'source', "seed.txt")
dro_file = os.path.join(generated_secrets_dir, "dro.pkl")
league_table = os.path.join(generated_secrets_dir, "league_table.csv")

# ==============================================================================
# 2. DETERMINISTIC COHORT GENERATION
# ==============================================================================
# Extract the random seed token to guarantee identical data generation characteristics
with open(seed_file, "r") as f:
    dro_seed = int(f.read().strip())

# Build the synthetic population reference database
generate_dro(dro_seed, dro_file) 

# ==============================================================================
# 3. SCORE EVALUATION & RANKING RUN
# ==============================================================================
# Pass processed parameters out to the grading stack to update standings
score_all_submissions(dro_file, data_file, league_table)