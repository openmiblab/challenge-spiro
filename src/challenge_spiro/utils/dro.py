import os
import pickle

import numpy as np
from tqdm import tqdm

from challenge_spiro.utils.model import forward


def generate_subject(rng=None):

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