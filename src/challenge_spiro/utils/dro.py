import os
import numpy as np
from tqdm import tqdm

from challenge_spiro.utils.model import forward


def generate_subject():
    dt = 1.5
    tacq_1 = np.arange(0, 45 * 60, dt)
    tacq_2 = np.arange((45 + 90) * 60, (45 + 90 + 45) * 60, dt)

    khe_i = 30 / 6000
    khe_f = 30 / 6000
    kbh_i = 2.2 / 6000
    kbh_f = 2.2 / 6000
    BAT = 5 * 60
    BAT2 = (45 + 90 + 5) * 60
    S0a = 2
    S0l = 3

    params = (khe_i, khe_f, kbh_i, kbh_f, BAT, BAT2, S0a, S0l)
    time = (tacq_1, tacq_2, tacq_1, tacq_2)

    signal = forward(*params, time)
    return params, time, signal


def generate_dro(seed, file):

    if os.path.exists(file):
        print('Skipping calculation of DRO (already exists)')
        return
    
    _, time, signal = generate_subject()

    rng = np.random.default_rng(seed=seed)
    nsubj = 64

    khe_i = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    khe_f = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    kbh_i = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    kbh_f = rng.uniform(low=0.0, high=1e-2, size=nsubj)

    aorta_1 = np.zeros((nsubj, signal[0].size))
    aorta_2 = np.zeros((nsubj, signal[1].size))
    liver_1 = np.zeros((nsubj, signal[2].size))
    liver_2 = np.zeros((nsubj, signal[3].size))

    for i in tqdm(range(nsubj), desc='Computing DRO'):

        signal_i = forward(khe_i[i], khe_f[i], kbh_i[i], kbh_f[i], time, rng=rng)

        aorta_1[i, :] = signal_i[0]
        aorta_2[i, :] = signal_i[1]
        liver_1[i, :] = signal_i[2]
        liver_2[i, :] = signal_i[3]

    print(f'Saving dro to {file}')

    np.savez(
        file, 
        khe_i=khe_i, 
        khe_f=khe_f, 
        kbh_i=kbh_i, 
        kbh_f=kbh_f, 
        aorta_1=aorta_1,
        aorta_2=aorta_2,
        liver_1=liver_1,
        liver_2=liver_2,
        time_1=time[0],
        time_2=time[1]
    )

    print(f'Finished saving dro to {file}')



        

