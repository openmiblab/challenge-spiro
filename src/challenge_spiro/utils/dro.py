import numpy as np

from challenge_spiro.utils.model import forward


def generate_dro(seed, file):

    rng = np.random.default_rng(seed=seed)
    nsubj = 64

    khe_i = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    khe_f = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    kbh_i = rng.uniform(low=0.0, high=1e-2, size=nsubj)
    kbh_f = rng.uniform(low=0.0, high=1e-2, size=nsubj)

    for i in range(nsubj):

        time, signal_i = forward(khe_i[i], khe_f[i], kbh_i[i], kbh_f[i], rng=rng)

        if i == 0:
            aorta_1 = np.zeros((nsubj, ) + signal_i[0].shape)
            aorta_2 = np.zeros((nsubj, ) + signal_i[1].shape)
            liver_1 = np.zeros((nsubj, ) + signal_i[2].shape)
            liver_2 = np.zeros((nsubj, ) + signal_i[3].shape)

        aorta_1[i, ...] = signal_i[0]
        aorta_2[i, ...] = signal_i[1]
        liver_1[i, ...] = signal_i[2]
        liver_2[i, ...] = signal_i[3]

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



        

