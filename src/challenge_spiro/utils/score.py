import numpy as np


def accuracy(inverse_model, dro_file):

    dro = np.load(dro_file)

    npix = dro['khe_i'].size
    err = 0

    for i in range(npix):
        time = (
            dro['time_1'], 
            dro['time_2'],
            dro['time_1'], 
            dro['time_2'],
        )
        signal = (
            dro['aorta_1'], 
            dro['arota_2'],
            dro['liver_1'], 
            dro['liver_2'],
        )

        khe_i, khe_f, kbh_i, kbh_f = inverse_model(time, signal)

        err += np.abs(khe_i - dro['khe_i'][i]) / dro['khe_i'][i]
        err += np.abs(khe_f - dro['khe_f'][i]) / dro['khe_f'][i]
        err += np.abs(kbh_i - dro['kbh_i'][i]) / dro['kbh_i'][i]
        err += np.abs(kbh_f - dro['kbh_f'][i]) / dro['kbh_f'][i]

    return err / npix / 4