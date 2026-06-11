import numpy as np
import dcmri as dc

def initial(time, signal):

    BAT = time[0][np.argmax(signal[0])]
    BAT2 = time[1][np.argmax(signal[1])]
                   
    R10a = 1 / dc.T1(3, 'blood'),
    R10l = 1 / dc.T1(3, 'liver'),

    # Estimate S0
    Srefb = dc.sig.signal('SS', R10a, 1, FA=20, TR=0.004)
    Srefl = dc.sig.signal('SS', R10l, 1, FA=20, TR=0.004)

    n0 = 15
    S0a = np.mean(signal[0][1:n0]) / Srefb
    S0l = np.mean(signal[2][1:n0]) / Srefl

    return BAT, BAT2, S0a, S0l


def inverse(time, signal):

    BAT, BAT2, S0a, S0l = initial(time, signal)

    khe_i = 30 / 6000
    khe_f = 30 / 6000
    kbh_i = 2.2 / 6000
    kbh_f = 2.2 / 6000

    return khe_i, khe_f, kbh_i, kbh_f, BAT, BAT2, S0a, S0l