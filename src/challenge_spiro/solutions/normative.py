import numpy as np
import dcmri as dc

from challenge_spiro.utils.model import PARAMS

def inverse(time, signal, **settings) -> dict:

    FA = settings['FA'] if 'FA' in settings else 20
    TR = settings['TR'] if 'TR' in settings else 0.004

    # --- Initialize model
    aorta_liver = dc.AortaLiver2scan(**settings)

    # --- Get initial values for all PARAMS parameters
    results = {k: aorta_liver.params(k) for k in PARAMS}

    # Estimate bolus arrival times from data
    results['BAT'] = time[0][np.argmax(signal[0])]
    results['BAT2'] = time[1][np.argmax(signal[1])] - time[1][0]
                   
    # Estimate signal scaling factors from data
    R10a = 1 / dc.T1(3, 'blood')
    R10l = 1 / dc.T1(3, 'liver')
    Srefb = dc.sig.signal('SS', R10a, 1, FA=FA, TR=TR)
    Srefl = dc.sig.signal('SS', R10l, 1, FA=FA, TR=TR)

    n0 = 5
    results['S0a'] = np.mean(signal[0][1:n0]) / Srefb
    results['S02a'] = results['S0a']
    results['S0l'] = np.mean(signal[2][1:n0]) / Srefl
    results['S02l'] = results['S0l']

    return results