import numpy as np
import dcmri as dc

from challenge_spiro.utils.model import PARAMS


def inverse(time, signal, **settings) -> dict:

    # --- Initialize model
    tmax = np.concatenate(time).max() + 60
    aorta_liver = dc.AortaLiver2scan(tmax=tmax, **settings)

    # --- Train model
    aorta_liver.train(time, signal, verbose=2, xtol=1e-3)

    # --- Return fitted parameters
    results = {k: aorta_liver.params(k) for k in PARAMS}

    return results
