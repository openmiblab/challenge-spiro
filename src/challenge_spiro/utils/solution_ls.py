import numpy as np
import dcmri as dc


def inverse_ls_model(time, signal):

    # --- Initialize model
    aorta_liver = dc.AortaLiver2scan(

        # Simulation settings
        dt = 0.5,
        dose_tolerance = 0.1,

        # Injection parameters
        weight = 70,
        agent = 'gadoxetate',
        dose = dc.ca_std_dose('gadoxetate') / 4,
        dose2 = dc.ca_std_dose('gadoxetate') / 4,
        rate = 1,

        # Liver
        ve = 0.1,

        # Acquisition parameters
        tmax = time[-1][-1] + 60,
        field_strength = 3,
        TR = 0.004,
        FA = 20,
        FA2 = 20,

        # Signal parameters
        R10a = 1 / dc.T1(3, 'blood'),
        R10l = 1 / dc.T1(3, 'liver'),
    )

    # --- Train model
    aorta_liver.train(time, signal, verbose=2, xtol=1e-3)

    return aorta_liver


def inverse(time, signal):

    aorta_liver = inverse_ls_model(time, signal)

    # Extract the parameters
    ve, khe_i, khe_f, Th_i, Th_f, BAT, BAT2, S0a, S0l = aorta_liver.params('ve', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'BAT', 'BAT2', 'S0a', 'S0l')

    kbh_i = (1 - ve) / Th_i if Th_i > 0 else 0
    kbh_f = (1 - ve) / Th_f if Th_f > 0 else 0

    return khe_i, khe_f, kbh_i, kbh_f, BAT, BAT2, S0a, S0l
