import numpy as np
import dcmri as dc

CONST = {

    # Simulation settings
    'dt' : 0.5,
    'dose_tolerance': 0.1,

    # Injection parameters
    'weight' : 70,
    'agent' : 'gadoxetate',
    'dose' : dc.ca_std_dose('gadoxetate') / 4,
    'dose2' : dc.ca_std_dose('gadoxetate') / 4,
    'rate' : 1,

    # Acquisition parameters
    'field_strength' : 3,
    't0' : 30,
    'TR' : 0.004,
    'FA' : 20,
    'FA2' : 20,

    # Signal parameters
    'R10a' : 1 / dc.T1(3, 'blood'),
    'R10l' : 1 / dc.T1(3, 'liver'),

    # Tissue parameters
    'vol' : 1000,
}

def forward_model(ve, khe_i, khe_f, kbh_i, kbh_f, TS=2.5, t_acq=(45, 45), t_wait=90):

    return dc.AortaLiver2scan(

        # Liver parameters
        ve = ve,
        khe_i = khe_i,
        khe_f = khe_f,
        Th_i = (1 - ve) / kbh_i if kbh_i > 0 else 0,
        Th_f = (1 - ve) / kbh_f if kbh_f > 0 else 0,

        # Acquisition parameters
        TS = TS,
        tmax = (t_acq[0] + t_wait + t_acq[1]) * 60,

        # Fixed parameters
        **CONST,
    )

def inverse_ls_model(time, signal):

    TS = time[0][1]

    # --- Initialize model
    aorta_liver = dc.AortaLiver2scan(

        # Acquisition parameters
        TS = TS,
        tmax = time[-1][-1] + TS,

        # Fixed parameters
        **CONST,
    )

    # --- Train model
    aorta_liver.train(time, signal, xtol=1e-3, verbose=2)

    return aorta_liver


def forward(ve, khe_i, khe_f, kbh_i, kbh_f, TS=2.5, t_acq=(45, 45), t_wait=90, SNR=10):

    # --- Initialize the model
    aorta_liver = forward_model(ve, khe_i, khe_f, kbh_i, kbh_f, TS=TS, t_acq=t_acq, t_wait=t_wait)

    # --- Acquired time points
    t_1 = np.arange(0, t_acq[0] * 60, TS)
    t_2 = np.arange(t_wait * 60, (t_wait + t_acq[1]) * 60, TS)
    time = (t_1, t_2, t_1, t_2)

    # --- Predict signals
    signal = aorta_liver.predict(time)
    signal = [dc.add_noise(s, s[0] / SNR) for s in signal]

    return time, tuple(signal)


def inverse_ls(time, signal):

    aorta_liver = inverse_ls_model(time, signal)

    # Extract the parameters
    ve, khe_i, khe_f, Th_i, Th_f = aorta_liver.params('ve', 'khe_i', 'khe_f', 'Th_i', 'Th_f')

    kbh_i = (1 - ve) / Th_i if Th_i > 0 else 0
    kbh_f = (1 - ve) / Th_f if Th_f > 0 else 0

    return ve, khe_i, khe_f, kbh_i, kbh_f