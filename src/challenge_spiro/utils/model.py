import numpy as np
import dcmri as dc


def forward_model(khe_i, khe_f, kbh_i, kbh_f, TS=2.5, t_acq=(45, 45), t_wait=90):

    ve = 0.1

    return dc.AortaLiver2scan(

        # Simulation settings
        dt = 0.5,
        dose_tolerance = 0.1,

        # Injection parameters
        weight = 70,
        agent = 'gadoxetate',
        dose = dc.ca_std_dose('gadoxetate') / 4,
        dose2 = dc.ca_std_dose('gadoxetate') / 4,
        rate = 1,

        # Liver parameters
        ve = 0.1,
        khe_i = khe_i,
        khe_f = khe_f,
        Th_i = (1 - ve) / kbh_i if kbh_i > 0 else 0,
        Th_f = (1 - ve) / kbh_f if kbh_f > 0 else 0,

        # Acquisition parameters
        TS = TS,
        tmax = (t_acq[0] + t_wait + t_acq[1]) * 60,
        BAT = 5 * 60,
        BAT2 = (t_acq[0] + t_wait + 5) * 60,
        field_strength = 3,
        TR = 0.004,
        FA = 20,
        FA2 = 20,

        # Signal parameters
        R10a = 1 / dc.T1(3, 'blood'),
        R10l = 1 / dc.T1(3, 'liver'),
    )


def forward(khe_i, khe_f, kbh_i, kbh_f, TS=2.5, t_acq=(45, 45), t_wait=90, SNR=75, rng=None):

    # --- Initialize the model
    aorta_liver = forward_model(khe_i, khe_f, kbh_i, kbh_f, TS=TS, t_acq=t_acq, t_wait=t_wait)

    # --- Acquired time points
    t_1 = np.arange(0, t_acq[0] * 60, TS)
    t_2 = np.arange((t_acq[0] + t_wait) * 60, (t_acq[0] + t_wait + t_acq[1]) * 60, TS)
    time = (t_1, t_2, t_1, t_2)

    # --- Predict signals
    signal = aorta_liver.predict(time)
    signal = [add_noise(s, s[0] / SNR, rng=rng) for s in signal]

    return time, tuple(signal)



def add_noise(signal, sdev, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise_x = rng.normal(0, sdev, np.size(signal))
    noise_y = rng.normal(0, sdev, np.size(signal))
    signal = np.sqrt((signal+noise_x)**2 + noise_y**2)
    return signal