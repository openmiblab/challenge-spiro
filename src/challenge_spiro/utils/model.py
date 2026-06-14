import numpy as np
import dcmri as dc

PARAMS = [
    'S0a', 
    'S0l', 
    'S02a', 
    'S02l', 
    'BAT', 
    'BAT2', 
    'CO', 
    'Thl', 
    'Dhl', 
    'To', 
    'Eo', 
    'Toe', 
    'Eb', 
    've', 
    'Te', 
    'De', 
    'khe_i', 
    'khe_f', 
    'Th_i', 
    'Th_f',
]

def initialize_model(params, time, **settings):
    tmax = np.concatenate(time).max() + 60

    return dc.AortaLiver2scan(tmax=tmax, **(params | settings))

def plot(params, time, signal, **settings):

    # --- Initialize the model
    aorta_liver = initialize_model(params, time, **settings)

    # --- Plot prediction against data
    aorta_liver.plot(time, signal)


def forward(params, time, rng=None, **settings):

    # --- Initialize the model
    aorta_liver = initialize_model(params, time, **settings)

    # --- Predict signals at given time points
    signal = aorta_liver.predict(time)

    # --- Fix incorrectly sampled points (dcmri v0.6 software bug)
    signal[0][-1] = signal[0][-2]
    signal[1][0] = signal[1][1]
    signal[2][-1] = signal[2][-2]
    signal[3][0] = signal[3][1]

    # --- Add noise
    SNR = 50
    signal = [add_noise(s, s[0] / SNR, rng=rng) for s in signal]

    return tuple(signal)



def add_noise(signal, sdev, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise_x = rng.normal(0, sdev, np.size(signal))
    noise_y = rng.normal(0, sdev, np.size(signal))
    signal = np.sqrt((signal + noise_x)**2 + noise_y**2)
    return signal