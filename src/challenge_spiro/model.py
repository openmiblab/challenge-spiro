import numpy as np
import dcmri as dc





def forward_model(TS=2.5, t_acq=(45, 45), t_wait=90, SNR=10):

    # --- Define the model
    model = dc.AortaLiver2scan(

        # Simulation settings
        dt = 0.1,
        tmax = 4 * 60,

        # Injection parameters
        weight = 70,
        agent = 'gadoxetate',
        dose = dc.ca_std_dose('gadoxetate') / 4,
        dose2 = dc.ca_std_dose('gadoxetate') / 4,
        rate = 1,

        # Acquisition parameters
        field_strength = 3,
        t0 = 30,
        TR = 0.004,
        FA = 20,
        FA2 = 20,
        TS = TS,

        # Signal parameters
        R10a = 1 / dc.T1(3, 'blood'),
        R10l = 1 / dc.T1(3, 'liver'),

        # Tissue parameters
        vol = 1000,
    )

    # --- Predict noise-free signals
    t_1 = np.arange(0, t_acq[0] * 60, TS)
    t_2 = np.arange(t_wait * 60, (t_wait* t_acq[1]) * 60, TS)
    signal = model.predict((t_1, t_2, t_1, t_2))

    # --- Add noise
    noisy_signal = [dc.add_noise(signal[i], signal[i][0] / SNR) for i in range(4)]

    return noisy_signal