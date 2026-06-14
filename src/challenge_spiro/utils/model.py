"""
Forward Modeling, Noise Simulation, and Visualization Pipeline
=============================================================

This module provides the core forward-modeling framework for predicting 
tissue-specific signal responses based on an aorta-liver kinetic engine. 
It functions as the numerical core of the simulation pipeline, translating 
physiological parameters into dynamic time-series curves.

Key capabilities include:
1. **Signal Prediction**: Instantiating and executing the forward model over 
   multi-phase experimental time grids.
2. **Stochastic Noise Simulation**: Injecting Gaussian/Rician experimental noise 
   modeled to achieve a target Signal-to-Noise Ratio (SNR) of 50.
3. **Visual Quality Control**: Wrapping low-level model visualization libraries 
   to overlay ground-truth predictions onto synthetic or real dataset points.

Dependencies
------------
- numpy
- dcmri

Usage Example
-------------
>>> import numpy as np
>>> from challenge_spiro.utils.dro import generate_subject
>>> from challenge_spiro.utils.model import forward, plot
>>> # Generate input states
>>> params, time_grids = generate_subject()
>>> # Compute forward pass with noise
>>> signals = forward(params, time_grids)
>>> # Plot result
>>> plot(params, time_grids, signals)
"""

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

# --- Helper function
def initialize_model(params, time, **settings):
    tmax = np.concatenate(time).max() + 60
    return dc.AortaLiver2scan(tmax=tmax, **(params | settings))

# --- Helper funcion
def add_noise(signal, sdev, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise_x = rng.normal(0, sdev, np.size(signal))
    noise_y = rng.normal(0, sdev, np.size(signal))
    signal = np.sqrt((signal + noise_x)**2 + noise_y**2)
    return signal


def plot(params, time, signal, **settings):
    """
    Visualize the simulated forward model predictions against reference signal data.

    This function instantiates the aorta-liver kinetic model using the provided
    physiological parameters and invokes its internal plotting routine to overlay
    the model's predicted signal curves onto the provided measurement data. It is
    typically used for qualitative model verification and visual quality control.

    Parameters
    ----------
    params : dict
        A dictionary containing the kinetic, signal, and physiological parameters
        required to properly configure and instantiate the model.
    time : tuple of numpy.ndarray
        A 4-element tuple `(tacq_1, tacq_2, tacq_1, tacq_2)` representing the
        measurement time grids (in seconds) corresponding to each acquisition phase.
    signal : tuple of numpy.ndarray
        A 4-element tuple of arrays containing the reference, observed, or noisy
        signal data to be plotted alongside the model predictions.
    **settings : dict, optional
        Additional keyword arguments passed directly to `initialize_model` to
        configure experimental conditions.

    Returns
    -------
    None
        The function displays a visualization plot directly (or modifies the 
        active matplotlib figure state) and returns nothing.

    Examples
    --------
    >>> import numpy as np
    >>> from challenge_spiro.utils.dro import generate_subject
    >>> from challenge_spiro.utils.model import forward, plot
    >>> rng = np.random.default_rng(42)
    >>> params, time_grids = generate_subject(rng)
    >>> noisy_signals = forward(params, time_grids, rng=rng)
    >>> # Generate and display the evaluation plot
    >>> plot(params, time_grids, noisy_signals)
    """
    # --- Initialize the model
    aorta_liver = initialize_model(params, time, **settings)

    # --- Plot prediction against data
    aorta_liver.plot(time, signal)


def forward(params, time, rng=None, **settings):
    """
    Simulate the forward model to predict noisy multi-phase measurement signals.

    This function instantiates an aorta-liver kinetic model using the provided 
    physiological parameters, predicts the baseline signal curves across 
    multiple acquisition phases, applies patch corrections for known sampling bugs, 
    and adds Rician noise based on a target Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    params : dict
        A dictionary containing the generated kinetic, signal, and physiological 
        parameters (e.g., 'S0a', 'S0l', 'BAT', 've') required to configure the model.
    time : tuple of numpy.ndarray
        A 4-element tuple `(tacq_1, tacq_2, tacq_1, tacq_2)` representing the 
        measurement time grids (in seconds) for each acquisition phase.
    rng : numpy.random.Generator, optional
        A specific NumPy random number generator instance to ensure reproducible 
        noise addition. If None, the random behavior inside `add_noise` is unseeded.
    **settings : dict, optional
        Additional keyword arguments passed directly to `initialize_model` to 
        configure experimental conditions.

    Returns
    -------
    signal : tuple of numpy.ndarray
        A 4-element tuple containing the simulated, corrected, and noisy 
        signal time-series arrays corresponding to each input time grid phase.

    Notes
    -----
    **Software Bug Workaround:**
    Due to a known indexing/sampling defect in the underlying `dcmri v0.6` library, 
    edge points at phase transitions are incorrectly calculated. This function 
    manually overrides those edge discontinuities using a nearest-neighbor patch:
    
    - Phase 1 (index 0 & 2): The last point (`[-1]`) is set to the second-to-last point (`[-2]`).
    - Phase 2 (index 1 & 3): The first point (`[0]`) is set to the second point (`[1]`).

    **Noise Model:**
    A constant Signal-to-Noise Ratio (SNR) of 50 is assumed. The noise standard 
    deviation ($\sigma$) is derived globally per phase from the initial baseline 
    signal value ($S_0$), calculated as:
    
    .. math:: \sigma = \frac{\text{signal}[0]}{\text{SNR}}

    Examples
    --------
    >>> import numpy as np
    >>> from challenge_spiro.utils.dro import generate_subject
    >>> from challenge_spiro.utils.model import forward
    >>> rng = np.random.default_rng(42)
    >>> params, time_grids = generate_subject(rng)
    >>> signal_curves = forward(params, time_grids, rng=rng)
    >>> type(signal_curves)
    <class 'tuple'>
    >>> len(signal_curves)
    4
    """

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



