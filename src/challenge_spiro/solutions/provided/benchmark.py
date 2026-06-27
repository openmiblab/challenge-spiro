"""
Baseline Inverse Solution & Model Training Module
================================================

This module implements a standard baseline parameter estimation pipeline (inverse model) 
for the SPIRO Challenge. It utilizes the `dcmri` library to fit an aorta-liver 
dual-scan pharmacokinetic model to dynamic signal data.

The module provides a functional wrapper that isolates time-series optimization 
from model initialization, handling maximum timeline boundaries, executing model training, 
and converting fitted parameters into a standard flat dictionary representation.

Architecture
------------
1. **Model Instantiation**: Automatically calculates a unified temporal horizon (`tmax`) 
   from split-session time grids and provisions a `dc.AortaLiver2scan` solver instance.
2. **Optimization Pass**: Trains model constants against observed data by minimizing 
   residual error up to a defined parameter tolerance (`xtol`).
3. **Parameter Extraction**: Selects and maps optimized values according to the global 
   challenge schema (`PARAMS`).

Dependencies
------------
- numpy
- dcmri (>= 0.6)
- challenge_spiro.utils.model.PARAMS
"""


import numpy as np
import dcmri as dc

from challenge_spiro.utils.model import PARAMS


def inverse(time, signal, **settings) -> dict:
    """
    Fit a two-scan aorta-liver pharmacokinetic model to estimate physiological parameters.

    This function sets up a dual-session parameter estimator, initializes the 
    `dcmri` computational engine to span the total experiment duration, executes 
    non-linear model fitting via optimization loops, and returns a key-value map 
    of the estimated ground-truth properties.

    Parameters
    ----------
    time : tuple of numpy.ndarray
        A 4-element tuple `(tacq_1, tacq_2, tacq_1, tacq_2)` representing the absolute 
        measurement time grids (in seconds) spanning both acquisition phases.
    signal : tuple of numpy.ndarray
        A 4-element tuple of arrays containing observed tissue/blood time-series 
        measurement curves (aorta and liver for Phases 1 & 2).
    **settings : dict, optional
        Experimental metadata and sequence criteria passed through to the underlying 
        `dc.AortaLiver2scan` model constructor (e.g., patient weights, injection doses).

    Returns
    -------
    results : dict
        A mapped registry of parameter terms containing estimated numerical outputs 
        (e.g., 've', 'BAT', 'CO'). Keys are explicitly filtered and ordered based 
        on the global reference `PARAMS` tracking array.

    Notes
    -----
    **Temporal Constraints:**
    The internal time horizon (`tmax`) is automatically calculated dynamically per subject 
    as the global maximum across all time segments, plus a 60-second safety padding:
    
    .. math:: t_{max} = \\max(\\mathbf{t}_{all}) + 60

    **Solver Convergence:**
    The iterative non-linear solver operates with a parameter variation stopping 
    tolerance (`xtol`) set strictly to $10^{-3}$. Logging data is dumped to the 
    console with an explicit optimization verbosity level of 2.

    Examples
    --------
    >>> # Get some test data
    >>> import dcmri as dc
    >>> from challenge_spiro.utils.score import parse_data
    >>> data_file = dc.fetch('tristan_humans_healthy_rifampicin')
    >>> time_grids, signal_curves, settings = parse_data(data_file, '002', 'control')
    >>> # Fit the model
    >>> from challenge_spiro.solutions.benchmark import inverse
    >>> fitted_params = inverse(time_grids, signal_curves, **settings)
    >>> print(f"Estimated Initial Uptake Rate (ve): {fitted_params['khe_i']:.4f}")
    """
    # --- Initialize model
    tmax = np.concatenate(time).max() + 60
    aorta_liver = dc.AortaLiver2scan(tmax=tmax, **settings)

    # --- Train model
    aorta_liver.train(time, signal, verbose=2, xtol=1e-3)

    # --- Return fitted parameters
    results = {k: aorta_liver.params(k) for k in PARAMS}

    return results
