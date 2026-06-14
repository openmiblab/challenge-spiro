"""
Analytical Parameter Initialization & Initial Value Profiler
============================================================

This module provides an optimized, non-iterative initialization routine 
(inverse solution mapping) for the SPIRO Challenge. Instead of running 
heavy numerical optimizations, it applies physical heuristics and signal equations 
to extract fast baseline approximations from observed time-series inputs.

The solution is intended as a normative reference for defining a score. 
It has no practical value as it returns constants for the key parameters.

The estimation architecture acts as a heuristic inverse solver by evaluating 
signal peaks to determine timing offsets, and utilizing Steady-State (SS) relaxation 
models to isolate native signal scaling factors.

Architecture
------------
1. **Model Blueprint Setup**: Provisions a raw `dc.AortaLiver2scan` layout to fetch 
   default baseline variables.
2. **BAT Heuristic Extraction**: Identifies Bolus Arrival Times (`BAT`, `BAT2`) 
   by tracking the index of maximum signal intensity over the aorta phases.
3. **Signal Scaling Factors**: Employs an analytical T1 signal model to calculate 
   baseline properties (`S0a`, `S02a`, `S0l`, `S02l`) via pre-bolus averaging 
   calibrated against steady-state reference equations.

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
    Estimate baseline physiological and signal parameters using direct empirical heuristics.

    This function avoids numerical model fitting and optimization entirely. Instead, 
    it extracts parameters deterministically from structural properties of the signal 
    data: it handles temporal parameters (`BAT`) via peak tracking, and baseline signal 
    gains (`S0`) by parsing the steady-state pre-contrast signal floor through 
    a calibrated T1 signal equation.

    Parameters
    ----------
    time : tuple of numpy.ndarray
        A 4-element tuple `(tacq_1, tacq_2, tacq_1, tacq_2)` representing absolute 
        measurement timelines (in seconds) for the respective data streams.
    signal : tuple of numpy.ndarray
        A 4-element tuple of arrays containing observed dynamic signal curves 
        (aorta and liver for Phases 1 & 2).
    **settings : dict, optional
        Configuration overrides passed to the underlying model and signal solvers. 
        Supported optional keys:
        
        - 'FA' : float, default 20
            Flip angle (in degrees) for the imaging sequence.
        - 'TR' : float, default 0.004
            Repetition time (in seconds) for the imaging sequence.

    Returns
    -------
    results : dict
        A compiled parameter dictionary mapped across `PARAMS`. Missing variables 
        retain default `dcmri` library baselines, while timing and scaling parameters 
        reflect subject-specific calculations.

    Notes
    -----
    **Temporal Peak Estimation:**
    Bolus arrival times are assumed to occur at maximum observed signal points:
    
    - `BAT`: Time of absolute maximum peak in Phase 1 Aorta.
    - `BAT2`: Relative time delay of maximum peak in Phase 2 Aorta, relative to 
      Phase 2 start time: :math:`t_{\\max} - t_{start}`.

    **Steady-State Signal Model:**
    Pre-contrast T1 relaxation rates are standard constants computed at $3\text{ Tesla}$ 
    field strength for blood and liver tissue. Baseline gains (`S0`) are derived 
    by averaging the first 5 frames (excluding index 0) and scaling them against the 
    analytical steady-state ('SS') signal equation output:
    
    .. math:: S_{0} = \\frac{\\text{mean}(\\mathbf{s}_{pre\\_bolus})}{\\text{Signal}_{SS}(R_{10}, 1, \\text{FA}, \\text{TR})}

    Examples
    --------
    >>> # Get some test data
    >>> import dcmri as dc
    >>> from challenge_spiro.utils.score import parse_data
    >>> data_file = dc.fetch('tristan_humans_healthy_rifampicin')
    >>> time_grids, signal_curves, settings = parse_data(data_file, '002', 'control')
    >>> # Fit the model
    >>> from challenge_spiro.solutions.normative import inverse
    >>> fitted_params = inverse(time_grids, signal_curves, **settings)
    >>> print(f"Estimated Initial Uptake Rate (ve): {fitted_params['khe_i']:.4f}")
    """

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