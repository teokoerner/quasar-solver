# src/quasar_solver/schedules.py

import numpy as np
from typing import Dict, Any

def geometric_cooling(current_temp: float, params: Dict[str, Any]) -> float:
    """
    Calculates the next temperature using a geometric cooling schedule.

    The formula is T_new = alpha * T_current.

    Args:
        current_temp (float): The current temperature.
        params: Expects 'alpha' key in the params dictionary. The cooling factor, typically a value between 0.8 and 0.99.

    Returns:
        float: The next temperature in the schedule.
    
    Raises:
        ValueError: If alpha is not in the range (0, 1).
    """
    try:
        alpha = params['alpha']
    except KeyError:
        raise KeyError("Geometric cooling requires an 'alpha' parameter in schedule_params.")

    
    if not 0.0 < alpha < 1.0:
        raise ValueError("Cooling factor alpha must be between 0 and 1.")
    
    return current_temp * alpha

def adaptive_cooling(current_temp: float, params: Dict[str, Any]) -> float:
    """
    Calculates the next temperature using Huang's adaptive cooling schedule.

    The formula is T_next = T * exp(- (lambda * T) / sigma_E).

    Args:
        current_temp (float): The current temperature.
        params: Expects 'lambda' (float) and 'sigma' (float).
               'lambda': Control parameter for cooling speed.
               'sigma': Standard deviation of energy at current temp.

    Returns:
        float: The next temperature.
    """
    try:
        lam = params['lambda']
        sigma = params['sigma']
    except KeyError as e:
        raise KeyError(f"Huang cooling requires parameter: {e}")
        
    if sigma <= 1e-9:
        # If sigma is 0 (or very small), the system is frozen or flat.
        # We can either stop cooling or cool very fast.
        # Huang's original paper suggests handling this, often by ending or standard cooling.
        # For robustness, let's fall back to a gentle decay, effectively T_next = T * 0.99
        # effectively assuming a small default ratio.
        return current_temp * 0.99
        
    delta = (lam * current_temp) / sigma
    return current_temp * np.exp(-delta)