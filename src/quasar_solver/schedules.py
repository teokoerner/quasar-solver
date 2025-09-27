# src/quasar_solver/schedules.py

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