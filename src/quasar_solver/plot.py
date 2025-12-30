
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Optional

# --- Plotting Configuration ---
def set_plot_style():
    """Sets a professional scientific plotting style."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'figure.figsize': (8, 5),
        'font.family': 'sans-serif',
    })

set_plot_style()

def _save_and_show(fig: plt.Figure, save_path: Optional[str] = None):
    """Helper to save and/or show the plot.

    Parameters
    ----------
    fig : plt.Figure
        The figure object to handle.
        
    save_path : str, optional
        Path where the figure should be reserved.
    """
    plt.tight_layout()
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_energy_convergence(history: Dict[str, Any], save_path: Optional[str] = None):
    """Plots the energy convergence over iterations (Time).
    
    Displays:
    1. Current Energy (noisy)
    2. Best-So-Far Energy (monotonic)
    
    Parameters
    ----------
    history : dict
        Solver history containing 'iterations', 'current_energies', 'best_energies'.
        
    save_path : str, optional
        Optional path to save the figure.
    """
    required_keys = ['current_energies', 'best_energies']
    if not all(key in history for key in required_keys):
        raise ValueError(f"History missing required keys: {required_keys}")
    
    # Fallback if iterations not present (e.g. from older solver runs)
    iterations = history.get('iterations', np.arange(len(history['current_energies'])))
    
    fig, ax = plt.subplots()
    
    # Plot Current Energy (Noisy) with lower opacity
    ax.plot(iterations, history['current_energies'], 
            label='Current Energy', 
            color='steelblue', 
            alpha=0.6, 
            linewidth=1)
            
    # Plot Best-So-Far Energy (Monotonic) with prominence
    ax.plot(iterations, history['best_energies'], 
            label='Best-So-Far Energy', 
            color='firebrick', 
            linewidth=2.5)
    
    ax.set_xlabel('Iterations (Time)')
    ax.set_ylabel('Energy (Objective Value)')
    ax.set_title('Energy Convergence Trajectory')
    ax.legend()
    
    _save_and_show(fig, save_path)

def plot_acceptance_probability(history: Dict[str, Any], save_path: Optional[str] = None):
    """Plots the acceptance probability of uphill moves against Temperature.
    
    Helps diagnose if the cooling schedule allows for sufficient exploration 
    (high acceptance at start) and proper exploitation (low acceptance at end).
    
    Parameters
    ----------
    history : dict
        Solver history containing 'temperatures', 'acceptance_rates'.
        
    save_path : str, optional
        Optional path to save the figure.
    """
    required_keys = ['temperatures', 'acceptance_rates']
    if not all(key in history for key in required_keys):
        raise ValueError(f"History missing required keys: {required_keys}")
    
    temps = history['temperatures']
    rates = history['acceptance_rates']
    
    fig, ax = plt.subplots()
    
    ax.plot(temps, rates, 'o-', color='seagreen', markersize=4, label='Acceptance Probability')
    
    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel('Acceptance Probability ($P_{accept}$)')
    ax.set_title('Acceptance Probability vs. Temperature')
    
    # Log scale for temperature is standard for annealing schedules
    ax.set_xscale('log')
    ax.invert_xaxis()  # High temp (start) on the left to show "cooling" L->R
    
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    
    _save_and_show(fig, save_path)

def plot_temperature_schedule(history: Dict[str, Any], save_path: Optional[str] = None):
    """Plots the temperature cooling schedule over iterations.
    
    Parameters
    ----------
    history : dict
        Solver history containing 'iterations', 'temperatures'.
        
    save_path : str, optional
        Optional path to save the figure.
    """
    required_keys = ['temperatures']
    if not all(key in history for key in required_keys):
        raise ValueError(f"History missing required keys: {required_keys}")
    
    iterations = history.get('iterations', np.arange(len(history['temperatures'])))
    temps = history['temperatures']
    
    fig, ax = plt.subplots()
    
    ax.plot(iterations, temps, '-', color='purple', linewidth=2, label='Temperature')
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Temperature ($T$)')
    ax.set_title('Temperature Cooling Schedule')
    
    ax.legend()
    
    _save_and_show(fig, save_path)
