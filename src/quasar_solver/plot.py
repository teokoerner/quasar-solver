# src/quasar_solver/plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")


def plot_energy_vs_temp(history: Dict[str, Any], save_path: str | None = None):
    """
    Plots the energy of the system as a function of temperature.

    This visualization provides a physical intuition for the annealing process,
    showing the noisy, high-energy "liquid" phase at high temperatures and the
    stable, low-energy "frozen" phase at low temperatures.

    It generates two plots on the same axes:
    1. A scatter plot of the 'current energy' at each temperature step.
    2. A line plot of the 'best-so-far energy' found during the run.

    Args:
        history (Dict[str, Any]): A dictionary from a SolverResult object.
            It must contain the keys 'temperatures', 'current_energies',
            and 'best_energies'.
        save_path (str | None, optional): If provided, the plot is saved to
            this file path. Defaults to None.

    Raises:
        ValueError: If the history dictionary does not contain the required keys,
                    which occurs if the solver was run with `track_history=False`.
    """
    required_keys = ['temperatures', 'current_energies', 'best_energies']
    if not all(key in history for key in required_keys):
        raise ValueError(
            "History data not found for plotting. "
            "Ensure the solver was run with `track_history=True`."
        )

    fig, ax = plt.subplots()

    # Scatter plot for the noisy current energy to show exploration
    ax.scatter(
        history['temperatures'],
        history['current_energies'],
        label='Current Energy',
        s=5,  # Small marker size for a dense look
        alpha=0.7,
        color='steelblue'
    )

    # Line plot for the best-so-far energy to show convergence
    ax.plot(
        history['temperatures'],
        history['best_energies'],
        label='Best-so-far Energy',
        color='firebrick',
        linewidth=2.0
    )

    ax.set_xlabel('Temperature ($T$)')
    ax.set_ylabel('Energy ($E$)')
    ax.set_title('Energy vs. Temperature during Annealing')

    # Use a logarithmic scale for the x-axis and reverse it
    ax.set_xscale('log')
    ax.invert_xaxis()  # Places high temperatures on the left

    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy vs. Temperature plot saved to {save_path}")

    plt.show()
