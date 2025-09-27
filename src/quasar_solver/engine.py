# src/quasar_solver/engine.py

import numpy as np
from typing import Tuple

from .qubo import QUBO

def run_mcmc_step(
    qubo: QUBO,
    current_state: np.ndarray,
    current_energy: float,
    temperature: float,
) -> Tuple[np.ndarray, float]:
    """
    Performs a single, efficient Markov Chain Monte Carlo (MCMC) step.

    This function implements the core logic of the Simulated Annealing algorithm
    for a single iteration at a fixed temperature. It proposes a random move
    (a single bit-flip) and decides whether to accept it based on the
    Metropolis-Hastings criterion.

    Args:
        qubo (QUBO): The QUBO problem instance.
        current_state (np.ndarray): The current binary state vector of the system.
        current_energy (float): The energy of the current state.
        temperature (float): The current temperature (T > 0).

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the new state and its
                                  corresponding energy. If the move is rejected,
                                  this will be the original state and energy.
    """
    # 1. Propose a move: Select a random bit to flip
    flip_index = np.random.randint(0, qubo.num_variables)

    # 2. Calculate the change in energy for this move
    delta_E = qubo.energy_delta(current_state, flip_index)

    # 3. Decide whether to accept the move using a single check
    # The acceptance probability is min(1, exp(-ΔE/T)).
    # If ΔE < 0, exp(-ΔE/T) > 1, so the move is always accepted.
    # Otherwise, it's accepted with probability exp(-ΔE/T).
    if delta_E < 0 or (temperature > 1e-9 and np.random.rand() < np.exp(-delta_E / temperature)):
        # Accept the move: update state and energy
        new_state = current_state.copy()
        new_state[flip_index] = 1 - new_state[flip_index]
        return new_state, current_energy + delta_E

    # Reject the move: return the original state and energy
    return current_state, current_energy
