# src/quasar_solver/engine.py

import numpy as np
from typing import Tuple
from .qubo import QUBO
from . import jit

def run_mcmc_step(
    qubo: QUBO,
    current_state: np.ndarray,
    current_energy: float,
    temperature: float,
) -> Tuple[np.ndarray, float]:
    """Performs a single, efficient Markov Chain Monte Carlo (MCMC) step using Numba njit.

    This function implements the core logic of the Simulated Annealing algorithm
    for a single iteration at a fixed temperature. It proposes a random move
    (a single bit-flip) and decides whether to accept it based on the
    Metropolis-Hastings criterion.

    Parameters
    ----------
    qubo : QUBO
        The QUBO problem instance.

    current_state : np.ndarray
        The current binary state vector of the system.

    current_energy : float
        The energy of the current state.

    temperature : float
        The current temperature (T > 0).

    Returns
    -------
    tuple
        A tuple containing (new_state, new_energy).
        If the move is rejected, this will be the original state and energy.
    """
    return jit.mcmc_step(
        qubo.Q, current_state, current_energy, temperature, qubo.num_variables
    )
