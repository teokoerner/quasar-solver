# src/quasar_solver/utils.py

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SolverResult:
    """
    A data structure to hold the results of a simulated annealing run.

    Using a dataclass provides type hints, attribute access, and immutability.

    Attributes:
        state (np.ndarray): The best binary state vector found.
        energy (float): The energy of the best state.
        history (List[float]): A list of the best energy found at each
                               temperature step, tracking convergence.
    """
    state: np.ndarray
    energy: float
    history: List[float]