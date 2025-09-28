# src/quasar_solver/utils.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class SolverResult:
    """
    A data structure to hold the results of a simulated annealing run.

    Using a dataclass provides type hints, attribute access, and immutability.

    Attributes:
        state (np.ndarray): The best binary state vector found.
        energy (float): The energy of the best state.
        history (Dict[str, Any]): A dictionary containing the run's history data,
                                  such as temperatures and energies over time.
                                  This will be empty if the solver was run
                                  with `track_history=False`.
    """
    state: np.ndarray
    energy: float
    history: Dict[str, Any]
    