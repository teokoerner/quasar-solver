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

class ModelConverter:
    """
    Utility class to interface between high-level modeling libraries 
    (like PyQUBO) and the Quasar Solver's NumPy-based engine.
    """

    @staticmethod
    def from_pyqubo(model, feed_dict):
        """
        Converts a PyQUBO model into a NumPy matrix and an energy offset.
        
        Args:
            model: A compiled pyqubo.Model object.
            feed_dict: A dictionary of values for placeholders in the model (usually penalty values).
            
        Returns:
            tuple: (Q_matrix, offset, label_map)
                - Q_matrix: np.ndarray of shape (N, N)
                - offset: float representing the constant energy shift
                - label_map: list of variable names corresponding to indices
        """
        # 1. Extract the QUBO dict and offset from PyQUBO
        # The dict keys are tuples of variable names (e.g., ('x[0]', 'x[1]'))
        qubo_dict, offset = model.to_qubo(feed_dict=feed_dict)
        
        # 2. Map variable names to unique integer indices
        # This ensures we maintain a consistent order in the matrix
        variables = sorted(list(model.variables))
        var_to_idx = {var: i for i, var in enumerate(variables)}
        num_vars = len(variables)
        
        # 3. Initialize the Q matrix
        Q = np.zeros((num_vars, num_vars))
        
        # 4. Fill the matrix
        for (u, v), bias in qubo_dict.items():
            i, j = var_to_idx[u], var_to_idx[v]
            # QUBO matrices are usually upper-triangular or symmetric.
            # We will store them as-is from the dict.
            Q[i, j] = bias
            
        return Q, offset, variables

    @staticmethod
    def decode_solution(binary_vector, label_map):
        """
        Maps the raw binary output from the solver back to PyQUBO variable names.
        """
        return {label: int(val) for label, val in zip(label_map, binary_vector)}
    