# src/quasar_solver/utils.py

import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

def estimate_parameters(Q: np.ndarray) -> Tuple[float, float, int, float, int]:
    """Estimates optimal solver parameters based on QUBO properties.

    Parameters
    ----------
    Q : np.ndarray
        The QUBO matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - initial_temp (float)
        - final_temp (float)
        - iterations_per_temp (int)
        - cooling_rate (float)
        - num_reads (int)
        - schedule (str): 'geometric' or 'adaptive'
    """
    N = Q.shape[0]
    abs_q = np.abs(Q)
    density = np.count_nonzero(Q) / (N * N)
    
    # Estimate Energy Scale (Max Delta E) using row sums as a proxy for potential flip impact
    max_delta = np.max(np.sum(abs_q, axis=1)) 
    
    # Estimate Minimum Gap (Min Delta E). Zeroes are filtered out to avoid min_delta being 0 
    # for sparse matrices
    non_zero_elements = abs_q[abs_q > 0]
    min_delta = np.min(non_zero_elements) if non_zero_elements.size > 0 else 0.1

    # Apply Heuristics
    initial_temp = -max_delta / np.log(0.8) # 80% initial acceptance
    final_temp = -min_delta / np.log(0.0001) # Near-zero acceptance
    iterations_per_temp = N * 10 
    
    # Cooling rate heuristic
    cooling_rate = 0.99 if N > 100 else 0.95
    
    # Num reads heuristic
    cpu_count = os.cpu_count() or 1
    num_reads = max(cpu_count, int(np.sqrt(N)))

    # Calculate Coefficient of Variation (CV) of the weights
    # High CV implies a 'spiky' landscape with strong local traps
    non_zero_q = Q[Q != 0]
    cv = np.std(non_zero_q) / np.abs(np.mean(non_zero_q)) if len(non_zero_q) > 0 else 0
    
    # The Heuristic Rule
    # If the problem is dense (>40%) OR has high weight variance (>1.5), 
    # use Huang. Otherwise, stick to Geometric for speed.
    if density > 0.4 or cv > 1.5:
        schedule = 'adaptive'
    else:
        schedule = 'geometric'

    return initial_temp, final_temp, iterations_per_temp, cooling_rate, num_reads, schedule


@dataclass(frozen=True)
class SolverResult:
    """A data structure to hold the results of a simulated annealing run.

    Using a dataclass provides type hints, attribute access, and immutability.

    Attributes
    ----------
    state : np.ndarray
        The best binary state vector found.

    energy : float
        The energy of the best state.

    history : dict
        A dictionary containing the run's history data (temperatures, energies, etc.).
        Empty if `track_history=False`.
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
        """Converts a PyQUBO model into a NumPy matrix and an energy offset.
        
        Parameters
        ----------
        model : pyqubo.Model
            A compiled pyqubo.Model object.

        feed_dict : dict
            A dictionary of values for placeholders in the model (usually penalty values).
            
        Returns
        -------
        tuple
            A tuple containing:
            - Q_matrix (np.ndarray): shape (N, N)
            - offset (float): constant energy shift
            - label_map (list): variable names corresponding to indices
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
        """Maps the raw binary output from the solver back to PyQUBO variable names.

        Parameters
        ----------
        binary_vector : array-like
            The binary solution vector.

        label_map : list of str
            List of variable names corresponding to the indices in `binary_vector`.

        Returns
        -------
        dict
            Dictionary mapping variable names to their binary values (0 or 1).
        """
        return {label: int(val) for label, val in zip(label_map, binary_vector)}
    