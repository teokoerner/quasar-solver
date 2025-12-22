# src/quasar_solver/qubo.py

import numpy as np
from . import jit

class QUBO:
    """
    Represents a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    This class encapsulates the QUBO matrix Q and provides efficient methods
    for calculating the energy of a given binary state vector. The energy E
    is defined by the quadratic form E = x^T * Q * x, where x is a
    column vector of binary variables {0, 1}.

    Attributes:
        Q (np.ndarray): The square, symmetric QUBO matrix.
        num_variables (int): The number of variables in the problem (N).
    """

    def __init__(self, Q: np.ndarray):
        """
        Initializes the QUBO problem.

        Args:
            Q (np.ndarray): A 2D NumPy array representing the QUBO matrix.
                            It must be a square matrix.

        Raises:
            ValueError: If Q is not a square 2D NumPy array.
        """
        if not isinstance(Q, np.ndarray) or Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square 2D NumPy array.")

        self.Q = Q
        self.num_variables = Q.shape[0]

    def energy(self, x: np.ndarray) -> float:
        """
        Calculates the total energy of a state vector using E = x^T * Q * x.

        This is an O(N^2) operation and should primarily be used for
        verification or calculating the energy of an initial state.

        Args:
            x (np.ndarray): A 1D NumPy array of binary variables {0, 1}.

        Returns:
            float: The calculated energy for the given state.

        Raises:
            ValueError: If x is not a 1D array of the correct length.
        """
        if x.shape != (self.num_variables,):
            raise ValueError(f"State vector x must have length {self.num_variables}.")
        
        # Efficiently compute x.T @ Q @ x
        return float(x.T @ self.Q @ x)

    def energy_delta(self, x: np.ndarray, flip_index: int) -> float:
        """
        Calculates the change in energy from flipping a single bit using Numba njit for 
        runtime acceleration. 

        This method provides an efficient O(N) calculation for the energy
        difference (ΔE) if the bit at `flip_index` were to be flipped.
        The change is given by: ΔE = (1 - 2*x_i) * (Q_ii + sum_{j!=i} (Q_ij + Q_ji) * x_j).

        Args:
            x (np.ndarray): The current 1D binary state vector.
            flip_index (int): The index of the bit to be flipped.

        Returns:
            float: The change in energy (ΔE) that would result from the flip.
        """
        return jit.energy_delta(self.Q, x, flip_index)
    