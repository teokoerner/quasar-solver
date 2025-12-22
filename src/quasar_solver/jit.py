# src/quasar_solver/jit.py

import numpy as np
from numba import njit
from typing import Tuple

@njit
def energy_delta(Q: np.ndarray, x: np.ndarray, flip_index: int) -> float:
    """
    JIT-optimized calculation of the change in energy from flipping a single bit.
    
    ΔE = (1 - 2*x_i) * (Q_ii + sum_{j!=i} (Q_ij + Q_ji) * x_j)
    """
    # Use a loop-based dot product for robust Numba/Scipy compatibility
    row_sum = 0.0
    col_sum = 0.0
    for j in range(len(x)):
        row_sum += Q[flip_index, j] * x[j]
        col_sum += Q[j, flip_index] * x[j]
    
    direction = 1.0 - 2.0 * x[flip_index]
    
    # Q[flip_index, flip_index] is added twice in row_sum + col_sum (when j == flip_index)
    # So we subtract it twice: row_sum + col_sum - 2 * Q[i,i] * x[i]
    # Then add the diagonal term Q[i,i] once: (Q[i,i] + row_sum + col_sum - 2 * Q[i,i] * x[i])
    delta_E = direction * (Q[flip_index, flip_index] + row_sum + col_sum - 2.0 * Q[flip_index, flip_index] * x[flip_index])
    
    return float(delta_E)

@njit
def mcmc_step(
    Q: np.ndarray,
    current_state: np.ndarray,
    current_energy: float,
    temperature: float,
    num_variables: int,
) -> Tuple[np.ndarray, float]:
    """
    JIT-optimized execution of a single MCMC step.
    """
    # 1. Propose a move: Select a random bit to flip
    flip_index = np.random.randint(0, num_variables)

    # 2. Calculate the change in energy for this move
    delta_E = energy_delta(Q, current_state, flip_index)

    # 3. Decide whether to accept the move
    if delta_E < 0 or (temperature > 1e-9 and np.random.rand() < np.exp(-delta_E / temperature)):
        # Accept the move: update state and energy
        new_state = current_state.copy()
        new_state[flip_index] = 1.0 - new_state[flip_index]
        return new_state, current_energy + delta_E

    # Reject the move: return original state and energy
    return current_state, current_energy

@njit
def mcmc_chain(
    Q: np.ndarray,
    current_state: np.ndarray,
    current_energy: float,
    temperature: float,
    iterations: int,
) -> Tuple[np.ndarray, float, int]:
    """
    JIT-optimized execution of multiple MCMC steps (a Markov chain) at a fixed temperature.
    This provides massive performance gains by avoiding Python loop overhead and 
    repetitive JIT function call overhead.
    
    Returns:
        Tuple[np.ndarray, float, int]: (new_state, new_energy, accepted_moves_count)
    """
    num_vars = len(current_state)
    accepted_moves = 0
    state = current_state.copy()
    energy = current_energy
    
    for _ in range(iterations):
        flip_index = np.random.randint(0, num_vars)
        delta_E = energy_delta(Q, state, flip_index)
        
        if delta_E < 0 or (temperature > 1e-9 and np.random.rand() < np.exp(-delta_E / temperature)):
            state[flip_index] = 1.0 - state[flip_index]
            energy += delta_E
            accepted_moves += 1
            
    return state, energy, accepted_moves
