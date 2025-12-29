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
def energy_delta_multi(Q: np.ndarray, x: np.ndarray, flip_indices: np.ndarray) -> float:
    """
    Calculates the energy change for flipping multiple bits simultaneously.
    """
    delta_total = 0.0
    
    # We must tentatively flip bits to correctly calculate the interaction terms
    # for subsequent flips in the set.
    for i in flip_indices:
        d = energy_delta(Q, x, i)
        delta_total += d
        x[i] = 1.0 - x[i]
        
    # Revert the flips (restore original state)
    for i in flip_indices:
        x[i] = 1.0 - x[i]
        
    return delta_total

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
    multi_flip_prob: float = 0.1,
    k: int = 2
) -> Tuple[np.ndarray, float, int, np.ndarray, float, float]:
    """
    JIT-optimized execution of multiple MCMC steps (a Markov chain) at a fixed temperature.
    
    Returns:
        Tuple[np.ndarray, float, int, np.ndarray, float, float]: 
            (final_state, final_energy, accepted_moves_count, best_state_in_chain, best_energy_in_chain, energy_std)
    """
    num_vars = len(current_state)
    accepted_moves = 0
    state = current_state.copy()
    energy = current_energy
    
    best_state = state.copy()
    best_energy = energy
    
    # Statistics for adaptive cooling
    sum_energy = 0.0
    sum_energy_sq = 0.0
    
    # Pre-allocate array for single flips to avoid overhead
    single_flip_array = np.zeros(1, dtype=np.int64)
    
    for _ in range(iterations):
        if np.random.rand() > multi_flip_prob:
            # Single flip (Standard)
            flip_index = np.random.randint(0, num_vars)
            delta_E = energy_delta(Q, state, flip_index)
            # Use pre-allocated array or just handle logic directly to avoid array creation overhead
            # Logic branch to avoid array creation for single flip case is faster
            is_multi = False
        else:
            # Multi-flip
            force_multi = True
            # Numba compatible choice without replacement
            # For small k, this loop is fine. For large k, permutation is better.
            # safe for k << N
            flip_indices = np.random.choice(num_vars, k, replace=False)
            delta_E = energy_delta_multi(Q, state, flip_indices)
            is_multi = True
        
        if delta_E < 0 or (temperature > 1e-9 and np.random.rand() < np.exp(-delta_E / temperature)):
            if is_multi:
                for i in flip_indices:
                    state[i] = 1.0 - state[i]
            else:
                state[flip_index] = 1.0 - state[flip_index]
                
            energy += delta_E
            accepted_moves += 1
            
            if energy < best_energy:
                best_energy = energy
                best_state = state.copy()
        
        # Sample energy at every step
        sum_energy += energy
        sum_energy_sq += energy * energy
            
    # Calculate Standard Deviation
    mean_energy = sum_energy / iterations
    variance = (sum_energy_sq / iterations) - (mean_energy * mean_energy)
    
    # Numerical stability check
    if variance < 0:
        variance = 0.0
        
    energy_std = np.sqrt(variance)
            
    return state, energy, accepted_moves, best_state, best_energy, energy_std
