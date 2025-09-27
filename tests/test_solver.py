# tests/test_solver.py

import numpy as np
import pytest

from quasar_solver.qubo import QUBO
from quasar_solver.solver import Solver

def test_simple_qubo_solve():
    """
    Tests the full solver pipeline on a simple 3-variable QUBO
    with a known optimal solution.
    """
    # 1. Arrange: Define a simple problem
    # The optimal solution is [1, 0, 1] with energy -2.
    q_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    optimal_state = np.array([1, 0, 1])
    optimal_energy = -2.0

    qubo = QUBO(q_matrix)

    # Use parameters that are likely to find the solution quickly
    solver = Solver(
        qubo=qubo,
        initial_temp=10.0,
        final_temp=0.05,
        iterations_per_temp=200,
        schedule_params={'alpha': 0.95}
    )

    # 2. Act: Run the solver
    result = solver.solve()

    # 3. Assert: Check if the result matches the known solution
    # The SA algorithm is heuristic, but for a simple problem like this,
    # it should reliably find the optimum.
    assert result.energy == optimal_energy
    np.testing.assert_array_equal(result.state, optimal_state)