# Quasar Solver

**QUASAR**: **QU**BO-based **A**nnealing **S**earch **A**nd **R**esolution

A from-scratch Simulated Annealing (SA) solver in Python, designed for solving **Quadratic Unconstrained Binary Optimization (QUBO)** problems.

## About This Project

Quasar is primarily a personal development project focused on the intersection of physics, mathematics, and computer science. The name is inspired by the astronomical phenomenon—just as a quasar is an extremely luminous object in the vastness of space, this project aims to be a tool for finding optimal solutions in a vast search space.

## Getting Started

*(Instructions to be added once the project is installable.)*

## Basic Usage

```python
import numpy as np
from quasar_solver import solve_qubo

# Define a simple QUBO problem (e.g., a MAX-CUT instance)
Q = np.array([
    [-1, 2, 2, 0],
    [0, -1, 2, 2],
    [0, 0, -1, 2],
    [0, 0, 0, -1]
])

# Run the solver
solver = Solver(
    qubo=Q,
    initial_temp=10.0,
    final_temp=0.1,
    iterations_per_temp=500,
    schedule_params={'alpha': 0.97}
)

# Run the annealing process
result = solver.solve()

print(f"Best energy found: {result.energy}")
print(f"Best solution found: {result.solution}")