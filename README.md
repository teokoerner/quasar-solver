# Quasar Solver

**QUASAR**: **QU**BO-based **A**nnealing **S**earch **A**nd **R**esolution

A from-scratch Simulated Annealing (SA) solver in Python, designed for solving **Quadratic Unconstrained Binary Optimization (QUBO)** problems.

![Photo of a quasar by NASA](quasar.jpg) 
*Photo by <a href="https://unsplash.com/@hubblespacetelescope?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">NASA Hubble Space Telescope</a> on <a href="https://unsplash.com/photos/an-image-of-a-very-large-and-colorful-object-in-the-sky-TZIorZKAXYo?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>*
      
## About This Project

Quasar is primarily a personal development project focused on the intersection of physics, mathematics, and computer science for solving combinatorial optimization problems. 

It's name is inspired by the astronomical phenomenon 'quasar'. A quasar is the active core of a galaxy that appears almost point-like (like a star) in the visible range of light and emits very large amounts of energy in other wavelength ranges. Just as a quasar is an extremely luminous object in the vastness of space, this project aims to be a tool for finding optimal solutions in a vast search space.

## Getting Started

*(Instructions to be added once the project is installable.)*

## Basic Usage

```python
import numpy as np
from quasar_solver import solve_qubo

# Define a simple QUBO problem (e.g., a MAX-CUT instance)
Q = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

# Run the solver
solver = Solver(
    qubo=Q,
    initial_temp=10.0,
    final_temp=0.05,
    iterations_per_temp=200,
    schedule_params={'alpha': 0.95}
)

# Run the annealing process
result = solver.solve()

print(f"Best energy found: {result.energy}")
print(f"Best solution found: {result.solution}")