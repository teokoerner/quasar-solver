# Quasar Solver

**QUASAR**: **QU**BO-based **A**nnealing **S**earch **A**nd **R**esolution

![Photo of a quasar by NASA](quasar.jpg)
*Photo by <a href="https://unsplash.com/@hubblespacetelescope?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">NASA Hubble Space Telescope</a> on <a href="https://unsplash.com/photos/an-image-of-a-very-large-and-colorful-object-in-the-sky-TZIorZKAXYo?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>*

Quasar is a Simulated Annealing (SA) solver built for QUBO-based optimization problems.

Its name is inspired by the astronomical phenomenon of a **Quasar** (Quasi-Stellar Radio Source). A quasar is the active core of a galaxy, powered by a supermassive black hole, that emits exceptionally large amounts of energy. Despite appearing as a single point of light (like a star) in visible wavelengths, it is one of the most luminous and energetic objects in the universe.

In the context of optimization, finding the global minimum in a complex, high-dimensional landscape is like searching for a specific point of light in the vast darkness of space. Just as a quasar illuminates the cosmos, this solver is designed to "illuminate" the search space, using energetic fluctuations (temperature) to escape local traps and converge upon the optimal solution.

## Features

*   **Automatic Parameter Estimation**: The solver analyzes your QUBO matrix to automatically set initial/final temperatures, cooling rates, and iteration counts.
*   **Adaptive Cooling**: Implements Adaptive Cooling Schedule, which dynamically adjusts the cooling rate based on the standard deviation of energy distributions, ensuring efficient exploration and exploitation.
*   **JIT Optimization**: Powered by Numba, critical inner loops (like energy delta calculations) are Just-In-Time compiled to machine code for C-like performance.
*   **Smart Moves**: Features multi-flip moves to escape deep local minima that single-bit flips cannot.
*   **Parallel Execution**: Native support for running multiple independent annealing chains in parallel to maximize the probability of finding the global optimum.

## Getting Started

To see Quasar in action, check out the provided examples:

*   [**01_getting_started.ipynb**](examples/01_getting_started.ipynb): A gentle introduction to defining a problem and running the solver.
*   [**02_advanced_usage.ipynb**](examples/02_advanced_usage.ipynb): Deep dive into parameter tuning, schedules, and performance analysis.

## Basic Usage

```python
import numpy as np
from quasar_solver import QUBO, Solver

# 1. Define a simple QUBO problem (e.g., a MAX-CUT instance)
#    Minimize E = -x0 -x1 -x2 + 2(x0x1 + x1x2 + x0x2) ... (Example matrix)
Q = np.array([
    [-1,  2,  2],
    [ 2, -1,  2],
    [ 2,  2, -1]
])

# 2. Wrap it in a QUBO object
qubo = QUBO(Q)

# 3. Initialize the solver
#    Parameters are estimated automatically if not provided!
solver = Solver(
    qubo=qubo
)

# 4. Run the annealing process
result = solver.solve()

print(f"Best energy found: {result.energy}")
print(f"Best solution state: {result.state}")

```

## About This Project

Quasar is a personal development project focused on the intersection of physics, mathematics, and computer science for solving combinatorial optimization problems.