"""
Quasar Solver: A Simulated Annealing solver for QUBO problems.
"""

from .qubo import QUBO
from .solver import Solver
from .utils import SolverResult

# This explicitly defines the public API of the package.
__all__ = [
    "QUBO",
    "Solver",
    "SolverResult",
]