# src/quasar_solver/solver.py

import numpy as np
from typing import Callable, List, Dict, Any, Optional

from .qubo import QUBO
from .engine import run_mcmc_step
from .schedules import geometric_cooling
from .utils import SolverResult

class Solver:
    """
    A Simulated Annealing (SA) solver for QUBO problems.

    This class orchestrates the entire annealing process, managing the state,
    temperature, and iterations to find a low-energy solution for a given
    QUBO problem.
    """

    def __init__(
        self,
        qubo: QUBO,
        initial_temp: float = 5.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.99,
        schedule: Callable[[float, Dict[str, Any]], float] = geometric_cooling,
        track_history: bool = False,
        schedule_params: Optional[Dict[str, Any]] = None,
        iterations_per_temp: int = 100,
    ):
        """
        Initializes the SA Solver with annealing parameters.

        Args:
            qubo (QUBO): The QUBO problem instance to be solved.
            initial_temp (float): The starting temperature.
            final_temp (float): The temperature at which to stop annealing.
            cooling_rate (float): The cooling factor for the geometric schedule (alpha).
            iterations_per_temp (int): The number of MCMC steps at each temperature
                                       (Markov chain length).
        """
        if not initial_temp > final_temp > 0:
            raise ValueError("Temperatures must be positive and initial > final.")
        
        self.qubo = qubo
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.track_history = track_history

        self.schedule = schedule
        # Set default parameters for our default schedule
        if schedule_params is None:
            self.schedule_params = {'alpha': 0.99}
        else:
            self.schedule_params = schedule_params

    def solve(self) -> SolverResult:
        """
        Runs the simulated annealing algorithm to find the optimal solution.

        Returns:
            SolverResult: A dataclass object containing the best state,
                          its energy, and the convergence history.
        """
        # 1. Initialize the system
        current_state = np.random.randint(0, 2, size=self.qubo.num_variables)
        current_energy = self.qubo.energy(current_state)

        best_state = current_state.copy()
        best_energy = current_energy

        current_temp = self.initial_temp
        history: Dict[str, Any] = {}
        
        if self.track_history:
            history = {
                "temperatures": [],
                "current_energies": [],
                "best_energies": [],
                "acceptance_rates": []  # Added for plotting
            }

        # 2. Main annealing loop
        while current_temp > self.final_temp:
            accepted_moves = 0
            # 2a. Run MCMC steps at the current temperature
            for _ in range(self.iterations_per_temp):
                new_state, new_energy = run_mcmc_step(
                    self.qubo, current_state, current_energy, current_temp
                )
                
                # Check if the move was accepted
                if new_energy != current_energy:
                    accepted_moves += 1

                current_state, current_energy = new_state, new_energy

                # 2b. Update the best-known solution
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()
            
            if self.track_history:
                history["temperatures"].append(current_temp)
                history["current_energies"].append(current_energy)
                history["best_energies"].append(best_energy)
                # Calculate and store the acceptance rate for this temperature
                acceptance_rate = accepted_moves / self.iterations_per_temp
                history["acceptance_rates"].append(acceptance_rate)
            
            # 2c. Cool down the system
            current_temp = self.schedule(current_temp, self.schedule_params)

        print(f"Annealing complete. Final energy: {best_energy}")

        return SolverResult(
            state=best_state,
            energy=best_energy,
            history=history
        )