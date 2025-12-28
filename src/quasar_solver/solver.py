# src/quasar_solver/solver.py

import numpy as np
import concurrent.futures
from typing import Callable, Dict, Any, Optional

from .qubo import QUBO
from .schedules import geometric_cooling
from .utils import SolverResult
from . import jit

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
        num_reads: int = 1,
        multiprocessing: bool = False
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
            num_reads (int): Number of independent annealing runs to perform in parallel.
        """
        if not initial_temp > final_temp > 0:
            raise ValueError("Temperatures must be positive and initial > final.")
        
        if num_reads < 1:
            raise ValueError("num_reads must be at least 1.")

        self.qubo = qubo
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.track_history = track_history
        self.num_reads = num_reads
        self.multiprocessing = multiprocessing

        self.schedule = schedule
        # Set default parameters for our default schedule
        if schedule_params is None:
            self.schedule_params = {'alpha': 0.99}
        else:
            self.schedule_params = schedule_params

    def _single_read(self) -> SolverResult:
        """
        Performs a single simulated annealing run.
        
        Returns:
            SolverResult: Result of the single annealing run.
        """
        # Re-seed random number generator to ensure independence in parallel processes
        # Using entropy from OS
        np.random.seed()

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
                "acceptance_rates": [],
                "iterations": []
            }

        # 2. Main annealing loop
        while current_temp > self.final_temp:
            # Run entire MCMC chain at once using JIT
            # Returns: (final_state, final_energy, accepted_moves, best_chain_state, best_chain_energy)
            current_state, current_energy, accepted_moves, chain_best_state, chain_best_energy = jit.mcmc_chain(
                self.qubo.Q, current_state, current_energy, current_temp, self.iterations_per_temp
            )
            
            # Update best found so far
            if chain_best_energy < best_energy:
                best_energy = chain_best_energy
                best_state = chain_best_state.copy()
            
            if self.track_history:
                history["temperatures"].append(current_temp)
                history["current_energies"].append(current_energy)
                history["best_energies"].append(best_energy)
                acceptance_rate = accepted_moves / self.iterations_per_temp
                history["acceptance_rates"].append(acceptance_rate)
                cumulative_iterations = len(history["temperatures"]) * self.iterations_per_temp
                history["iterations"].append(cumulative_iterations)
            
            # 2c. Cool down the system
            current_temp = self.schedule(current_temp, self.schedule_params)

        return SolverResult(
            state=best_state,
            energy=best_energy,
            history=history
        )

    def solve(self) -> SolverResult:
        """
        Runs the simulated annealing algorithm. If num_reads > 1, runs multiple
        annealing runs (in parallel) and returns the best result.

        Returns:
            SolverResult: The best result found across all reads.
        """
        best_result = None
        
        if self.multiprocessing:
            # Parallel execution
            results = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # We map the _single_read function to a range of run_ids
                futures = [executor.submit(self._single_read) for i in range(self.num_reads)]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Run generated an exception: {e}")

            # Find the best result
            if not results:
                raise RuntimeError("No results generated from parallel execution.")

            best_result = min(results, key=lambda x: x.energy)
            print(f"Parallel annealing complete ({self.num_reads} reads). Best energy: {best_result.energy}")

        else:
            for i in range(self.num_reads):
                result = self._single_read()
                print(f"Annealing run {i+1} complete. Final energy: {result.energy}")
            
                if best_result is None or result.energy < best_result.energy:
                    best_result = result

        if best_result is None:
            raise RuntimeError("Solver failed to produce any result.")

        print(f"Annealing complete. Final energy: {best_result.energy}")
        
        return best_result