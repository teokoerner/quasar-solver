# src/quasar_solver/solver.py

import numpy as np
import concurrent.futures
from typing import Callable, Dict, Any, Optional

from .qubo import QUBO
from .schedules import geometric_cooling, adaptive_cooling
from .utils import SolverResult, estimate_parameters
from . import jit

class Solver:
    """A Simulated Annealing (SA) solver for QUBO problems.

    This class orchestrates the entire annealing process, managing the state,
    temperature, and iterations to find a low-energy solution for a given
    QUBO problem.

    Parameters
    ----------
    qubo : QUBO
        The QUBO problem instance to be solved.

    initial_temp : float, optional
        The starting temperature. If None, it is estimated from the QUBO matrix.

    final_temp : float, optional
        The temperature at which to stop annealing. If None, it is estimated.

    cooling_rate : float, optional
        The cooling factor for the geometric schedule (alpha). If None, it is estimated.

    schedule : callable, optional
        Function to calculate the next temperature. Signature: `f(current_temp, params) -> next_temp`.
        If None, defaults to `geometric_cooling` or `adaptive_cooling` based on heuristics.

    track_history : bool, optional (default=False)
        If True, records temperature, energy, and acceptance rates during annealing.

    schedule_params : dict, optional
        Parameters passed to the schedule function (e.g., {'alpha': 0.99}).

    iterations_per_temp : int, optional
        The number of MCMC steps at each temperature. If None, it is estimated.

    num_reads : int, optional
        Number of independent annealing runs to perform. If None, it is estimated.

    multiprocessing : bool, optional (default=True)
        If True and num_reads > 1, runs reads in parallel using ProcessPoolExecutor.

    Attributes
    ----------
    qubo : QUBO
        The QUBO problem instance.

    initial_temp : float
        Starting temperature.

    final_temp : float
        Stopping temperature.

    cooling_rate : float
        Geometric cooling rate (if used).

    iterations_per_temp : int
        Steps per temperature level.

    num_reads : int
        Number of independent runs.

    schedule : callable
        The cooling schedule function.

    schedule_params : dict
        Parameters for the cooling schedule.
    """

    def __init__(
        self,
        qubo: QUBO,
        initial_temp: Optional[float] = None,
        final_temp: Optional[float] = None,
        cooling_rate: Optional[float] = None,
        schedule: Callable[[float, Dict[str, Any]], float] = None,
        track_history: bool = False,
        schedule_params: Optional[Dict[str, Any]] = None,
        iterations_per_temp: Optional[int] = None,
        num_reads: Optional[int] = None,
        multiprocessing: bool = True
    ):
        """Initializes the SA Solver with annealing parameters.

        If parameters are not provided (None), they will be estimated based on the QUBO matrix.
        """
        # Estimate defaults if any are missing
        est_initial, est_final, est_iter, est_cooling, est_reads,est_schedule = estimate_parameters(qubo.Q)
        
        self.initial_temp = initial_temp if initial_temp is not None else est_initial
        self.final_temp = final_temp if final_temp is not None else est_final
        self.cooling_rate = cooling_rate if cooling_rate is not None else est_cooling
        self.iterations_per_temp = iterations_per_temp if iterations_per_temp is not None else est_iter
        self.num_reads = num_reads if num_reads is not None else est_reads
        default_schedule = geometric_cooling if est_schedule == 'geometric' else adaptive_cooling
        self.schedule = schedule if schedule is not None else default_schedule
        
        # Validation
        if self.initial_temp <= 0:
            raise ValueError(f"initial_temp must be positive, got {self.initial_temp}")
        if self.final_temp <= 0:
            raise ValueError(f"final_temp must be positive, got {self.final_temp}")
        if self.initial_temp <= self.final_temp:
            raise ValueError(f"initial_temp ({self.initial_temp}) must be greater than final_temp ({self.final_temp})")
            
        if not (0 < self.cooling_rate < 1):
             raise ValueError(f"cooling_rate must be between 0 and 1 (exclusive), got {self.cooling_rate}")
             
        if self.iterations_per_temp < 1:
            raise ValueError(f"iterations_per_temp must be at least 1, got {self.iterations_per_temp}")

        if self.num_reads < 1:
            raise ValueError(f"num_reads must be at least 1, got {self.num_reads}")

        if self.schedule is None:
            raise ValueError(f"schedule must be provided, got {self.schedule}")

        self.qubo = qubo
        self.track_history = track_history
        self.multiprocessing = multiprocessing

        
        # Set default parameters based on schedule type
        if schedule_params is None:
            self.schedule_params = {}
        else:
            self.schedule_params = schedule_params.copy() # Avoid side effects

        if self.schedule == geometric_cooling:
             if 'alpha' not in self.schedule_params:
                self.schedule_params['alpha'] = 0.99
        elif self.schedule == adaptive_cooling:
             if 'lambda' not in self.schedule_params:
                # 1. Internal Heuristic for Lambda (Hidden from User)
                # Dense matrices with high off-diagonal variance imply a rough landscape
                q_off_diag = self.qubo.Q - np.diag(np.diag(self.qubo.Q))
                if np.any(q_off_diag):
                    coupling_strength = np.std(q_off_diag[q_off_diag != 0])
                else:
                    coupling_strength = 1.0
                
                # Map coupling strength to a lambda between 0.4 (hard) and 0.8 (easy)
                # This automatically slows down for your 12-city TSP
                auto_lambda = max(0.4, min(0.8, 1.0 / (1.0 + np.log1p(coupling_strength))))
                
                self.schedule_params['lambda'] = auto_lambda

    def _single_read(self) -> SolverResult:
        """Performs a single simulated annealing run.
        
        Returns
        -------
        SolverResult
            Result of the single annealing run containing the best state,
            energy, and history (if tracked).
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
        
        # Adaptive multi-flip probability
        multi_flip_prob = 0.05
        
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
            # Returns: (final_state, final_energy, accepted_moves, best_chain_state, best_chain_energy, energy_std)
            current_state, current_energy, accepted_moves, chain_best_state, chain_best_energy, energy_std = jit.mcmc_chain(
                self.qubo.Q, current_state, current_energy, current_temp, self.iterations_per_temp,
                multi_flip_prob=multi_flip_prob, k=2
            )
            
            # Update best found so far
            if chain_best_energy < best_energy:
                best_energy = chain_best_energy
                best_state = chain_best_state.copy()
            
            acceptance_rate = accepted_moves / self.iterations_per_temp
            
            # Dynamic adjustment of multi-flip probability to escape local minima
            if acceptance_rate < 0.05:
                multi_flip_prob = 0.2 # System is stuck, increase perturbation
            elif acceptance_rate > 0.2:
                multi_flip_prob = 0.05 # Healthy acceptance, revert to standard
            
            if self.track_history:
                history["temperatures"].append(current_temp)
                history["current_energies"].append(current_energy)
                history["best_energies"].append(best_energy)
                history["acceptance_rates"].append(acceptance_rate)
                cumulative_iterations = len(history["temperatures"]) * self.iterations_per_temp
                history["iterations"].append(cumulative_iterations)
            
            # 2c. Cool down the system
            # Update schedule_params with current statistics
            self.schedule_params['sigma'] = energy_std
            current_temp = self.schedule(current_temp, self.schedule_params)

        return SolverResult(
            state=best_state,
            energy=best_energy,
            history=history
        )

    def solve(self) -> SolverResult:
        """Runs the simulated annealing algorithm.

        If num_reads > 1, runs multiple annealing runs (in parallel or sequentially)
        and returns the best result found.

        Returns
        -------
        SolverResult
            The best result found across all reads, minimizing the energy.
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

        else:
            for i in range(self.num_reads):
                result = self._single_read()
            
                if best_result is None or result.energy < best_result.energy:
                    best_result = result

        if best_result is None:
            raise RuntimeError("Solver failed to produce any result.")

        print(f"Annealing complete. Final energy: {best_result.energy}")
        
        return best_result