from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pypop7.optimizers.cc import COCMA

from py_abm.fitness.abm import Market, Objective, OptimizationType
from py_abm.runners.abm_runner import ABMRunner
from py_abm.runners.utils.pypop7 import get_problem_definition

logger = logging.getLogger(__name__)


class PypopCOCMARunnerABM(ABMRunner):
    """
    A class that represents an agent-based model (ABM) runner using the COCMA (Covariance Matrix Adaptation Evolution
    Strategy with a Clustering Search Space Decomposition) optimization algorithm from the pypop7 library.

    Parameters:
        market (Market): The market instance for the ABM simulation.
        optimization_type (OptimizationType): The type of optimization ('maximize' or 'minimize').
        time_steps (int): The number of time steps for the ABM simulation.
        mc_runs (int): The number of Monte Carlo runs for the ABM simulation.
        n_groups (int | None): The number of groups for the ABM simulation. If None, groups will not be used.
        objectives (list[Objective]): A list of objectives to optimize in the ABM simulation.
        discretize_search_space (bool): A flag indicating whether to discretize the search space for optimization.
        n_workers (int): The number of workers to use for the optimization algorithm.
        population_size (int): The number of individuals in the population for the optimization algorithm.
        save_directory (Path): The directory to save the optimization results.
        seed (int, optional): The random seed for reproducibility. Default is 42781.
        max_fn_evaluation (int, optional): The maximum number of function evaluations for the optimization algorithm.
            Default is 30000.
        sigma (float, optional): The initial step size for the COCMA algorithm. Default is 1/3.
        dims_subproblem (int, optional): The number of dimensions for each subproblem. Default is None, which
            automatically calculates based on the total number of dimensions.
        log_frequency (int, optional): The frequency of logging during the optimization process. Default is 100.

    Note:
        The `get_problem_definition` function is used to obtain the problem configuration for the optimization algorithm.
    """

    def __init__(self,
                 market: Market,
                 optimization_type: OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 n_groups: int | None,
                 objectives: list[Objective],
                 discretize_search_space: bool,
                 n_workers: int,
                 population_size: int,
                 save_directory: Path,
                 seed: int = 42781,
                 max_fn_evaluation: int = 30000,
                 sigma: float = None,
                 dims_subproblem: int = None,
                 milestone_interval: int = 100,
                 log_frequency: int = 100) -> None:

        super().__init__(market=market,
                         optimization_type=optimization_type,
                         time_steps=time_steps,
                         mc_runs=mc_runs,
                         n_groups=n_groups,
                         objectives=objectives,
                         discretize_search_space=discretize_search_space,
                         n_workers=1,
                         save_directory=save_directory,
                         seed=seed,
                         max_fn_evaluation=max_fn_evaluation,
                         milestone_interval=milestone_interval,
                         log_frequency=log_frequency)
        if sigma is None:
            sigma = 1 / 3

        if dims_subproblem is None:
            dims_subproblem = max(1, self._n_dims // 100)

        self._options = {
            'n_individuals': population_size,
            'max_function_evalutions': self._max_evaluation,
            'seed_rng': self._seed,
            'n_dim_subproblem': dims_subproblem,
            'sigma': sigma,
        }

        self._problem = get_problem_definition(self._fn,
                                               self._fitness_callback,
                                               self._n_dims)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        optimizer = COCMA(self._problem, self._options)
        result = optimizer.optimize()
        x = np.array(result['best_so_far_x'], dtype=np.float32)
        y = np.array(result['best_so_far_y'], dtype=np.float32)

        if y.shape == ():
            y = np.expand_dims(y, axis=0)

        return x, y

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': 'COCMA',
                'provider': 'pypop7',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': self._options
            }
        }

        return config
