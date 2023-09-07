"""Esse módulo contém o runner para o PSO.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pypop7.optimizers.cc import HCC

from py_abm.fitness.abm import Market, Objective, OptimizationType
from py_abm.runners.abm_runner import ABMRunner
from py_abm.runners.utils.pypop7 import get_problem_definition

logger = logging.getLogger(__name__)


class PypopHCCRunnerABM(ABMRunner):
    """Runner do HCC para otimização do ABM através
    do pypop7.
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
        # Inicializando classe base
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

        # Criando configurações do algoritmo
        self._options = {
            'n_individuals': population_size,
            'max_function_evalutions': self._max_evaluation,
            'seed_rng': self._seed,
            'n_dim_subproblem': dims_subproblem,
            'sigma': sigma,
        }

        # Obtendo configurações do problema
        self._problem = get_problem_definition(self._fn,
                                               self._fitness_callback,
                                               self._n_dims)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        optimizer = HCC(self._problem, self._options)
        result = optimizer.optimize()
        x = np.array(result['best_so_far_x'], dtype=np.float32)
        y = np.array(result['best_so_far_y'], dtype=np.float32)

        if y.shape == ():
            y = np.expand_dims(y, axis=0)

        return x, y

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': 'HCC',
                'provider': 'pypop7',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': self._options
            }
        }

        return config
