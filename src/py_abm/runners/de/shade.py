"""Esse módulo contém o runner para o SHADE.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from shade_ils.shade import SHADEOptimizer

from py_abm.fitness.abm import Market, Objective, OptimizationType
from py_abm.runners.abm_runner import ABMRunner
from py_abm.runners.utils import shade_ils as utils

logger = logging.getLogger(__name__)


class SHADEABMRunner(ABMRunner):
    """Runner do SHADE para calibração de ABMs através
    do shade_ils.
    """

    def __init__(
            self,
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
            history_size: int = 100,
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
                         n_workers=n_workers,
                         save_directory=save_directory,
                         seed=seed,
                         max_fn_evaluation=max_fn_evaluation,
                         milestone_interval=milestone_interval,
                         log_frequency=log_frequency)

        # Armazenando configurações
        self._pop_size = population_size
        self._history_size = history_size

        # Função de fitness wrapped
        fn = utils.ABMFitnessFunction(fn=self._fn,
                                      callback=self._fitness_callback,
                                      n_workers=self._n_workers,
                                      n_dims=self._n_dims,
                                      lower_bound=0.0,
                                      upper_bound=1.0)

        # Instanciando o otimizador
        self._optimizer = SHADEOptimizer(fn=fn,
                                         population_size=self._pop_size,
                                         max_evaluations=self._max_evaluation,
                                         seed=seed,
                                         history_size=self._history_size)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        result = self._optimizer.optimize()
        solution = result.solution
        fitness = result.fitness

        if fitness.shape == ():
            fitness = np.expand_dims(fitness, axis=0)

        return solution, fitness

    def _algorithm_params(self) -> dict:
        config = {
            'algorithm': {
                'name': self._optimizer.__class__.__name__,
                'provider': 'shade_ils',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': {
                    'population_size': self._pop_size,
                    'history_size': self._history_size,
                    'seed': self._seed
                }
            }
        }

        return config
