"""Esse módulo contém o runner para o PSO.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pyswarms as ps
from pyswarms.backend.topology import (Pyramid, Random, Ring, Star, Topology,
                                       VonNeumann)

from py_abm.fitness.abm import Market, Objective, OptimizationType
from py_abm.runners.abm_runner import ABMRunner
from py_abm.runners.utils.abm import ABMFitnessWithCallback

logger = logging.getLogger(__name__)


class PySwarmsPSORunnerABM(ABMRunner):
    """Runner do PSO para calibração de ABMs através
    do PySwarms.
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
            topology: str | Topology = 'ring',
            seed: int = 42781,
            max_fn_evaluation: int = 30000,
            w: float = 0.7213475204444817,  # pylint: disable=invalid-name
            c1: float = 1.1931471805599454,  # pylint: disable=invalid-name
            c2: float = 1.1931471805599454,  # pylint: disable=invalid-name
            k: int | None = None,  # pylint: disable=invalid-name
            p: int | None = None,
            r: int | None = None,
            initial_position: None | np.ndarray = None,
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

        if isinstance(topology, str):
            available = ['random', 'ring', 'vonneuman', 'pyramid', 'star']
            topologies = [Random(), Ring(), VonNeumann(), Pyramid(), Star()]
            topology = next(c
                            for t, c in zip(available, topologies)
                            if topology == t)

        requires_k = any(map(lambda t: isinstance(topology, t),
                             [Ring, VonNeumann, Random]))

        if k is None and requires_k:
            # Caso não seja passado o valor de K,
            #   selecionamos 1/5 da população para ele.
            k = max(1, population_size // 5)

        requires_p = any(map(lambda t: isinstance(topology, t),
                             [Ring, VonNeumann]))

        if p is None and requires_p:
            # Caso não seja passado o valo de P,
            #   selecionamos a normal L2 (euclidiana).
            p = 2

        requires_r = isinstance(topology, VonNeumann)
        if r is None and requires_r:
            # Caso não seja passado o range para
            #   a arquitetura, utilizamos o valor 1.
            r = 1

        self._optimizer = ps.single.GeneralOptimizerPSO(
            n_particles=population_size,
            dimensions=self._n_dims,
            options=dict(c1=c1, c2=c2, w=w, k=k, p=p, r=r),
            topology=topology,
            bounds=(np.zeros((self._n_dims,), dtype=np.float32),
                    np.ones((self._n_dims,), dtype=np.float32)),
            init_pos=initial_position)

    def _run(self,
             *args,
             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        fn = ABMFitnessWithCallback(fn=self._fn,
                                    callback=self._fitness_callback,
                                    n_workers=self._n_workers,
                                    post_evaluation=np.squeeze)
        n_iterations = self._max_evaluation / self._optimizer.n_particles
        n_iterations = math.ceil(n_iterations)
        logger.info(f'Setting number of iterations to {n_iterations}...')
        logger.info('Setting global random seed with np.random.seed...')
        np.random.seed(self._seed)

        logger.info('Starting optimization...')
        fitness, individual = self._optimizer.optimize(fn,
                                                       iters=n_iterations,
                                                       verbose=False)

        return (np.array(individual, dtype=np.float32),
                np.array([fitness], dtype=np.float32))

    def _algorithm_params(self) -> dict:
        params = dict()
        params.update(self._optimizer.options)
        params.update({
            'population_size': self._optimizer.n_particles,
            'topology': self._optimizer.top.__class__.__name__,
            'seed': self._seed
        })

        config = {
            'algorithm': {
                'name': self._optimizer.__class__.__name__,
                'provider': 'pyswarms',
                'max_fn_evaluation': self._max_evaluation,
                'parameters': params
            }
        }

        return config
