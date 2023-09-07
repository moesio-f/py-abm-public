"""Esse módulo contém funções e funções utilitárias
para utilizar em conjunto com a biblioteca shade_ils.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from shade_ils.entities import FitnessFunction

from py_abm.entities import Fitness

from .abm import ABMFitnessWithCallback


class ABMFitnessFunction(FitnessFunction):
    """Representa uma função de fitness do py_abm
    como um Problem do shade_ils.
    """

    def __init__(self,
                 fn: Fitness,
                 callback: Callable[[np.ndarray,
                                     Fitness,
                                     np.ndarray],
                                    None],
                 n_workers: int,
                 n_dims: int,
                 lower_bound: float,
                 upper_bound: float):
        """Construtor.

        Args:
            fn (Fitness): função de fitness.
            callback (Callable[[np.ndarray, Fitness, np.ndarray], None]):
                callback chamado após avaliação (evaluate) da
                função de fitness.
            n_workers (int): quantidade de workers para execução
                paralela.
            n_dims (int): número de dimensões do problema.
            lower_bound (float): limite inferior.
            upper_bound (float): limite superior.
        """
        if callback is None:
            def _pass(*args, **kwargs):
                del args
                del kwargs

            callback = _pass

        self._n_dims = n_dims
        self._lower = lower_bound
        self._upper = upper_bound
        self._fn = ABMFitnessWithCallback(fn,
                                          callback,
                                          n_workers)

    def call(self, population: np.ndarray) -> np.ndarray:
        return self._fn(population).squeeze(axis=-1)

    def info(self) -> dict:
        return {
            'lower': self._lower,
            'upper': self._upper,
            'dimension': self._n_dims
        }

    def name(self) -> str:
        return 'ABM'
