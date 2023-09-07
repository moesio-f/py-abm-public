"""Esse módulo contém classes e funções utilitárias
para utilizar em conjunto com a biblioteca pymoo.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from pymoo.core.problem import Problem

from py_abm.entities import Fitness
from py_abm.fitness.abm import Market, Objective, OptimizationType

from .abm import ABMFitnessWithCallback


class FitnessProblem(Problem):
    """Representa uma função de fitness do py_abm
    como um Problem do pymoo.
    """

    def __init__(self,
                 fn: Fitness,
                 callback: Callable[[np.ndarray,
                                     Fitness,
                                     np.ndarray],
                                    None],
                 n_workers: int,
                 n_dims: int,
                 n_objectives: int,
                 lower_bounds: np.ndarray | float,
                 upper_bounds: np.ndarray | float,
                 **kwargs):
        """Construtor.

        Args:
            fn (Fitness): função de fitness.
            callback (Callable[[np.ndarray, Fitness, np.ndarray], None]):
                callback chamado após avaliação (evaluate) da
                função de fitness.
            n_workers (int): quantidade de workers para execução
                paralela.
            n_dims (int): número de dimensões do problema.
            n_objectives (int): número de objetivos.
            lower_bounds (np.ndarray | float): limite inferior
                das variáveis de decisão.
            upper_bounds (np.ndarray | float): limite superior
                das variáveis de decisão.
        """
        if callback is None:
            def _pass(*args, **kwargs):
                del args
                del kwargs

            callback = _pass

        self._fn = ABMFitnessWithCallback(fn,
                                          callback,
                                          n_workers)
        super().__init__(n_var=n_dims,
                         n_obj=n_objectives,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         elementwise=False,
                         **kwargs)

    def _evaluate(self,
                  x,
                  out,
                  *args,
                  **kwargs):
        """Método de avaliação de indivíduos.

        Args:
            x: NumPy array em formato de matriz.
            out: dicionário para escrever a saída.
        """
        # Armazenando o resultado da avaliação
        out["F"] = self._fn(x)
