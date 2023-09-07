"""Esse módulo contém classes e funções utilitárias
para utilizar em conjunto com a biblioteca pypop7.
"""
from __future__ import annotations

import numpy as np

from py_abm.entities import Fitness

from .abm import ABMFitnessWithCallback


def get_problem_definition(fn: Fitness,
                           callback,
                           n_dims: int):
    fn_ = ABMFitnessWithCallback(fn,
                                 callback,
                                 1)

    def fitness(x):
        return fn_(np.expand_dims(x, axis=0))

    upper_bound = 1 * np.ones((n_dims,), dtype=np.float32)
    lower_bound = np.zeros((n_dims,), dtype=np.float32)

    return {
        'fitness_function': fitness,  # define problem arguments
        'ndim_problem': n_dims,
        'lower_boundary': lower_bound,
        'upper_boundary': upper_bound
    }