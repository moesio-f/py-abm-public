# %%
import time
from pathlib import Path

import numpy as np

from py_abm.fitness.abm import ABMFitness, DiscreteABMFitness
from py_abm.fitness.abm.entities import Market, Objective, OptimizationType

from py_abm.runners.pso_abm import PSORunnerABM

# %%

runner = PSORunnerABM(Market.AUTOMAKERS,
                      OptimizationType.GLOBAL,
                      time_steps=1,
                      mc_runs=1,
                      objectives=[Objective.RMSE],
                      discretize_search_space=True,
                      n_workers=4,
                      population_size=4,
                      seed=42,
                      max_fn_evaluation=16,
                      save_directory=Path('test1.db/'),
                      n_groups=None)

runner.run()


#%%


def split(v):
    """Split.
    """
    return v[:, 0], v[:, 1]


def split_(v):
    """Split 2
    """

    return v[0], v[1]


fn = ABMFitness(Market.AUTOMAKERS,
                OptimizationType.GLOBAL,
                10,
                10,
                split_,
                objectives=[Objective.MAE,
                            Objective.RMSE,
                            Objective.R2])

#%%
print(fn.evaluate_single(np.array([0.5, 0.0])))

# %%
fn_ = DiscreteABMFitness(Market.AUTOMAKERS,
                         OptimizationType.AGENT,
                         1,
                         1,
                         split,
                         sql_file_path=Path('database.db'),
                         objectives=[Objective.RMSE])

start = time.perf_counter()
print(fn_.evaluate_single(np.array([[0.1, 0.0]] * 8253)))
print('Duration:', time.perf_counter() - start)

start = time.perf_counter()
print(fn_.evaluate_single(np.array([[0.1, 0.0]] * 8253)))
print('Duration:', time.perf_counter() - start)

# %%
start = time.perf_counter()
print(fn_.evaluate(np.array([[[0.5, 0.0]] * 8253,
                            [[0.0, 0.5]] * 8253,
                            [[0.1, 0.0]] * 8253,
                            [[0.0, 0.1]] * 8253]),
                   n_parallel=4))
print('Duration:', time.perf_counter() - start)

start = time.perf_counter()
print(fn_.evaluate(np.array([[[0.5, 0.0]] * 8253,
                            [[0.0, 0.5]] * 8253,
                            [[0.1, 0.0]] * 8253,
                            [[0.0, 0.1]] * 8253]),
                   n_parallel=4))
print('Duration:', time.perf_counter() - start)

#%%

start = time.perf_counter()
print(fn.evaluate_single(np.array([0.5, 0.0])))
print(fn.evaluate_single(np.array([0.0, 0.5])))
print(fn.evaluate_single(np.array([0.1, 0.0])))
print(fn.evaluate_single(np.array([0.0, 0.1])))
print('Duration:', time.perf_counter() - start)

# %%


start = time.perf_counter()
print(fn.evaluate(np.array([[0.5, 0.0],
                            [0.0, 0.5],
                            [0.1, 0.0],
                            [0.0, 0.1]]),
                  n_parallel=4))
print('Duration:', time.perf_counter() - start)

# %%
