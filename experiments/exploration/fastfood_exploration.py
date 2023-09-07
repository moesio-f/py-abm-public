"""Código para o experimento de exploração
do mercado FASTFOOD.
"""

import itertools
import time

import numpy as np
import pandas as pd

from py_abm.fitness.abm import ABMFitness
from py_abm.fitness.abm.entities import Market, Objective, OptimizationType


def individual_generator() -> np.ndarray:
    """Gera indivíduos para avaliação.

    Returns:
        np.ndarray: indivíduo.
    """
    # Definindo algumas propriedades do espaço de busca
    start = 0.0
    stop = 1.01
    step = 0.01

    # Definindo o conjunto de pontos (1d)
    space = np.arange(start=start,
                      stop=stop,
                      step=step)

    # Percorrendo o espaço de busca
    for i in range(space.shape[0]):
        for j in range(space.shape[0]):
            yield np.array([space[i], space[j]],
                           dtype=np.float32)


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."

    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield np.stack(batch)


if __name__ == '__main__':
    # Instanciando função de fitness
    time_steps = 10
    mc_runs = 10

    fn = ABMFitness(market=Market.FASTFOOD,
                    optimization_type=OptimizationType.GLOBAL,
                    time_steps=time_steps,
                    mc_runs=mc_runs,
                    r_values_from_individual=lambda v: (v[0], v[1]),
                    objectives=[Objective.MAE,
                                Objective.R2,
                                Objective.RMSE])

    # Calculando função de fitness
    batch_size = 20
    n_workers = 20
    remaining = 101**2
    rows = []

    print('------- INICIANDO AVALIAÇÃO -------')
    start = time.perf_counter()
    for individuals in iter(batched(individual_generator(),
                                    batch_size)):
        output = fn.evaluate(individuals,
                             n_parallel=n_workers)
        rows.append(np.concatenate((individuals,
                                    output),
                                   axis=1))
        remaining -= output.shape[0]
        print(f'Batch com {output.shape[0]} indivíduos finalizado.')
        print(f'Restam {remaining} indivíduos.\n')

    end = time.perf_counter()
    print('Execução finalizada\n'
          f'Tempo total: {end - start}s')

    df = pd.DataFrame(np.concatenate(rows),
                      columns=['r_1',
                               'r_2',
                               'mae',
                               'r2',
                               'rmse'])
    df.to_csv(f'fastfood_ts{time_steps}_mcruns{mc_runs}.csv',
              index=False)
