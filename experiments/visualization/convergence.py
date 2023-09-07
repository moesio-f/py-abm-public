from pathlib import Path

import numpy as np

from py_abm.visualization.parser import ExperimentParser
from py_abm.visualization.plots import Convergence


def create_convergence_graph(results_paths):
    exp_results = []
    rmse_values = []

    for path in results_paths:
        exp_results.append(ExperimentParser(path))

    for result in exp_results:
        rmse_values.append(np.array([r['RMSE']
                           for r in result.best_fitness_per_milestone]))

    average_best_fitness = np.mean(rmse_values, axis=0)

    plot = Convergence(values=average_best_fitness,
                       labels=exp_results[0].milestones)
    return plot
