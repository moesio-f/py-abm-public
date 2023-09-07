import json
from pathlib import Path

from py_abm.visualization.plots import BoxPlot
from py_abm.visualization.parser import ExperimentParser


class GetBestRmseValuesFromMultiplesJsonArquives:
    def __init__(self):
        pass

    @classmethod
    def GetRmseValues(cls, results_json_paths):
        rmse_values = []

        for file_path in results_json_paths:
            exp_result = ExperimentParser(file_path)
            rmse_values.append(exp_result.best_fitness['RMSE'])
        return rmse_values


def create_boxplot(results_paths):
    rmse_values = GetBestRmseValuesFromMultiplesJsonArquives.GetRmseValues(
        results_paths)
    return BoxPlot(rmse_values)
