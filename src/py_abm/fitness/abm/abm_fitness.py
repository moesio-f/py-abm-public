"""Esse módulo define o ABM.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from py_abm.entities import Fitness
from py_abm.fitness.abm import entities
from py_abm.fitness.abm._core.process_handler import ProcessHandler
from py_abm.fitness.abm._core.simulation_pool import SimulationPool


class ABMFitness(Fitness):
    """Essa classe representa a função de fitness
    base para o ABM considerado. 
    """

    def __init__(self,
                 market: entities.Market,
                 optimization_type: entities.OptimizationType,
                 time_steps: int,
                 mc_runs: int,
                 r_values_from_individual: Callable[[np.ndarray],
                                                    tuple[np.ndarray,
                                                          np.ndarray]],
                 objectives: list[entities.Objective] | None = None) -> None:
        """Construtor.

        Args:
            market (entities.Market): mercado.
            optimization_type (entities.OptimizationType): tipo de otimização (agente, 
                segmento, global).
            r_values_from_individual: função que converte um indivíduo (vetor)
                para seus valores de R1 e R2.
            objectives (list[entities.Objective, optional): lista de erros
                que devem ser calculados. Defaults to None (MAE, R2 e RMSE).
        """
        assert market is not None
        assert optimization_type is not None
        assert r_values_from_individual is not None
        assert time_steps >= 1
        assert mc_runs >= 1

        self._market = market
        self._opt_type = optimization_type
        self._objectives = objectives
        self._ts = time_steps
        self._mc_runs = mc_runs
        self._r_values_from_individual = r_values_from_individual
        self._last_output: list[entities.ABMJarOutput] | None = None

        if objectives is None:
            self._objectives = list(entities.Objective)

    @property
    def market(self) -> entities.Market:
        """Retorna o mercado.

        Returns:
            entities.Market: mercado.
        """
        return self._market

    @property
    def time_steps(self) -> int:
        """Retorna a quantidade de time steps
        da simulação.

        Returns:
            int: time steps.
        """
        return self._ts

    @property
    def mc_runs(self) -> int:
        """Retorna a quantidade de mcruns da 
        simulação.

        Returns:
            int: mcruns.
        """
        return self._mc_runs

    @property
    def optimization_type(self) -> entities.OptimizationType:
        """Retorna o tipo de otimização.

        Returns:
            entities.OptimizationType: tipo de otimização.
        """
        return self._opt_type

    @property
    def objectives(self) -> list[entities.Objective]:
        """Retorna os objetivos.

        Returns:
            list[entities.Objective]: objetivos.
        """
        return self._objectives

    @property
    def r_values_from_individual(self) -> Callable[[np.ndarray],
                                                   tuple[np.ndarray,
                                                         np.ndarray]]:
        """Retorna a função para obter r-values.

        Returns:
            Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]: função.
        """
        return self._r_values_from_individual

    @property
    def abm_output(self) -> list[entities.ABMJarOutput] | None:
        """Retorna a saída do ABM para a última avaliação 
        de fitness realizada (None caso nenhuma tenha sido 
        realizada). Esse valor é atualizado a cada chamada à 
        "evaluate" ou "evaluate_single".

        Returns:
            list[entities.ABMJarOutput] | None: última saída produzida
                pelo ABM.
        """
        return self._last_output

    def evaluate(self,
                 individuals: np.ndarray,
                 n_parallel: int | None = None,
                 **kwargs) -> np.ndarray:
        """Principal método de avaliação. Permite avaliar
        um conjunto de indivíduos (representados como Numpy Arrays)

        Args:
            individuals (np.ndarray): array com os indivíduos.
            n_parallel (int, optional): quantidade de workers. 
                Defaults to None (1 worker).

        Returns:
            np.ndarray: resultado da avaliação de fitness.
        """
        n_individuals = individuals.shape[0]
        if n_parallel is None:
            n_parallel = 1

        pool = SimulationPool(n_parallel)

        for i in range(n_individuals):
            # Obtenção dos valores de r1 e r2
            r1, r2 = self._r_values_from_individual(individuals[i])

            # Instanciação do ProcessHandler para execução da simulação
            abm_jar = ProcessHandler(market=self._market,
                                     run_type=self._opt_type,
                                     time_steps=self._ts,
                                     mc_runs=self._mc_runs,
                                     r1=r1,
                                     r2=r2)

            # Executar handler
            pool.run_handler(abm_jar, i)

        # Aguardamos até a finalização dos demais processos
        pool.wait_all()

        # Armazenamos esses resultados
        self._last_output = pool.results()

        return np.array(list(map(self._fitness_result,
                                 self._last_output)),
                        dtype=np.float32)

    def evaluate_single(self,
                        individual: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Realiza a avaliação de um único indivíduo.

        Args:
            individual (np.ndarray): indivíduo.

        Returns:
            np.ndarray: valor da função de fitness.
        """
        # Obtenção dos valores de r1 e r2
        r1, r2 = self._r_values_from_individual(individual)

        # Instanciação do ProcessHandler para execução da simulação
        abm_jar = ProcessHandler(market=self._market,
                                 run_type=self._opt_type,
                                 time_steps=self._ts,
                                 mc_runs=self._mc_runs,
                                 r1=r1,
                                 r2=r2)
        abm_jar.run(join=True)

        # Armazenamos esses resultados
        self._last_output = [abm_jar.results()]

        # Parse dos resultados
        return self._fitness_result(self._last_output[0])

    def _fitness_result(self, result: entities.ABMJarOutput) -> np.ndarray:
        """Método auxiliar, realiza o parse dos resultados
        do simulador para os valores da função de fitness.

        Args:
            result (entities.ABMJarOutput): saída do simulador.

        Returns:
            np.ndarray: valor da função de fitness.
        """
        return np.array([result.objectives[k]
                         for k in self._objectives],
                        dtype=np.float32)
