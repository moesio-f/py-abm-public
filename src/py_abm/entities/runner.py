"""Esse módulo contém a definição das interfaces 
relativas ao Runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Runner(ABC):
    """Essa classe representa um runner.

    Um runner, representa todo o conjunto de configurações 
    necessárias para calibrar os parâmetros de um ABM
    usando um dado algoritmo evolucionário.

    Em outras palavras, o runner é composto por:
        - Algoritmo (e configurações);
        - Função de fitness;
        - Definição do problema/experimento;
        - Métodos para execução e recuperação dos resultados;
    """

    @abstractmethod
    def run(self,
            *args,
            **kwargs) -> dict:
        """Esse método executa o runner seguindo as
        configurações definidas.
        """

    @abstractmethod
    def result(self,
               *args,
               **kwargs) -> dict | None:
        """Esse método retorna os resultados obtidos 
        após execução do Runner.
        """

    @abstractmethod
    def config(self,
               *arg,
               **kwargs) -> dict:
        """Esse método retorna a configuração do
        runner.
        """


class FitnessHistory:
    """Representa o histórico
    da função de fitness.
    """


class SolutionHistory:
    """Representa o histórico
    de soluções.
    """


class RunnerConfiguration:
    """Representa o conjunto de
    configurações de um runner.
    """


@dataclass
class RunnerResult:
    """Essa classe representa os resultados
    de uma execução de um runner.
    """
    runner_configuration: RunnerConfiguration
    fitness_history: FitnessHistory
    solution_hist: SolutionHistory
    execution_time: list[int]
