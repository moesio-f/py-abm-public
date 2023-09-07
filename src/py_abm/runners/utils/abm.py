"""Esse módulo contém funções e classes utilitárias
para otimizações do ABM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from py_abm.entities import Fitness
from py_abm.fitness.abm import Market, OptimizationType


class ABMFitnessWithCallback:
    """Wrapper que invoca uma função de
    fitness de ABM e uma função de callback.
    """

    def __init__(self,
                 fn: Fitness,
                 callback: Callable[[np.ndarray,
                                     Fitness,
                                     np.ndarray],
                                    None],
                 n_workers: int,
                 post_evaluation: Callable[[np.ndarray],
                                           np.ndarray] = None):
        """Construtor.

        Args:
            fn: função de fitness a ser utilizada.
            callback: função de callback.
            n_workers: quantidade de workers
                para a função de fitness.
            post_evaluation: função de pós-processamento.
        """
        assert fn is not None
        assert n_workers >= 1

        if post_evaluation is None:
            def post_evaluation(x): return x

        self._fn = fn
        self._callback = callback
        self._n_workers = n_workers
        self._post_eval = post_evaluation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Matriz de indivíduos.

        Args:
            x (np.ndarray): matriz de indivíduos.

        Returns:
            fitness.
        """
        # Realizando a avaliação dos indivíduos
        output = self._fn.evaluate(x, n_parallel=self._n_workers)

        # Aplicando método de callback
        self._callback(x,
                       self._fn,
                       output)

        return self._post_eval(output)


@dataclass(frozen=True)
class HistoryEntry:
    average_solution: np.ndarray
    average_fitness: np.ndarray
    best_solution: np.ndarray
    best_fitness: np.ndarray


class HistoryManager:
    """Classe que permite armazenar
    algumas métricas que descrevem
    o processo evolutivo até o momento.

    TODO: adicionar cálculo da variância
    através de Welford ou Parallel.
    """

    def __init__(self) -> None:
        self._n = 0
        self._avg_solution = None
        self._avg_fitness = None
        self._best_fitness = None
        self._best_solution = None

    def update(self,
               individuals: np.ndarray,
               fitness: np.ndarray) -> None:
        """Atualiza as métricas controladas por
        esse histórico.

        Args:
            individuals (np.ndarray): indivíduos da população,
                no formato (n_individuals, n_dims).
            fitness (np.ndarray): respectivo valores de fitness,
                no formato (n_individuals, n_objectives).
        """
        assert individuals.shape[0] == fitness.shape[0]

        if self._best_solution is None:
            self._set_initial(individuals, fitness)
        else:
            self._update_cumulative(individuals, fitness)

    def _set_initial(self,
                     individuals: np.ndarray,
                     fitness: np.ndarray) -> None:
        self._avg_fitness = fitness.mean(axis=0)
        self._avg_solution = individuals.mean(axis=0)
        self._n = individuals.shape[0]
        self._set_best(individuals, fitness)

    def _update_cumulative(self,
                           individuals: np.ndarray,
                           fitness: np.ndarray) -> None:
        # Atualizando médias
        self._avg_solution = self._get_new_average(self._avg_solution,
                                                   individuals)
        self._avg_fitness = self._get_new_average(self._avg_fitness,
                                                  fitness)

        # Juntando indivíduos e fitness do batch
        #   com os melhores conhecidos.
        individuals = np.concatenate([individuals, [self._best_solution]])
        fitness = np.concatenate([fitness, [self._best_fitness]])

        # Calculando o melhor deles
        self._set_best(individuals, fitness)

        # Atualizando a quantidade de amostras
        self._n += individuals.shape[0]

    def _set_best(self,
                  individuals: np.ndarray,
                  fitness: np.ndarray):
        assert fitness.shape[1] == 1, 'Multi-objective not implemented.'
        arg_min = fitness.argmin()
        self._best_solution = individuals[arg_min]
        self._best_fitness = fitness[arg_min]

    def _get_new_average(self,
                         current: np.ndarray,
                         new_batch: np.ndarray) -> np.ndarray:
        # Média cumulativa
        # Obtendo a quantidade de novas amostras
        batch_size = new_batch.shape[0]

        # Calculando Sn (soma atual)
        previous_sum = self._n * current

        # Calculando Sk (soma da amostra)
        current_sum = new_batch.sum(axis=0)

        # Obtendo a soma total (Sn + Sk)
        total_sum = previous_sum + current_sum

        # Calculando a nova média: (Sn + Sk) / total
        return total_sum / (batch_size + self._n)

    def get_current(self) -> HistoryEntry:
        return HistoryEntry(
            best_solution=self._best_solution,
            best_fitness=self._best_fitness,
            average_fitness=self._avg_fitness,
            average_solution=self._avg_solution)


def valid_n_groups(market: Market,
                   optimization_type: OptimizationType,
                   n_groups: int) -> bool:
    """Realiza uma checagem se a quantidade de grupos
    é válida para o par <mercado, tipo de otimização>.

    Args:
        market (Market): mercado.
        optimization_type (OptimizationType): tipo de otimização.
        n_groups (int): quantidade de grupos.

    Returns:
        bool: se a quantidade de grupos é válida.
    """
    valid_conditions = [n_groups > 1,
                        optimization_type == OptimizationType.AGENT,
                        n_groups <= market.agents,
                        (n_groups % market.segments) == 0]
    return all(valid_conditions)


def create_groups(n_groups: int,
                  market: Market) -> np.ndarray:
    """Define a quantidade de agentes por grupo.

    Args:
        n_groups (int): quantidade de grupos.
        market (Market): mercado.

    Returns:
        np.ndarray: quantidade de agentes por grupo.

    """
    segments = market.agents_per_segment
    n_groups_per_segment = n_groups // len(segments)
    groups = []
    segments_count = dict()

    # Para cada segmento
    for i, n_agents in enumerate(segments):
        # Obtemos quantos agentes devem estar em cada grupo
        #   dentro desse segmento
        n_agent_per_group = n_agents // n_groups_per_segment

        # Adicionalmente, podemos ter uma quantidade de agentes
        #   que não puderam ser alocados igualmente entre os
        #   grupos desse segmento
        remaining_agents = n_agents % n_groups_per_segment

        for _ in range(n_groups_per_segment):
            # Para cada grupo nesse segmento,
            #   adicionamos a quantidade base de agentes
            groups.append(n_agent_per_group)

            # Caso existam agentes não alocados,
            #   adicionamos um deles nesse grupo.
            if remaining_agents > 0:
                groups[-1] += 1
                remaining_agents -= 1

            segments_count[i] = segments_count.get(i, 0) + groups[-1]

        # Garantimos que a quantidade de agentes
        #   restante é 0.
        assert remaining_agents == 0

    # Pós-condições para garantir que os grupos
    #   são consistentes.
    assert sum(groups) == market.agents
    assert len(groups) == n_groups
    assert all(segments_count[i] == segments[i]
               for i in range(len(segments)))

    return np.array(groups, dtype=np.int32)


def r_values_split(value: np.ndarray) -> tuple[np.ndarray,
                                               np.ndarray]:
    """Estratégia geral para obter os r-values a partir
    de um indivíduo. A ideia é dividir o indivíduo (array)
    no meio: a 1ª parte é r1; a 2ª parte é r2.

    Args:
        value (np.ndarray): array representando um indivíduo.

    Returns:
        tuple[np.ndarray, np.ndarray]: r1 e r2.
    """
    return tuple(np.split(value, 2))


def broadcast_group(value: np.ndarray,
                    groups: np.ndarray) -> np.ndarray:
    """Realiza o broadcast de um indivíduo que representa
    os valores de r1 e r2 por grupo, para sua representação
    completa (r1 e r2 por agente).

    A ideia é os valores de r1 e r2 por grupo sejam repetidos
    para a quantidade de agentes no grupo.

    Args:
        value (np.ndarray): indivíduo.
        groups (np.ndarray): quantidade de agentes por grupo.

    Returns:
        np.ndarray: representação do indivíduo com valores 
            de r1 e r2 por agente.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    # pylint: disable=invalid-name
    r1, r2 = r_values_split(value)
    r1 = np.repeat(r1, groups)
    r2 = np.repeat(r2, groups)

    return np.concatenate(r1, r2)


def r_values_split_w_broadcast(value: np.ndarray,
                               groups: np.ndarray) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Realiza o broadcast de um indivíduo que representa
    os valores de r1 e r2 por grupo, para sua representação
    completa (r1 e r2 por agente). Retorna a tupla com os valores
    de r1 e r2.

    Args:
        value (np.ndarray): indivíduo.
        groups (np.ndarray): quantidade de agentes por grupo.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuplas com r1 e r2.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    # pylint: disable=invalid-name
    r1, r2 = r_values_split(value)
    r1 = np.repeat(r1, groups)
    r2 = np.repeat(r2, groups)
    return r1, r2
