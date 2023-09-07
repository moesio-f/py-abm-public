"""Módulo com o parser de arquivos
de experimentação envolvendo o ABM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ijson

from py_abm.fitness.abm import Market, OptimizationType


@dataclass
class Individual:
    """Essa classe representa um indivíduo
    do processo de otimização.
    """
    r1: list[float]
    r2: list[float]
    array: list[float]


class ExperimentParser:
    """Essa classe realiza o parse de experimentos
    que produzem como resultado JSONs padronizados.
    """
    _HISTORY = 'history.json'
    _CONFIG = 'config.json'
    _RESULT = 'result.json'

    _BS_KEY = 'best_solution'
    _BF_KEY = 'best_fitness'
    _DEC_KEY = 'decisions'
    _DEC_COUNT_KEY = 'decisions_count'
    _FN_KEY = 'fitness'

    def __init__(self,
                 results_path: Path) -> None:
        """Construtor. Recebe o caminho com os
        3 arquivos de resultados gerados e
        expôe propriedades relativas aos resultados.

        Args:
            results_path: caminho para o diretório
                com os arquivos de resultado.
        """
        self._dir = results_path

        # Arquivos JSON potencialmente grandes,
        #   melhor guardar apenas a referência do caminho
        #   e usar o IJSON para parsing on-demand.
        self._history_path = self._dir.joinpath(self._HISTORY)
        self._results_path = self._dir.joinpath(self._RESULT)

        # Config costuma ser pequeno e pode ser carregado
        # diretamente
        self._config_path = self._dir.joinpath(self._CONFIG)
        config_text = self._config_path.read_text('utf-8')
        self._config = json.loads(config_text)

        self._cache = dict()

    @property
    def best_fitness(self) -> dict[str, float]:
        """Retorna o melhor valor para a função de
        fitness.

        Returns:
            dict[str, float]: dicionário objetivo -> valor.
        """
        with open(self._results_path, 'r') as result_file:
            best_fitness = next(ijson.items(result_file,
                                            self._BF_KEY,
                                            use_float=True))
        return best_fitness

    @property
    def best_individual(self) -> Individual:
        """Retorna o melhor indivíduo encontrado. Os
        r-values do indivíduo são iguais aos passados
        para o simulador.

        Returns:
            Individual: indivíduo.
        """
        with open(self._results_path, 'r') as result_file:
            best_individual = next(ijson.items(result_file,
                                               self._BS_KEY,
                                               use_float=True))
        return Individual(**best_individual)

    @property
    def best_fitness_per_milestone(self) -> list[dict[str, float]]:
        """Retorna o melhor valor de fitness por
        milestone.

        Returns:
            list[dict[str, float]]: melhor fitness por iteração.
        """
        key = 'best_fitness_per_milestone'

        if key not in self._cache:
            bf_milestone = []
            best_value = None
            with self._history_path.open('r') as hist_file:
                for entry in ijson.items(hist_file,
                                         'item',
                                         use_float=True):
                    objectives = entry['objectives']
                    if any(map(lambda d: len(d) > 1, objectives)):
                        raise ValueError('A implementação não suporta '
                                         'funções multi-objetivas.')
                    best_vale_for_entry = min(
                        map(lambda o: next(iter(o.values())),
                            objectives))
                    k = next(iter(objectives[0].keys()))

                    if best_value is None:
                        best_value = best_vale_for_entry
                    else:
                        best_value = min(best_value, best_vale_for_entry)

                    bf_milestone.append({k: best_value})

            self._cache[key] = bf_milestone

        return self._cache[key]

    @property
    def milestones(self) -> list[int]:
        """Retorna os milestones presentes no
        histórico.

        Returns:
            list[int]: milestones.
        """
        key = 'milestones'

        if key not in self._cache:
            with self._history_path.open('r') as hist_file:
                self._cache[key] = list(map(lambda d: d['milestone'],
                                            ijson.items(hist_file,
                                                        'item',
                                                        use_float=True)))

        return self._cache[key]

    @property
    def time_steps(self) -> int:
        """Retorna a quantidade de time steps.

        Returns:
            int: time steps.
        """
        return self._config[self._FN_KEY]['time_steps']

    @property
    def mc_runs(self) -> int:
        """Retorna a quantidade de mc runs.

        Returns:
            int: mc runs.
        """
        return self._config[self._FN_KEY]['mc_runs']

    @property
    def market(self) -> Market:
        """Retorna o mercado.

        Returns:
            Market: mercado
        """
        return Market.from_str(self._config[self._FN_KEY]['market'])

    @property
    def optimization_type(self) -> OptimizationType:
        """Retorna o tipo de otimização realizado.

        Returns:
            OptimizationType: tipo de otimização.
        """
        return OptimizationType.from_str(
            self._config[self._FN_KEY]['optimization_type'])

    @property
    def groups(self) -> bool | list[int]:
        """Retorna os grupos (se aplicável) ou False.

        Returns:
            bool | list[int]: grupos ou False.
        """
        return self._config[self._FN_KEY]['groups']

    @property
    def decisions_frequency(self) -> list[dict[str, float]]:
        """Retorna a frequência de heurísticas escolhidas
        por cada um dos agentes.

        Returns:
            list[dict[str, float]]: frequência de decisão por
                agente.
        """
        with self._results_path.open('r') as result_file:
            decisions_frequency = next(ijson.items(result_file,
                                                   self._DEC_KEY,
                                                   use_float=True))

        return decisions_frequency

    @property
    def decisions_count(self) -> list[list[dict[str, int]]]:
        """Retorna a frequência de heurísticas escolhidas
        por cada um dos agentes.

        Returns:
            list[dict[str, float]]: frequência de decisão por
                agente.
        """
        with self._results_path.open('r') as result_file:
            decisions_count = next(ijson.items(result_file,
                                               self._DEC_COUNT_KEY,
                                               use_float=False))

        return decisions_count

    @property
    def probabilities(self) -> list[dict[str, float]]:
        """Retorna as probabilidades de escolha das heurísticas
        por cada um dos agentes.

        Returns:
            list[dict[str, float]]: probabilidade de decisão
                de heurística por agente.
        """
        key = 'probabilities'

        if key not in self._cache:

            with self._results_path.open('r') as result_file:
                probabilities = next(ijson.items(result_file,
                                                 'probabilities',
                                                 use_float=True))
                opt = self.optimization_type
                market = self.market

                if opt == OptimizationType.GLOBAL:
                    probabilities = probabilities * market.agents
                elif len(probabilities) != market.agents:
                    # Caso seja otimização a nível de segmentos
                    #   ou agentes, podemos precisar fazer
                    #   o broadcast.
                    if opt == OptimizationType.SEGMENTS:
                        groups = market.agents_per_segment
                    else:
                        groups = ([1] * self.market.agents
                                  if not self.groups
                                  else self.groups)
                        assert sum(groups) == market.agents

                    assert len(probabilities) == len(
                        groups), f'{len(probabilities)} != {len(groups)}'

                    probabilities = [repeat
                                     for i, g in enumerate(groups)
                                     for repeat in [probabilities[i]] * g]

            assert len(probabilities) == market.agents
            self._cache[key] = probabilities

        return self._cache[key]

    def broadcast_r_value(self, r_value: list[float]) -> list[float]:
        """Faz o broadcast de uma lista de r_values (globa, agent,
        segments ou grupo) para sua versão por agente.

        Args:
            r_value (list[float]): r_values originais.

        Returns:
            list[float]: r_value por agente.
        """
        n_repeats = None

        if self.optimization_type == OptimizationType.GLOBAL:
            n_repeats = [self.market.agents]
        elif self.optimization_type == OptimizationType.SEGMENTS:
            n_repeats = self.market.agents_per_segment
        else:
            n_repeats = [1] * len(r_value)

        assert len(n_repeats) == len(r_value)
        result = [r
                  for i, s in enumerate(n_repeats)
                  for r in [r_value[i]] * s]
        assert len(result) == self.market.agents
        return result
