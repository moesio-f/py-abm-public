"""Esse módulo contém a implementação
de uma análise baseada em agrupamento
para as soluções do ABM.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import sklearn.cluster

from py_abm.fitness.abm import OptimizationType
from py_abm.visualization.parser import ExperimentParser, Individual

from . import utils


class ClusterFeatures(Enum):
    RVALUES = 'rvalues'
    PROBABILITY = 'probability'
    BEHAVIOR = 'behavior'


@dataclass(frozen=True)
class ClusteredAgent:
    r1: float
    r2: float
    agent: int
    segment: int
    cluster: int


class ClusterABMSolution:
    def __init__(self,
                 experiment: ExperimentParser,
                 cluster_by: list[ClusterFeatures]):
        """Construtor.

        Args:
            experiment (ExperimentParser): experimento.
            cluster_by (list[ClusterFeatures]): características
                que devem ser utilizados para o agrupamento.
            algorithm: objeto que possui fit/predict.
        """
        self._market = experiment.market
        self._opt_type = experiment.optimization_type
        self._ind = experiment.best_individual
        self._probs = experiment.probabilities
        self._decs = experiment.decisions_frequency
        self._features = cluster_by
        self._alg = sklearn.cluster.KMeans(
            n_clusters=self._market.segments)

    def get_clusters(self) -> list[ClusteredAgent]:
        assert self._opt_type == OptimizationType.AGENT

        features = np.array([self._get_feature(i)
                             for i in range(self._market.agents)],
                            dtype=np.float32)

        clusters = self._alg.fit_predict(features)
        result = []
        for agent, c in enumerate(clusters):
            result.append(ClusteredAgent(
                r1=self._ind.r1[agent],
                r2=self._ind.r2[agent],
                agent=agent,
                segment=utils.get_agent_segment(agent,
                                                self._market),
                cluster=c))

        return result

    def _get_feature(self, i: int) -> list[float]:
        features = []

        if ClusterFeatures.RVALUES in self._features:
            features.extend([self._ind.r1[i],
                             self._ind.r2[i]])

        if ClusterFeatures.BEHAVIOR in self._features:
            d = dict(sorted(self._decs[i].items()))
            features.extend(list(d.values()))

        if ClusterFeatures.PROBABILITY in self._features:
            d = dict(sorted(self._probs[i].items()))
            features.extend(list(d.values()))

        return features
