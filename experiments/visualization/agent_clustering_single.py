"""Esse script cria os plots considerando dados
de um único experimento e apenas a solução final
de otimizações a nível de agentes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

from py_abm.fitness.abm import Heuristic, OptimizationType, Market
from py_abm.visualization.analysis.clustering import (ClusterABMSolution,
                                                      ClusterFeatures)
from py_abm.visualization.parser import ExperimentParser
from py_abm.visualization.plots import Scatter

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Geração de gráficos para experimento "
                     "considerando o agrupamento de agentes."))

    parser.add_argument('experiment_directory',
                        help='Diretório com os resultados da execução.',
                        type=str)

    parser.add_argument('-o',
                        '--output-directory',
                        dest='output_dir',
                        help='Diretório para salvamento dos gráficos.',
                        type=str,
                        default='.')

    parser.add_argument('--scale',
                        dest='scale',
                        help='Diretório para salvamento dos gráficos.',
                        type=int,
                        default=4.0)

    args = parser.parse_args()
    source = Path(args.experiment_directory)
    target = Path(args.output_dir).joinpath('agent-clustering')
    target.mkdir(exist_ok=True, parents=True)
    scale = args.scale

    # Carregando resultados
    logger.info('Carregando resultados para o experimento %s...',
                str(source))
    experiment = ExperimentParser(source)
    assert experiment.optimization_type == OptimizationType.AGENT
    assert experiment.market in {Market.DAIRIES, Market.FASTFOOD}

    # Configurações dos scatter
    r_range = (0.0, 1.0)
    axis_ticks = dict(tickmode='linear',
                      tick0=0.0,
                      dtick=0.05)
    opacity = 0.5

    for features in [[ClusterFeatures.RVALUES],
                     [ClusterFeatures.BEHAVIOR],
                     [ClusterFeatures.PROBABILITY],
                     [ClusterFeatures.RVALUES,
                      ClusterFeatures.BEHAVIOR,
                      ClusterFeatures.PROBABILITY]]:
        features_names = '_'.join([f.value for f in features])
        feature_path = target.joinpath(f'clustering_{features_names}')
        feature_path.mkdir(parents=False, exist_ok=True)

        # Criação do scatter plot considerando o agrupamento.
        logger.info(f'Coletando clusters dada as features: {features_names}')
        cluster = ClusterABMSolution(experiment, cluster_by=features)
        clusters = cluster.get_clusters()
        points = pd.DataFrame(list(map(asdict, clusters)))
        points = points.sort_values(by='cluster').astype({'cluster': str})

        scatter = Scatter(
            points,
            x_column='r1',
            y_column='r2',
            color_column='cluster',
            title=f'Clusters with KMeans ({features_names})',
            range_x=r_range,
            range_y=r_range,
            opacity=opacity)
        scatter.save(
            feature_path.joinpath(f'scatter_clusters.jpeg'),
            resolution=scale,
            xaxis=axis_ticks,
            yaxis=axis_ticks)

        # Calculando métricas
        y_true = []
        y_pred = []
        for r in clusters:
            y_true.append(r.segment)
            y_pred.append(r.cluster)

        results = {
            'rand_index': [metrics.rand_score(y_true,
                                              y_pred)],
            'adj_rand_index': [metrics.adjusted_rand_score(y_true,
                                                           y_pred)],
            'norm_mutual_inf': [metrics.normalized_mutual_info_score(y_true,
                                                                     y_pred)],
            'adj_mutual_inf': [metrics.adjusted_mutual_info_score(y_true,
                                                                  y_pred)],
            'homogeneity': [metrics.homogeneity_score(y_true,
                                                      y_pred)],
            'completeness': [metrics.completeness_score(y_true,
                                                        y_pred)],
            'v-measure': [metrics.v_measure_score(y_true,
                                                  y_pred)]
        }

        # Salvando CSVs
        points.to_csv(
            feature_path.joinpath(f'results.csv'),
            index=False)
        pd.DataFrame(results).to_csv(
            feature_path.joinpath(f'metrics.csv'),
            index=False)

        # Gerando gráficos por segmento
        for i, seg_desc in enumerate(experiment.market.segments_desc):
            seg_path = feature_path.joinpath(f'segment-{i}')
            seg_path.mkdir(parents=False, exist_ok=True)

            points_segment = points[points.segment == i]
            scatter = Scatter(
                points_segment,
                x_column='r1',
                y_column='r2',
                color_column='cluster',
                title=f'Segment {i} ({seg_desc})',
                range_x=r_range,
                range_y=r_range,
                opacity=opacity)
            scatter.save(
                seg_path.joinpath(f'agents_and_groups.jpeg'),
                resolution=scale,
                xaxis=axis_ticks,
                yaxis=axis_ticks)

    logger.info('Processo finalizado.')
