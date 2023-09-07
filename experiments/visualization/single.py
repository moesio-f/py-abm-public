"""Esse script cria os plots considerando dados
de um único experimento.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from py_abm.fitness.abm import Heuristic
from py_abm.visualization.analysis import extraction
from py_abm.visualization.parser import ExperimentParser
from py_abm.visualization.plots import (BoxPlotR1R2, Convergence, Histogram,
                                        Scatter)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Geração de gráficos para experimento.")

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
    target = Path(args.output_dir)
    target.mkdir(exist_ok=True, parents=True)
    scale = args.scale

    # Carregando resultados
    logger.info('Carregando resultados para o experimento %s...',
                str(source))
    experiment = ExperimentParser(source)
    r1 = experiment.best_individual.r1
    r2 = experiment.best_individual.r2

    # Criação do gráfico de convergência
    logger.info('Criando o gráfico de convergência para o erro RMSE...')
    values = extraction.best_fitness_per_milestone(experiment, 'RMSE')
    convergence = Convergence(values=values,
                              labels=experiment.milestones,
                              y_axis='Best Fitness',
                              x_axis='No. Fitness Evaluations')
    convergence.save(str(target.joinpath('convergence.jpeg')),
                     resolution=scale)

    # Criação do Box plot para os valores de r1 e r2
    logger.info('Criando Box Plot para os r-values...')
    box_plot_r_values = BoxPlotR1R2(r1_values=r1,
                                    r2_values=r2)
    box_plot_r_values.save(target.joinpath('boxplot_r_values.jpeg'),
                           resolution=scale)

    # Cores dos histogramas
    heuristic_color_map = {
        'UMAX': '#636EFA',
        'SAT': '#EF553B',
        'EBA': '#2ca02c',
        'MAJ': '#9467bd',
    }

    # Histograma dos valores de r1 e r2 da melhor solução
    logger.info('Criando histogramas para os r-values da melhor solução...')
    for r, name in zip([r1, r2], ['r1', 'r2']):
        histogram_r_value = Histogram(
            df=pd.DataFrame({name: r}),
            x_column=name,
            y_axis_title='No. of occurences in best solution',
            title=(f'Distribution of {name}-values for agent '
                   'in best solution'),
            x_axis_bins=dict(start=-0.1,
                             end=1.0,
                             size=0.2),
            bargap=0.2)

        histogram_r_value.save(
            target.joinpath(f'histogram_{name}_values.jpeg'),
            resolution=scale)

    # Criação do histograma das decisões
    logger.info('Criando histograma de decisões...')
    logger.info('Coletando heurísticas escolhidas pelos agentes...')
    decisions = extraction.chosen_heuristics(experiment)
    decisions = decisions.sort_values(by='Heuristic')

    logger.info('OBS:. pode existir erros de aproximação nas '
                'frequência de heurísticas escolhidas pelos agentes.\n'
                'O total de decisões coletadas foi %d.', len(decisions))
    histogram_decisions = Histogram(
        df=decisions,
        x_column='Heuristic',
        color_column='Heuristic',
        color_discrete_map=heuristic_color_map,
        y_axis_title='No. of times chosen',
        title='Chosen heuristics histogram (best solution)')
    histogram_decisions.save(
        target.joinpath('histogram_decisions.jpeg'),
        resolution=scale)

    # Criação do histograma considerando as probabilidades de decisão
    logger.info('Criando histograma com heurísticas '
                'com maior probabilidade...')
    logger.info('Coletando heurísticas com maior probabilidade...')
    max_heuristics = pd.DataFrame({
        'Heuristic': [max(d, key=d.get)
                      for d in experiment.probabilities]
    }).sort_values(by='Heuristic')
    histogram_probabilities = Histogram(
        df=max_heuristics,
        x_column='Heuristic',
        color_column='Heuristic',
        color_discrete_map=heuristic_color_map,
        y_axis_title='No. of Agents',
        title='Max probability heuristic histogram (best solution)')
    histogram_probabilities.save(
        target.joinpath('histogram_probabilities.jpeg'),
        resolution=scale)

    # Configurações dos scatter
    r_range = (0.0, 1.0)
    axis_ticks = dict(tickmode='linear',
                      tick0=0.0,
                      dtick=0.05)
    opacity = 0.5

    # Criação do scatter plot considerando a heurística mais utilizada
    logger.info('Criando scatter plot das decisões...')
    logger.info('Coletando heurística mais utilizada por agente...')
    points = extraction.max_heuristic(experiment, probability=False)
    points = points.sort_values(by='Heuristic')
    scatter_decision = Scatter(points,
                               x_column='R1',
                               y_column='R2',
                               color_column='Heuristic',
                               size_column='Size',
                               color_discrete_map=heuristic_color_map,
                               title='Most chosen Heuristic by Agent',
                               range_x=r_range,
                               range_y=r_range,
                               opacity=opacity)
    scatter_decision.save(target.joinpath('scatter_decisions.jpeg'),
                          resolution=scale,
                          xaxis=axis_ticks,
                          yaxis=axis_ticks)
    points.to_csv(target.joinpath('scatter_decisions.csv'),
                  index=False,
                  float_format="%.2f")

    # Criação do scatter plot considerando a heurística com maior probabilidade
    logger.info('Criando scatter plot das probabilidades...')
    logger.info('Coletando heurística com maior probabilidade por agente...')
    points = extraction.max_heuristic(experiment, probability=True)
    points = points.sort_values(by='Heuristic')
    scatter_probabilities = Scatter(
        points,
        x_column='R1',
        y_column='R2',
        color_column='Heuristic',
        size_column='Size',
        color_discrete_map=heuristic_color_map,
        title='Max probability Heuristic for Agent',
        range_x=r_range,
        range_y=r_range,
        opacity=opacity)
    scatter_probabilities.save(
        target.joinpath('scatter_probabilities.jpeg'),
        resolution=scale,
        xaxis=axis_ticks,
        yaxis=axis_ticks)
    points.to_csv(target.joinpath('scatter_probabilities.csv'),
                  index=False,
                  float_format="%.2f")

    logger.info('Processo finalizado.')
