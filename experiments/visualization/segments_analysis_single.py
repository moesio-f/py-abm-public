"""Esse script cria os plots considerando dados
de um único experimento e apenas o comportamento
dos segmentos.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from py_abm.fitness.abm import Heuristic, Market
from py_abm.visualization.analysis import extraction
from py_abm.visualization.parser import ExperimentParser, Individual
from py_abm.visualization.plots import (BoxPlotR1R2, Convergence, Histogram,
                                        Scatter)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Geração de gráficos para experimento "
                     "considerando segmentos."))

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
    target = Path(args.output_dir).joinpath('segment-analysis')
    target.mkdir(exist_ok=True, parents=True)
    scale = args.scale

    # Carregando resultados
    logger.info('Carregando resultados para o experimento %s...',
                str(source))
    experiment = ExperimentParser(source)
    r1 = experiment.broadcast_r_value(experiment.best_individual.r1)
    r2 = experiment.broadcast_r_value(experiment.best_individual.r2)
    decisions = experiment.decisions_frequency
    probabilities = experiment.probabilities
    market = experiment.market
    segments_description = market.segments_desc
    assert market in {Market.FASTFOOD, Market.DAIRIES}
    assert len(r1) == len(r2)
    assert len(r1) == market.agents

    # Cores das heurísticas
    heuristic_colors = {
        'UMAX': '#636EFA',
        'SAT': '#EF553B',
        'EBA': '#2ca02c',
        'MAJ': '#9467bd',
    }

    # Símbolos dos segmenos
    segment_symbol = {k: v for k, v in zip(segments_description,
                                           ['circle', 'square', 'diamond',
                                            'cross', 'x', 'triangle-up',
                                            'star'])}

    # Cores dos segmentos
    segment_colors = {k: v for k, v in zip(segments_description,
                                           px.colors.qualitative.Antique)}

    # Configurações do heatmap
    r_range = (0.0, 1.0)
    nbins = 100
    axis_ticks = dict(tickmode='linear',
                      tick0=0.0,
                      dtick=0.1)

    # Análise por segmento
    logger.info('Gerando perfil dos segmentos...')

    for i in range(market.segments):
        logger.info(f'Segmento {i}...')
        start = 0 if i == 0 else np.cumsum(market.agents_per_segment)[i - 1]
        end = start + market.agents_per_segment[i]
        r1_seg = r1[start:end]
        r2_seg = r2[start:end]
        decisions_seg = decisions[start:end]
        probabilities_seg = probabilities[start:end]
        segment_desc = segments_description[i]
        segment_id = f'segment {i} ({segment_desc})'
        t_ = target.joinpath(f'segment-{i}')
        t_.mkdir(exist_ok=True, parents=False)

        logger.info('Gerando box-plot para os valores r-values...')
        box_plot_r_values = BoxPlotR1R2(
            r1_values=r1_seg,
            r2_values=r2_seg,
            title=f'Boxplot r1 and r2 values for {segment_id}')
        box_plot_r_values.save(t_.joinpath('boxplot.jpeg'),
                               resolution=scale)

        logger.info('Gerando heatmap para os r-values...')
        heatmap = px.density_heatmap(x=r1,
                                     y=r2,
                                     range_x=r_range,
                                     range_y=r_range,
                                     nbinsx=nbins,
                                     nbinsy=nbins,
                                     title=('Heatmap of r-values for '
                                            f'{segment_id}'),
                                     width=600,
                                     height=600)
        heatmap.update_layout(xaxis_title='R1',
                              yaxis_title='R2',
                              xaxis=axis_ticks,
                              yaxis=axis_ticks)
        heatmap.write_image(t_.joinpath('heatmap.jpeg'),
                            scale=scale)

        logger.info('Gerando pie chart das heurísticas....')

        def _pie(dicts: list[dict],
                 title: str):
            count = Counter([max(d, key=d.get)
                             for d in decisions_seg])
            df = pd.DataFrame({
                'heuristic': count.keys(),
                'count': count.values()
            })

            fig = px.pie(df,
                         names='heuristic',
                         values='count',
                         color='heuristic',
                         color_discrete_map=heuristic_colors,
                         title=title)
            fig.update_traces(textinfo='percent+value')
            return fig
        fig = _pie(decisions_seg,
                   f'Most chosen Heuristic for {segment_id}')
        fig.write_image(t_.joinpath('pie_chart_decisions.jpeg'),
                        scale=scale)

        fig = _pie(probabilities_seg,
                   f'Heuristic with greatest probability for {segment_id}')
        fig.write_image(t_.joinpath('pie_chart_probability.jpeg'),
                        scale=scale)

        # logger.info('Gerando medidas de estatísticas...')
        # TODO: depois pesquisar as métricas utilizadas

    # Configurações dos scatter
    opacity = 0.5
    font_size = 6

    # Criação do scatter plot considerando a heurística mais utilizada
    logger.info('Criando scatter plot das decisões...')
    logger.info('Coletando heurística mais utilizada por agente...')
    points = extraction.max_heuristic(experiment, probability=False)
    points = points.sort_values(by='Heuristic')
    scatter_decision = Scatter(points,
                               x_column='R1',
                               y_column='R2',
                               color_column='Heuristic',
                               symbol_column='Segment',
                               size_column='Size',
                               color_discrete_map=heuristic_colors,
                               title='Most chosen Heuristic by Agent',
                               range_x=r_range,
                               range_y=r_range,
                               opacity=opacity,
                               legend_font_size=font_size)
    scatter_decision.save(target.joinpath('scatter_segments_decisions.jpeg'),
                          resolution=scale,
                          xaxis=axis_ticks,
                          yaxis=axis_ticks)

    # Criação dos gráficos de pizza por Heurística
    def _pie(points: pd.DataFrame,
             title: str,
             prefix: str):
        total = points.groupby(['Heuristic', 'Segment'],
                               as_index=False).sum()
        total = total[['Heuristic', 'Segment', 'Size']]

        for h in Heuristic:
            name = h.value
            df = total[total.Heuristic == name].sort_values(by='Segment')
            tot = df.Size.sum().item()
            fig = px.pie(df,
                         names='Segment',
                         values='Size',
                         color='Segment',
                         color_discrete_map=segment_colors,
                         title=title.format(tot, name))
            fig.update_traces(textinfo='percent+value')
            fig.write_image(
                target.joinpath(f'pie_chart_{prefix}_{name}.jpeg'),
                scale=scale)

    _pie(points,
         title='Agents (total: {}) with {} as most chosen heuristic',
         prefix='decisions')

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
        symbol_column='Segment',
        color_discrete_map=heuristic_colors,
        title='Max probability Heuristic for Agent',
        range_x=r_range,
        range_y=r_range,
        opacity=opacity,
        legend_font_size=font_size)
    scatter_probabilities.save(
        target.joinpath('scatter_segments_probabilities.jpeg'),
        resolution=scale,
        xaxis=axis_ticks,
        yaxis=axis_ticks)

    _pie(points,
         title='Agents (total: {}) with {} as likeliest heuristic',
         prefix='probabilities')

    logger.info('Processo finalizado.')
