"""Esse script cria os plots considerando dados
de múltiplos experimentos com as mesmas configurações.
"""
import argparse
from pathlib import Path

from boxplot_for_multiple_fitness_functions import create_boxplot
from convergence import create_convergence_graph

if __name__ == '__main__':
    # TODO: adicionar convergência média
    # TODO: adicionar box-plot das funções de fitness
    # TODO: adicionar comportamento médio (decisões) das soluções finais - Histograma
    # TODO: adicionar probabilidades média das soluções finais - Histograma
    parser = argparse.ArgumentParser(
        description='Script to aggregate data from multiple experiments.')

    parser.add_argument('root_path', metavar='path', type=str,
                        help='Root path of the experiment data directories')
    parser.add_argument('--output_dir', metavar='path',
                        type=str, help='Output directory for saving images')

    args = parser.parse_args()

    # Access the root path
    root_path = Path(args.root_path)

    # Get all subdirectories within the root path
    subdirectories = [directory for directory in root_path.glob('*/')]

    # BoxPlot
    boxplot_plot = create_boxplot(subdirectories)
    boxplot_plot.show()
    boxplot_output_path = Path(
        args.output_dir) if args.output_dir else root_path
    boxplot_plot.save(boxplot_output_path /
                      'BoxPlotForRmseInMultiplesJsonFiles.png')

    # Convergence Graph
    convergence_plot = create_convergence_graph(subdirectories)
    convergence_plot.show()
    convergence_output_path = Path(
        args.output_dir) if args.output_dir else root_path
    convergence_plot.save(convergence_output_path / 'Image.png')

""" Bash Command
python aggregate_Main.py db/ --output_dir /Storage/
"""
