import plotly.express as px
import pandas as pd


class DensityHeatmap:
    """
    Example
    --------

    Use the DensityHeatmap to plot the heat map graph
    `Heat map <https://en.wikipedia.org/wiki/Heat_map>`_:

    .. code-block:: python
        :linenos:
    
    >>> import plotly.express as px
    >>> import pandas as pd
    >>> plot = DensityHeatmap('Dataset.csv', 'r_1', 'r_2', 'rmse',
                              histfunc='avg', nbinsx=21, nbinsy=21)
    >>> plot.generate_plot('R1', 'R2', width=800, height=600)
    """
        
    def __init__(self, data_file_path, x_col_name, y_col_name, z_col_name,
                 nbinsx=20, nbinsy=20, color_continuous_scale="Viridis", histfunc="sum") -> None:
        
        """
        Constructor for the DensityHeatmap class.
        
        Parameters:
        - data_file_path: str, path to the data file.
        - x_col_name: str, column name for the x-axis values.
        - y_col_name: str, column name for the y-axis values.
        - z_col_name: str | None, column name for the z-axis values.
        - nbinsx: int, optional, number of bins for x-axis. Default is 20.
        - nbinsy: int, optional, number of bins for y-axis. Default is 20.
        - color_continuous_scale: str, optional, color scale for the heatmap. Default is "Viridis".
        - histfunc: str, optional, aggregation function for the z-axis values. Default is "sum".
        """
        self.df = pd.read_csv(data_file_path)
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.z_col_name = z_col_name
        self.nbinsx = nbinsx
        self.nbinsy = nbinsy
        self.color_continuous_scale = color_continuous_scale
        self.histfunc = histfunc
    
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}(Dataset.csv, {self.x_col_name!r}, {self.y_col_name!r}, {self.z_col_name!r}, {self.nbinsx!r}, {self.nbinsy!r}, {self.color_continuous_scale!r}, {self.histfunc!r})'
    
    def generate_plot(self, xaxis_title, yaxis_title, height=500, width=300) -> None:

        """
        Generates a density heatmap plot using Plotly Express.
        
        Parameters:
        - xaxis_title: str, title for the x-axis.
        - yaxis_title: str, title for the y-axis.
        - height: int, optional, height of the plot in pixels. Default is 500.
        - width: int, optional, width of the plot in pixels. Default is 300.
        """
        fig = px.density_heatmap(self.df, x=self.x_col_name, y=self.y_col_name,
                                 z=self.z_col_name, nbinsx=self.nbinsx, nbinsy=self.nbinsy,
                                 color_continuous_scale=self.color_continuous_scale,
                                 histfunc=self.histfunc, labels=None)
        
        fig.update_traces(colorbar=dict(title=None))
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            width=width,
            showlegend=False 
        )
        fig.show()

if __name__ == "__main__":
    plot = DensityHeatmap('data.csv', 'r_1', 'r_2',
                                'rmse', histfunc='avg',
                                    nbinsx=21, nbinsy=21)
    print(plot)
    # plot.generate_plot('R1', 'R2', width=800, height=600)
