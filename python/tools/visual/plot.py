import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class figure_cache():
    def __init__(self, data):
        self.data = data
        self.figure = dict()

    def get_figure(self, fun, **kwargs):
        key = (fun, tuple(sorted(kwargs.items())))
        if key in self.figure:
            return self.figure[key]
        fig = fun(self.data, **kwargs)
        self.figure[key] = fig
        return self.figure[key]

    def invalid_cache(self):
        self.figure = dict()


def linear_plot(data, name=None, max_sampling=1000):
    import numpy as np
    from . import plot_utils as plt
    blob_fp, blob_int = data.tensor(name)
    data_size = blob_fp.size
    if data_size > max_sampling:
        step = data_size // max_sampling
    else:
        step = 1
    index = np.arange(0, data_size, step)
    fig = plt.plot_float_vs_fixpoint(
        index, (blob_fp.flatten()[::step], blob_int.flatten()[::step]),
        subplot_titles=(name, "float - int8\t "))
    fig.update_layout(margin=dict(l=24, r=10, t=20, b=60),
                      hovermode='x unified')
    return fig


def metrics_plot(tensor_info):
    from .metrics import metrics

    x = [str(x) for x in tensor_info.index]
    x_name = tensor_info.tensor.values

    fig = go.Figure()

    for name in metrics.keys():
        name = name.upper()
        if name == "SQNR":
            import math
            max_value = np.max(tensor_info[name].values, where=(tensor_info[name] != math.inf), initial=1.0)
            new_value = np.clip(tensor_info[name]/max_value, 0, 1)
            name = name + "/" + "{:.1f}".format(max_value)
        else:
            new_value = np.clip(tensor_info[name].values, 0, 1)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(x_name)),
                y=new_value,
                name=name,
                text=x,
                customdata=x_name,
                hovertemplate='%{y}',
            ))
    fig.update_traces(mode="lines+markers")
    fig.update_yaxes(range=[-0.05, 1.05])
    fig.update_xaxes(tickmode='array', tickvals=np.arange(len(x_name)-1), ticktext=x_name, range=[-0.5, len(x_name)], type='linear')
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10),
                      xaxis_visible=True,
                      xaxis_title=None,
                      xaxis_showticklabels=False,
                      xaxis_showgrid=True,
                      xaxis_showline=True,
                      yaxis_showgrid=False,
                      yaxis_title=None,
                      hovermode='x unified')

    return fig
