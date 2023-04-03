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
    fig.update_layout(legend=dict(yanchor="top", y=0.99,
                      xanchor="right", x=0.98, bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20),
                      hovermode='x unified')
    return fig


def dist_plot(data, name=None, scale=1.0, dtype='I8'):
    import numpy as np
    from . import plot_utils as plt
    blob_fp, blob_int = data.tensor(name)
    if dtype == 'I8':
        min_ = -128*scale
        max_ = 127*scale
    elif dtype == 'U8':
        min_ = 0
        max_ = 255*scale
    max_f = np.max(np.max(blob_fp), 0)
    min_f = np.min(np.min(blob_fp), 0)
    max_bin = np.ceil(max_f/scale)
    min_bin = np.abs(np.floor(min_f/scale))
    bins = int(max_bin+min_bin)
    hist = np.histogram(blob_fp, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
    hist_fp = hist[0]
    hist = np.histogram(blob_int, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
    hist_quant = hist[0]
    index = np.arange(bins)
    fig = plt.plot_dist_fp_fixpoint(
        index, (hist_fp, hist_quant),
        subplot_titles=(name, "dist fp vs int8\t "))
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
            max_value = np.max(tensor_info[name].values, where=(
                tensor_info[name] != math.inf), initial=1.0)
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
    fig.update_xaxes(tickmode='array', tickvals=np.arange(
        len(x_name)-1), ticktext=x_name, range=[-0.5, len(x_name)], type='linear')
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
