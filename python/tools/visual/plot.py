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


def linear_plot(data, name=None, max_sampling=1000, bias=None):
    import numpy as np
    from . import plot_utils as plt
    from .metrics import kurtosis
    from .metrics import skewness

    blob_fp, blob_int = data.tensor(name)
    data_size = blob_fp.size
    if data_size > max_sampling:
        step = data_size // max_sampling
    else:
        step = 1
    index = np.arange(0, data_size, step)
    kur=kurtosis(blob_fp)
    ske=skewness(blob_fp)
    fig = plt.plot_float_vs_fixpoint(
        index, (blob_fp.flatten()[::step], blob_int.flatten()[::step]),
        subplot_titles=(name+'[{:.2f}/{:.2f}]'.format(kur,ske), "float - int8\t "))
    fig.update_layout(legend=dict(yanchor="top", y=0.99,
                    xanchor="right", x=0.98, bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20),
                    hovermode='x unified')
    return fig

def param_plot(data, name=None, max_sampling=1000, bias=None):
    import numpy as np
    from . import plot_utils as plt
    shape = None
    if name is None:
        return None
    weight, weight_shape, bias, bias_shape = data.weight(name, bias)
    data_sizew = weight.size
    if data_sizew > max_sampling:
        stepw = data_sizew // max_sampling
    else:
        stepw = 1
    indexw = np.arange(0, data_sizew, stepw)
    if bias is not None:
        data_sizeb = bias.size
        if data_sizeb > max_sampling:
            stepb = data_sizeb // max_sampling
        else:
            stepb = 1
        indexb = np.arange(0, data_sizeb, stepb)
    else:
        indexb = None

    if bias is not None:
        fig = plt.plot_weight_and_transposed(indexw, indexb, (weight.flatten()[::stepw], weight.transpose().flatten()[::stepw], bias.flatten()[::stepb]),
            subplot_titles=(name, name+"-transposed", name+"-bias"))
    else:
        fig = plt.plot_weight_and_transposed(indexw, indexb, (weight.flatten()[::stepw], weight.transpose().flatten()[::stepw], None),
            subplot_titles=(name, name+"-transposed", name+"-bias"))
    fig.update_layout(legend=dict(yanchor="top", y=0.99,
                    xanchor="right", x=0.98, bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20),
                    hovermode='x unified')
    return fig

def dist_plot(data, name=None, scale=1.0, dtype='I8'):
    import numpy as np
    from . import plot_utils as plt
    import matplotlib.pyplot as pyplt
    blob_fp, blob_int = data.tensor(name)
    d_max = np.maximum(np.max(blob_fp), 0)
    d_min = np.minimum(np.min(blob_fp), 0)
    fig = make_subplots(rows=1,cols=1,shared_xaxes=True)
    _,ax = pyplt.subplots()
    if dtype == 'I8':
        min_ = -128*scale
        max_ = 127*scale
        max = np.maximum(max_,d_max)
        min = np.minimum(min_,d_min)
        min_bin = int(np.abs(np.floor(min/scale)))
        max_bin = int(np.ceil(max/scale))
        bins = min_bin+max_bin
        xticklabels=['']*bins
        xticklabels[min_bin-128] = '-128/{:.4f}'.format(-128.0*scale)
        xticklabels[min_bin] = '0/0.0'
        xticklabels[min_bin+126] = '127/{:.4f}'.format(127.0*scale)

        hist = np.histogram(blob_fp, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
        hist_fp = hist[0]
        hist = np.histogram(blob_int, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
        hist_quant = hist[0]
        q_range = np.zeros(bins)
        q_range[min_bin-128:min_bin+127] = np.ones(255)*np.maximum(np.max(hist_fp),np.max(hist_quant))
        index = np.arange(bins)
        fig = plt.plot_dist_fp_fixpoint( fig,
            index, (hist_fp, hist_quant, q_range),
            subplot_titles=(name, "dist fp vs int8\t "))
    elif dtype == 'U8':
        min_ = 0
        max_ = 255*scale
        max = np.maximum(max_,d_max)
        min = np.minimum(min_,d_min)
        min_bin = int(np.abs(np.floor(min/scale)))
        max_bin =int(np.ceil(max/scale))
        bins = min_bin+max_bin
        xticks = [min_bin, min_bin+255]
        xticklabels=['']*bins
        xticklabels[min_bin] = '0/0.0'
        xticklabels[min_bin+254] = '255/{:.4f}'.format(255.0*scale)
        hist = np.histogram(blob_fp, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
        hist_fp = hist[0]
        hist = np.histogram(blob_int, bins=bins,
                        range=(-min_bin*scale, max_bin*scale))
        hist_quant = hist[0]
        q_range = np.zeros(bins)
        q_range[min_bin:min_bin+255] = np.ones(255)*np.maximum(np.max(hist_fp),np.max(hist_quant))
        index = np.arange(bins)
        fig = plt.plot_dist_fp_fixpoint( fig,
            index, (hist_fp, hist_quant, q_range),
            subplot_titles=(name, "dist fp vs uint8\t "))
    else:
        index = np.arange(255)
        hist = np.histogram(blob_fp, bins=255,
                        range=(np.min(blob_fp), np.max(blob_fp)))
        hist_fp = hist[0]
        fig = plt.plot_dist_fp_fixpoint( fig,
            index, (hist_fp, [], []),
            subplot_titles=(name, "dist fp\t "))
        xticklabels = ['']*255
        bins=255
        xticklabels[0] = '{:.4f}'.format(np.min(blob_fp))
        xticklabels[254] = '{:.4f}'.format(np.max(blob_fp))
        xticklabels[int(np.abs(np.min(blob_fp))/(np.max(blob_fp)-np.min(blob_fp)+1e-4)*255)] = '0.0'

    fig.update_xaxes(tickmode='array', tickvals=np.arange(bins), ticktext=xticklabels, type='linear')
    fig.update_layout(margin=dict(l=20, r=20, t=5, b=5),
                      xaxis_visible=True,
                      xaxis_title=None,
                      xaxis_showticklabels=True,
                      xaxis_showgrid=True,
                      xaxis_showline=True,
                      yaxis_showgrid=True,
                      yaxis_title=None,
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
