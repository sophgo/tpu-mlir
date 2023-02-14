import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot(x, y, **keywords):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(y=y, x=x, **keywords))
    fig.show()


def fig_sub_plot(shape, **keywords):
    assert (len(shape) == 2)
    rows = shape[0]
    cols = shape[1]
    index = 0

    fig = make_subplots(rows=rows, cols=cols, **keywords)

    def add_plot(x, ys, sub=None, **keywords):
        nonlocal index
        if sub is None:
            sub = np.unravel_index(index, shape)
        else:
            index = np.ravel_multi_index(sub, dims=shape)

        def get_v(value, index):
            if isinstance(value, tuple):
                return value[index]
            return value

        if isinstance(ys, np.ndarray) or isinstance(ys, list):
            ys = (ys, )
        for i, y in enumerate(ys):
            args_ = {k: get_v(v, i) for k, v in keywords.items()}
            fig.add_trace(go.Scattergl(y=y, x=x, **args_),
                          row=sub[0] + 1,
                          col=sub[1] + 1)
        index += 1

    return fig, add_plot


def plot_float_vs_fixpoint(index, data, **keywords):
    fp_data, int_data = data
    fig, plot = fig_sub_plot([2, 1],
                             **dict(shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    horizontal_spacing=0.07,
                                    **keywords))
    style = dict(name=('float32', 'int8'),
                 mode='lines+markers',
                 marker=({
                     "size": 6,
                     "symbol": 300
                 }, {
                     "size": 6,
                     "symbol": 304,
                     "opacity": 0.8
                 }),
                 line={"width": 1})
    plot(index, (fp_data, int_data), **style)
    plot(index, fp_data - int_data, **dict(name='diff', line={"width": 1}))
    return fig
