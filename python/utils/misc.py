import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_tensor_diff_subplot(tensor_ref, tensor_target, index_list, ref_name, target_name, file_prefix):
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("{} vs {}".format(target_name, ref_name)))
    fig.add_trace(go.Scattergl(y=tensor_ref.reshape([-1]),
                                x=index_list,
                                name=ref_name,
                                mode='lines+markers',
                                marker={
                                    "size": 6,
                                    "symbol": 300
                                },
                                line={"width": 1}),
                    row=1,
                    col=1)
    fig.add_trace(go.Scattergl(y=tensor_target.reshape([-1]),
                                x=index_list,
                                name=target_name,
                                mode='lines+markers',
                                marker={
                                    "size": 6,
                                    "symbol": 304,
                                    "opacity": 0.8
                                },
                                line={"width": 1}),
                    row=1,
                    col=1)

    fig.update_layout(height=400, width=900)
    fig.write_html(file_prefix+'.html')
    fig.write_image(file_prefix+'.png')
