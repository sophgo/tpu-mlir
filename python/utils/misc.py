import os
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


def save_tensor_subplot(tensor, index_list, tensor_name, file_prefix):
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(tensor))

    fig.add_trace(go.Scattergl(y=tensor.reshape([-1]),
                                x=index_list,
                                name=tensor_name,
                                mode='lines+markers',
                                marker={
                                    "size": 6,
                                    "symbol": 300
                                },
                                line={"width": 1}),
                    row=1,
                    col=1)

    fig.update_layout(height=400, width=900)
    fig.write_html(file_prefix+'.html')
    fig.write_image(file_prefix+'.png')


def parse_debug_cmd(debug_cmd):
    debug_cmd_dict = {}
    for cmd in debug_cmd.split(';'):
        tmp = cmd.split('=')
        if len(tmp) == 1:
            debug_cmd_dict[tmp[0]] = None
        elif len(tmp) >= 2:
            debug_cmd_dict[tmp[0]] = '='.join(tmp[1:])
        else:
            print('error debug_cmd format')
            exit(1)
    return debug_cmd_dict


def show_mem_info(info):
    mem_info = [i for i in os.popen('free').readlines()[1].split(' ') if len(i.strip()) > 0]
    info_str = 'total mem is {}, used mem is {}'.format(mem_info[1], mem_info[2])
    info_str = '{}:{}'.format(info, info_str)
    print(info_str)
