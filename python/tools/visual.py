#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
import argparse

import dash
import visual.component as component
import visual.callback as callback
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc


app = dash.Dash(__name__,
                title="TPU-MLIR Calibration-Analysis-Tool",
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        dest='debug',
                        help='debug mode')
    parser.add_argument('-p',
                        '--port',
                        type=int,
                        default=10000,
                        dest='port',
                        help='server port number, default:10000')
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        dest='host',
                        help='host ip, default:0.0.0.0')
    parser.add_argument('--f32_mlir',
                        type=str,
                        default='',
                        dest='f32_mlir',
                        help='float tpu mlir file')
    parser.add_argument('--quant_mlir',
                        type=str,
                        default='',
                        dest='quant_mlir',
                        help='quantized tpu mlir file')
    parser.add_argument('--input',
                        type=str,
                        default='',
                        dest='input',
                        help='input to nets, either ends with .npz or ends with .jpg/.png')
    parser.add_argument('--manual_run',
                        action='store_true',
                        dest='manual_run',
                        help='do not auto forward the net when loaded')

    return parser


class global_state():
    def __init__(self):
        self.graph = None
        self.analysis_data = None
        self.figure_cache = None
        self.dist_cache = None
        self.weight_cache = None
        self.draggable = None
        self.zIndex = 50
        self.f32_mlir = ""
        self.quant_mlir = ""
        self.input = ""
        self.manual_run = False


# global variable
app.Global = global_state()

# App
app.layout = html.Div(
    style=component.page_style,
    children=[
        component.auto_load(),
        dbc.Row([
            dbc.Col(
                component.top_label('top-label'),
            ),
            dbc.Col(
                dbc.Row(
                    [
                        dbc.Col(
                            component.forward_buttom("forward", "forward-N"),
                            width="auto",
                        ),
                        dbc.Col(
                            component.mini_menu(
                                "mini-menu",
                                "select-layout",
                                "show-toolbox",
                                "show-figure",
                                "split-HW",
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                ),
                width="auto",
                align="center",
            )
        ],
            justify="between"),
        dcc.Store(id='forward-state',
                  data="Inactive"),  # state machine: Idle, Forward, Metrics
        dcc.Interval(id='forward-interval', interval=800, disabled=True),
        dbc.Progress(id='forward-progress',
                     value=0,
                     animated=True,
                     striped=True,
                     style={"height": "3px"}),
        dcc.Store(id='cytoscape-style', data=component.cy_stylesheet),
        dcc.Store(id='cytoscape-edge-info', data=dict()),
        component.split_panel(
            'split-panel',
            split='horizontal',
            size="120pt",
            children=[
                component.metrics_figure('metrics-figure'),
                component.split_panel(
                    "split-figure",
                    split='vertical',
                    size="40%",
                    children=[
                        html.Div(
                            component.cyto_graph(
                                'cytoscape-responsive-layout'),
                        ),
                        html.Div(
                            component.info_tab(
                                'info-tab', 'info-tab0', 'info-tab1', 'info-tab2', 'info-tab3'),
                            style={'height': '100vh'}),
                    ]),
            ]),
        component.draggable_toolbox('draggable-toolbox', 'toolbox'),
        dcc.Location(id='url'),
        html.Div(id='viewport-container',
                 children="test",
                 style={'display': 'none'}),
        html.Div(id='viewport-container-cy',
                 children="test",
                 style={'display': 'none'}),
        html.Div(id='viewport-container-edge-info',
                 children="test",
                 style={'display': 'none'}),
    ])

for call in callback:
    call(app)

if __name__ == '__main__':
    args = parse_args().parse_args()
    app.Global.f32_mlir = args.f32_mlir
    app.Global.quant_mlir = args.quant_mlir
    app.Global.input = args.input
    app.Global.manual_run = args.manual_run
    if args.debug:
        app.run_server(host=args.host, port=args.port, debug=True)
    else:
        app.run_server(host=args.host, port=args.port, debug=False,
                       dev_tools_silence_routes_logging=True, dev_tools_serve_dev_bundles=False)
