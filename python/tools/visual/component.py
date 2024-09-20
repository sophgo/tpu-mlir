import dash_bootstrap_components as dbc
from dash import html
import dash_draggable as ddrage
from dash import dcc
import dash_split_pane as dsp
from dash import dash_table as dtb
import numpy as np

draggable_layout = dict(width=2400, gridCols=24)

page_style = {
    'position': 'fixed',
    'display': 'flex',
    'flexDirection': 'column',
    'height': '100%',
    'width': '100%'
}

cy_stylesheet = [{
    "selector": 'node',
    "style": {
        "shape": "round-rectangle",
        'width': "data(width)",
        'height': 23,
        "background-color": "#023e8a",
        "content": "data(label)",
        "font-size": "10px",
        "text-valign": "center",
        "text-halign": "center",
        "color": "#ffffff",
        'min-zoomed-font-size': '4px',
        "overlay-padding": "6px",
        "zIndex": "10"
    }
}, {
    "selector": 'edge',
    'style': {
        "curve-style": "bezier",
        'target-arrow-shape': 'vee',
        'width': 0.9,
        'label': 'data(label)',
        'arrow-scale': 0.6,
        'line-color': '#001219',
        'target-arrow-color': '#001219',
        'text-background-shape': 'round-rectangle',
        'text-background-padding': '4px',
        'text-background-opacity': 0.7,
        'text-background-color': '#F5F5F5',
        'color': 'SteelBlue',
        'font-size': '8px',
        'min-zoomed-font-size': '6px',
        'text-wrap': 'wrap',
        'line-height': 1.2,
    }
}, {
    "selector": '.edge-flash',
    'style': {
        "border-style": 'dashed',
        'line-style': 'dashed',
        'text-background-color': '#fca311',
        'color': 'white',
        'width': 5,
        "opacity": 1,
    }
}, {
    "selector": '.all-fade-flash',
    'style': {
        "opacity": 0.2,
    }
}, {
    "selector": '.fake-node',
    'style': {
        'visibility': 'hidden'
    }
}, {
    "selector": ':selected',
    'style': {
        "background-color": '#e63946',
        'line-color': '#e63946',
        'target-arrow-color': '#e63946'
    }
}]


def cyto_graph(id):
    import dash_cytoscape as cyto
    cyto.load_extra_layouts()
    return cyto.Cytoscape(id=id,
                          stylesheet=cy_stylesheet,
                          style={
                              'position': 'absolute',
                              'width': '100%',
                              'height': '100%',
                          },
                          minZoom=1e-1,
                          maxZoom=4.0,
                          layout={
                              'name': 'dagre',
                          },
                          responsive=False)


top_lable_style = {
    "fontSize": "16px",
    "fontWeight": "bold",
    "color": "blue",
    "paddingTop": "5px",
    "paddingLeft": "10px",
}


def top_label(id):
    return dbc.Label("Model Path", id=id, style=top_lable_style)


def auto_load():
    return dcc.Interval(
        id="auto_load",
        n_intervals=0,
        max_intervals=0,  # <-- only run once
        interval=1
    )


def graph_layout_dropdown(id):
    return dbc.Select(id=id,
                      placeholder='DAGRE',
                      size="sm",
                      options=[{
                          'label': x.upper(),
                          'value': x
                      } for x in ('dagre', 'klay', 'cola', 'euler', 'grid',
                                  'circle', 'concentric', 'breadthfirst')])


def forward_buttom(*id):
    return dbc.Button(
        [
            "Forward",
            dbc.Badge(
                "0",
                id=id[1],
                color="danger",
                pill=True,
                className="position-absolute top-0 start-100 translate-middle",
            ),
        ],
        id=id[0],
        outline=True,
        color="primary",
        size="sm",
        n_clicks=None,
        disabled=False,
        className="position-relative",
    )


def mini_menu(*ids):
    id_m, id_layout, id_toolbox, id_figure, id_split = ids
    return dbc.DropdownMenu([
        dbc.DropdownMenuItem("Graph Layout", header=True),
        graph_layout_dropdown(id=id_layout),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem(
            "hide toolbox", id=id_toolbox, n_clicks=1, disabled=True),
        dbc.DropdownMenuItem("hide figure", id=id_figure, n_clicks=1),
        dbc.DropdownMenuItem("split Vertical", id=id_split, n_clicks=1),
    ],
        label="Display",
        color="primary",
        className="m-1",
        size="sm",
        group=True,
        id=id_m)


def layer_info_card(*ids):
    return dbc.Table(id=ids[0])


def layer_info(layer):
    def shape():
        shape = list(layer.shape)
        shape_ = "["
        for s in shape:
            shape_ = shape_ + " " + str(s)
        shape_ = shape_ + "]"
        shape_ = shape_.replace("[ ", "[")
        shape_ = shape_.replace(" ", ",")
        return shape_

    max_inout = np.maximum(len(layer.opds), len(layer.outputs))
    ins = [' ']*max_inout
    outs = [' ']*max_inout
    ins[0:len(layer.opds)] = layer.opds
    outs[0:len(layer.outputs)] = layer.outputs

    tmp_attrs = {x: layer.attrs[x] for x in layer.attrs if x !=
                 "multipliers" and x != "rshifts" and x != "multiplier" and x != "rshift" and x != "quant_mode"}
    max_att = (len(tmp_attrs) + 1)//2 * 2
    k = [x for x in tmp_attrs]
    v = [tmp_attrs[x] for x in tmp_attrs]
    if len(tmp_attrs) % 2 == 1:
        k.extend(' ')
        v.extend(' ')
    att = []
    for i in np.arange(max_att//2):
        att.append([k[i], v[i], k[i+max_att//2], v[i+max_att//2]])

    table_style = {
        'border': '1px solid',
        'border-color': '#92a8d1',
        'stripped': 'True',
        'border-collapse': 'collapse',
        'word-wrap': 'break-word',
        'white-space': 'nowrap',
        'font-size': '14px'
    }

    return html.Div(dbc.Table([
        html.Thead([
            html.Tr([html.Th("Layer Name", style=table_style), html.Th(layer.name, style=table_style), html.Th(
                "Layer Type", style=table_style), html.Th(layer.type, style=table_style)], style=table_style),
        ]),
        html.Tbody([
            html.Tr([html.Td('Inputs', style=table_style), html.Td(f'{in_}', style=table_style), html.Td('Outputs', style=table_style), html.Td(f'{out_}', style=table_style)]) for in_, out_ in zip(ins, outs)
        ]),
        html.Tbody([
            html.Tr([html.Td(f'{x[0]}', style=table_style), html.Td(f'{x[1]}', style=table_style), html.Td(f'{x[2]}', style=table_style), html.Td(f'{x[3]}', style=table_style)]) for x in list(att)
        ]),
    ], style=table_style),
        style={'overflow-y': 'scroll'})


def draggable_toolbox(*ids):
    id_m, id_toolbox = ids
    return html.Div(
        ddrage.GridLayout(id=id_m,
                          clearSavedLayout=True,
                          children=[
                              # toolbox(id_toolbox),
                          ],
                          verticalCompact=False,
                          layout=[{
                              'i': id_toolbox,
                              'x': 10,
                              'y': 5,
                              'w': 3,
                              'h': 9,
                              'isResizable': False
                          }],
                          **draggable_layout),
        style={
            'position': 'absolute',
            'display': 'none'
        },
    )


def info_tab(*ids):
    tabs_styles = {'zIndex': 99, 'display': 'inlineBlock', 'height': '3vh', 'width': '30vw',
                   "background": "#323130", 'border': 'grey', 'border-radius': '4px'}
    tab_style = {
        "background": "#92a8d1",
        'text-transform': 'uppercase',
        'color': 'white',
        'border': 'gray',
        'font-size': '11px',
        'font-weight': 600,
        'align-items': 'center',
        'justify-content': 'center',
        'border-radius': '4px',
        'padding': '4px',
    }

    tab_selected_style = {
        "background": "Blue",
        'text-transform': 'uppercase',
        'color': 'white',
        'font-size': '11px',
        'font-weight': 600,
        'align-items': 'center',
        'justify-content': 'center',
        'border-radius': '4px',
        'padding': '6px'
    }

    return html.Div([
        dcc.Tabs(
            id=ids[0],
            value='tab0',
            children=[
                dcc.Tab(label='Layer/Tensor Info', value='tab0',
                        id=ids[1],
                        children=[
                            html.Div([
                                tensor_figure(
                                    'figure-graph', 'figure-sample',
                                    'figure-sample-display',
                                    'figure-store'),
                            ], style={'width': '100%', 'height': '45vh'}),
                            html.Div(
                                dbc.Table(id='layer-info-card0',),
                                style={'position': 'absolute', 'top': '51vh',  'height': '100%', 'width': '100%'}),
                        ], style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Distribution Info', value='tab1',
                        id=ids[2],
                        children=[
                            html.Div(
                                dist_figure('dist-graph', 'dist-store'),
                                style={'height': '100vh'}),
                        ], style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Weight Info', value='tab2',
                        id=ids[3],
                        children=[
                            html.Div(
                                tensor_figure(
                                    'weight-graph', 'weight-sample',
                                    'weight-sample-display',
                                    'weight-store'),
                                    style={'height': '80vh'}),
                        ], style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Summary Info', value='tab3',
                        id=ids[4],
                        children=[
                            html.Div(
                                info_tabulator('info-tabulator'),
                                style={'height': '100vh'}),
                        ], style=tab_style, selected_style=tab_selected_style),
            ], style=tabs_styles),
    ], style={'height': '100vh'})


def draggable_figure(*ids):
    id_m, id_figure = ids
    return ddrage.GridLayout(id=id_m,
                             clearSavedLayout=True,
                             children=[
                                 dcc.Graph(id=id_figure,
                                           responsive=True,
                                           style={
                                               "width": "100%",
                                               "height": "100%",
                                           },
                                           config={'displayModeBar': False}),
                             ],
                             **draggable_layout)


def tensor_figure(*ids):
    id_figure, id_fig_sample, id_fig_sample_display, id_fig_store = ids
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Input(id=id_fig_sample, type='range', value=1000,
                              min=1000, max=20000, step=100),
                    width=10),
            dbc.Col(html.Div("1000",
                             id=id_fig_sample_display,
                             style={'fontSize': '11px',
                                    'fontWeight': 'bold',
                                    'backgroundColor': 'rgba(0, 128, 255, 0.1)',
                                    'borderRadius': '5px',
                                    'padding': '8px',
                                    },
                             ),
                    align="center")
        ],
        ),
        dcc.Store(id=id_fig_store, data=""),
        dcc.Graph(id=id_figure,
                  responsive=True,
                  style={
                      "width": "100%",
                      "height": "100%",
                  },
                  config={'displayModeBar': False})
    ], style={"width": "100%",
              "height": "100%",
              })


def dist_figure(*ids):
    id_figure, id_fig_store = ids
    return html.Div([
        dcc.Store(id=id_fig_store, data=""),
        dcc.Graph(id=id_figure,
                  responsive=True,
                  style={
                      "width": "100%",
                      "height": "60%",
                  },
                  config={'displayModeBar': False})
    ], style={"width": "100%",
              "height": "100%",
              })

def weight_figure(*ids):
    id_figure, id_fig_store = ids
    return html.Div([
        dcc.Store(id=id_fig_store, data=""),
        dcc.Graph(id=id_figure,
                  responsive=True,
                  style={
                      "width": "100%",
                      "height": "60%",
                  },
                  config={'displayModeBar': False})
    ], style={"width": "100%",
              "height": "100%",
              })

def metrics_figure(*ids):
    id_m, *_ = ids
    return dcc.Graph(id=id_m,
                     responsive=True,
                     style={
                         "width": "100%",
                         "height": "100%",
                     },
                     config={
                         'displayModeBar': False,
                     })


def split_panel(*ids, **kwargs):
    id_m, *_ = ids
    return html.Div(dsp.DashSplitPane(id=id_m, **kwargs),
                    style={
                        'position': 'relative',
                        'width': '100%',
                        'height': '100%',
    })


def update_edge_info(tensor_info):
    info = tensor_info.filter(regex="tensor|dtype|shape")
    info = info.to_dict('index')

    def format_value(meta):
        msg = "{0}\n {1} \n {2}"
        if meta['fp32net_shape'] == meta['int8net_shape']:
            shape = meta['int8net_shape']
        else:
            shape = [meta['fp32net_shape'], meta['int8net_shape']]
        dtype = (meta['fp32net_dtype'], meta['int8net_dtype'])
        return msg.format(meta['tensor'], shape, dtype)

    return {k: format_value(v) for k, v in info.items()}


def info_tabulator(*ids):
    id_tab, *_ = ids
    return html.Div([
        html.Div(
            dtb.DataTable(
                id=id_tab,
                # columns=columns,
                # data=data,
                virtualization=True,
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                fixed_rows={'headers': True, 'data': 0},
                style_cell={
                    'minWidth': 10,
                    'maxWidth': 200,
                    'width': 95,
                    'textAlign': 'left'
                },
                style_table={
                    'height': '100vh',
                    'overflowY': 'scroll',
                },
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white'
                },
                style_data_conditional=[{
                    'if': {
                        'row_index': 'odd'
                    },
                    'backgroundColor': 'rgb(240, 240, 240)',
                }],
                style_header={
                    'backgroundColor': 'rgb(210, 210, 210)',
                    'color': 'black',
                    'fontWeight': 'bold'
                }),
        ),
        html.Div(id="test-output",
                 style={'display': 'none', 'height': '1%'}),
    ], style={'height': '100vh'})


def update_tab_info(tensor_info):
    from .metrics import metrics
    metrics_str = "|".join(metrics.keys()).upper()
    info = tensor_info.filter(regex="layer|^type|tensor|" + metrics_str)

    columns = [{'id': c, 'name': c} for c in info.columns]
    records = [{'id': k, **v} for k, v in info.to_dict('index').items()]
    return columns, records


class draggable_manager():
    def __init__(self, graph):
        self.graph = graph
