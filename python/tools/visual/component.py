import dash_bootstrap_components as dbc
from dash import html
import dash_draggable as ddrage
from dash import dcc
import dash_split_pane as dsp
from dash import dash_table as dtb

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
        max_intervals=0, #<-- only run once
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
    id_m, id_layout, id_card, id_toolbox, id_figure, id_split = ids
    return dbc.DropdownMenu([
        dbc.DropdownMenuItem("Graph Layout", header=True),
        graph_layout_dropdown(id=id_layout),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("hide card", id=id_card, n_clicks=1),
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


def draggalbe_card(*ids):
    return ddrage.GridLayout(
        id=ids[0],
        clearSavedLayout=True,
        children=[
            dbc.Card(
                [
                    dbc.CardBody([
                        html.H4("Layer Information"),
                        html.P("Hover the cursor to a node of the graph."),
                    ],
                                 id=ids[2]),
                ],
                id=ids[1],
                style={"width": "18rem"},
            ),
        ],
        verticalCompact=False,
        style={
            'position': 'absolute',
        },
        layout=[{
            'i': 'info-card',
            'x': 0,
            'y': 5,
            'w': 3,
            'h': 13,
            'resizeHandles': ['s']
        }],
        **draggable_layout)


def layer_info(layer):
    def io(blobs):
        for blob in blobs:
            yield html.H6(["name: ", html.Code(blob)])

    def param(attrs):
        for att in attrs:
            if att == "multiplier" or att == "rshift" or att == "quant_mode":
                continue
            att_ = "{} :".format(att)
            yield html.H6([att_, html.Code(attrs[att])])

    def shape():
        shape = list(layer.shape)
        shape_ = "["
        for s in shape:
            shape_ = shape_ + " " + str(s)
        shape_ = shape_ + "]"
        shape_ = shape_.replace("[ ","[")
        shape_ = shape_.replace(" ",",")
        return html.H6(["shapes: ", html.Code(shape_)])


    return [
        html.H5("Layer Information"),
        html.Div([
            html.H6(["name: ", html.Code(layer.name)]),
            html.H6(["type: ", html.Code(layer.type)])
        ]),
        html.H5("Attributes"),
        html.Div(list(param(layer.attrs))),
        html.H5("Inputs"),
        html.Div(list(io(layer.opds))),
        html.H5("Outputs"),
        html.Div(list(io(layer.outputs))),
        #html.H5("OutShapes"),
        html.Div(shape()),

    ]


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
        dtb.DataTable(
            id=id_tab,
            # columns=columns,
            # data=data,
            virtualization=True,
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            fixed_rows={'headers': True},
            style_cell={
                'minWidth': 10,
                'maxWidth': 200,
                'width': 95,
                'textAlign': 'left'
            },
            style_table={
                'height': 800,
                'overflowY': 'auto'
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
        html.Div(id="test-output",
                 style={'display': 'none'})
    ])


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
