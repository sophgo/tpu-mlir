from . import component, graph, plot
from dash.dependencies import Input, Output, State
import dash
from . import mlirnet

callback = set()


def register(func):
    return callback.add(func)


@register
def forward_state(app):
    @app.callback(Output('forward', 'disabled'), Input('forward', 'n_clicks'),
                  Input('forward-N', 'children'),
                  Input('input-model-path', 'valid'))
    def callback(click, done, path_valid):
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        if 'input-model-path' in button_id:
            return not path_valid

        if 'forward-N' in button_id:
            return False

        if 'forward' in button_id:
            return True


@register
def forward_net(app):
    @app.callback(Output('forward-N', 'children'),
                  Output('forward-interval', 'disabled'),
                  Input('forward', 'n_clicks'),
                  Input('forward-state', 'data'),
                  prevent_initial_call=True)
    def callback(value, state):
        if value is None or app.Global.analysis_data is None:
            return dash.no_update
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        if 'forward' in button_id:
            app.Global.analysis_data.forward()
            return dash.no_update, False
        if 'forward-state' in button_id:
            if state == "Forward":
                return dash.no_update, dash.no_update
            app.Global.figure_cache = plot.figure_cache(app.Global.analysis_data)
            if state == "Idle":
                return app.Global.analysis_data.forward_count, True
            return dash.no_update, False


@register
def forward_net_state(app):
    @app.callback(Output('forward-state', 'data'),
                  Output('forward-progress', 'value'),
                  Input('forward-interval', 'n_intervals'),
                  Input('forward', 'n_clicks'),
                  State('forward-state', 'data'),
                  prevent_initial_call=True)
    def callback(triger, forward, state):
        if app.Global.analysis_data is None:
            return dash.no_update, dash.no_update
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        if 'forward' in button_id:
            return "Forward", 0
        if state == "Forward":
            progress = app.Global.analysis_data.progress
            if app.Global.analysis_data.forwarding:
                return dash.no_update, progress
            app.Global.analysis_data.run_metrics()
            return "Metrics", progress
        if state == "Metrics":
            progress = app.Global.analysis_data.metric_state * 100
            for w in app.Global.analysis_data.work:
                if w.is_alive():
                    return dash.no_update, progress
            return "Idle", progress

        return "Idle", 100


@register
def change_layout(app):
    @app.callback(Output('cytoscape-responsive-layout', 'layout'),
                  Input('select-layout', 'value'))
    def callback(value):
        if value is None:
            return dash.no_update
        return {'name': value}


@register
def show_node_flow(app):
    # show node flow
    app.clientside_callback(
        """
        function(node, data) {
           var id = node.id;
           var my_style = [{
                   'selector': `node[id = "${id}"]`,
                   'style': {
                      'background-color': '#38b000',
                      'color': '#004b23',
                      'border-width': 0.3,
                      'border-color': '#004b23'
                   }}, {
                   'selector': `edge[target = "${id}"]`,
                   'style': {
                      'line-color': '#fb8500',
                      'target-arrow-color': '#fb8500',
                   }}, {
                   'selector': `edge[source = "${id}"]`,
                   'style': {
                      'line-color': '#fb8500',
                      'target-arrow-color': '#fb8500',
                   }}];
           var de_flow_style = [{
                  'selector': `.${id}`,
                  'style': {
                     "opacity": 0.2,
                  }}];
           var neighbor_style = [{
                  'selector': `.nb${id}`,
                  'style': {
                  'background-color': '#fb8500'
                  }}];
           var new_style= data.concat(my_style, de_flow_style, neighbor_style);
           cy.style(new_style);
        // return JSON.stringify(new_style);
        }
        """,
        Output('viewport-container-cy', 'children'),
        Input('cytoscape-responsive-layout', 'mouseoverNodeData'),
        State('cytoscape-style', 'data'),
        prevent_initial_call=True,
    )


@register
def draggable_card(app):
    # show hide card
    @app.callback(Output('draggable-card', 'style'),
                  Output('show-card', 'children'),
                  Input('draggable-card', 'layout'),
                  Input('show-card', 'n_clicks'),
                  prevent_initial_call=True)
    def callback(data, clicks):
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        app.Global.zIndex += 1
        if 'draggable-card' in button_id:
            return {
                'position': 'absolute',
                'zIndex': app.Global.zIndex
            }, dash.no_update
        if 'show-card' in button_id:
            if clicks % 2 == 0:
                return {'position': 'absolute', 'display': 'none'}, "show card"
            else:
                return {
                    'position': 'absolute',
                    'zIndex': app.Global.zIndex
                }, "hide card"


@register
def draggable_toolbox(app):
    @app.callback(Output('draggable-toolbox', 'style'),
                  Output('show-toolbox', 'children'),
                  Input('draggable-toolbox', 'layout'),
                  Input('show-toolbox', 'n_clicks'),
                  prevent_initial_call=True)
    def draggable_toolbox(data, clicks):
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        app.Global.zIndex += 1
        if 'draggable-toolbox' in button_id:
            return {
                'position': 'absolute',
                'zIndex': app.Global.zIndex
            }, dash.no_update
        if 'show-toolbox' in button_id:
            if clicks % 2 == 0:
                return {
                    'position': 'absolute',
                    'display': 'none'
                }, "show toolbox"
            else:
                return {
                    'position': 'absolute',
                    'zIndex': app.Global.zIndex
                }, "hide toolbox"


@register
def load_model(app):
    # show graph
    @app.callback(Output('cytoscape-responsive-layout', 'elements'),
                  Output('input-model-path', 'valid'),
                  Output('input-model-path', 'invalid'),
                  Output('top-label', 'children'),
                  Output('top-label', 'style'),
                  Output('forward', 'n_clicks'),
                  Input('input-model-path', 'value'))
    def callback(path):
        errmsg_style = component.top_lable_style
        if path is None:
            return dash.no_update, False, False, dash.no_update, dash.no_update
        try:
            app.Global.analysis_data = mlirnet.analysis_data(path, app.Global.f32_mlir, app.Global.quant_mlir, app.Global.input)
        except Exception as e:
            errmsg_style["color"] = "#e63946"
            return dash.no_update, False, True, str(e), errmsg_style, dash.no_update
        app.Global.graph = graph.Graph(app.Global.analysis_data.quant_mlir)
        app.Global.analysis_data.build_blob_info(app.Global.graph)
        errmsg_style["color"] = 'black'
        if app.Global.manual_run:
            return app.Global.graph.cy_nodes() + app.Global.graph.cy_edges(
            ), True, False, "Model Path", errmsg_style, dash.no_update
        else:
            return app.Global.graph.cy_nodes() + app.Global.graph.cy_edges(
            ), True, False, "Model Path", errmsg_style, 1


@register
def show_figure(app):
    # show graph
    @app.callback(Output('figure-graph', 'figure'),
                  Output('figure-store', 'data'),
                  Input('cytoscape-responsive-layout', 'tapEdgeData'),
                  Input('metrics-figure', 'clickData'),
                  Input('info-tabulator', 'selected_cells'),
                  Input('figure-sample', 'value'),
                  State('figure-store', 'data'),
                  prevent_initial_call=True)
    def show_figure(edgeData, click, cell, samples, name):
        if app.Global.figure_cache is None:
            return dash.no_update, dash.no_update
        ctx = dash.callback_context
        button_id = [x['prop_id'].split('.')[0] for x in ctx.triggered]
        if 'cytoscape-responsive-layout' in button_id:
            id = tuple(int(edgeData[x]) for x in ('source', 'target'))
            name = app.Global.graph.edge(id)
        if 'metrics-figure' in button_id:
            name = click['points'][0]['customdata']
        if 'info-tabulator' in button_id:
            if len(cell) == 0:
                return dash.no_update
            id = cell[0]['row_id']
            name = app.Global.graph.edge(eval(id))
        if name == "":
            return dash.no_update, dash.no_update

        fig = app.Global.figure_cache.get_figure(plot.linear_plot,
                                                 name=name,
                                                 max_sampling=int(samples))
        return fig, name


@register
def show_node_info(app):
    # show layer information
    @app.callback(Output('info-card-body', 'children'),
                  Input('cytoscape-responsive-layout', 'mouseoverNodeData'),
                  prevent_initial_call=True)
    def show_node_info(nodeData):
        if app.Global.graph is None:
            return dash.no_update
        id = int(nodeData['id'])
        layer = app.Global.analysis_data.quant_net.layer_by_idx(id)
        return component.layer_info(layer)


@register
def split_layout(app):
    # change split panel layout to horizontal or vertical
    @app.callback(Output('split-panel', 'split'),
                  Output('split-HW', 'children'),
                  Input('split-HW', 'n_clicks'),
                  prevent_initial_call=True)
    def split_layout(clicks):
        if clicks % 2 == 0:
            return 'vertical', 'split Horizontal'
        return 'horizontal', 'split Vertical'


@register
def metrics_figure(app):
    # update metrics figure
    @app.callback(Output('metrics-figure', 'figure'),
                  Output('cytoscape-edge-info', 'data'),
                  Input('forward-state', 'data'),
                  prevent_initial_call=True)
    def metrics_figure(state):
        if state == "Idle" and app.Global.analysis_data is not None:
            if app.Global.analysis_data.tensor_info is None or len(app.Global.analysis_data.tensor_info) == 0:
                return dash.no_update, dash.no_update
            tensor_info = app.Global.analysis_data.tensor_info
            tensor_info = tensor_info[tensor_info.forward_index.eq(
                app.Global.analysis_data.forward_count)]
            return plot.metrics_plot(tensor_info), component.update_edge_info(
                tensor_info)
        return dash.no_update, dash.no_update


@register
def info_tab(app):
    # update metrics figure
    @app.callback(Output('info-tabulator', 'columns'),
                  Output('info-tabulator', 'data'),
                  Input('forward-state', 'data'),
                  prevent_initial_call=True)
    def updata_table(state):
        if state == "Idle" and app.Global.analysis_data is not None:
            if app.Global.analysis_data.tensor_info is None or len(app.Global.analysis_data.tensor_info) == 0:
                return dash.no_update, dash.no_update
            tensor_info = app.Global.analysis_data.tensor_info
            tensor_info = tensor_info[tensor_info.forward_index.eq(
                app.Global.analysis_data.forward_count)]
            return component.update_tab_info(tensor_info)
        return dash.no_update, dash.no_update


@register
def table_triger(app):
    # update center edge
    app.clientside_callback(
        """
        function(cell) {
            if (cell.length === 0)
                return;
            var e_id = cell[0].row_id
            var e = cy.$id( e_id );
            cy.$().unselect();
            cy.style().selector('node').style("opacity", 1)
                      .selector('edge').style("opacity", 1)
                      .update();
            e.select();
            // cy.$().flashClass("all-fade-flash", 600);
            e.flashClass("edge-flash", 1300);
            // cy.fit(e, 410);
            cy.center(e);
        }
        """,
        Output('test-output', 'children'),
        Input('info-tabulator', 'selected_cells'),
        prevent_initial_call=True,
    )


@register
def update_edge_info(app):
    # update center edge
    app.clientside_callback(
        """
        function(data) {
            Object.entries(data).forEach(function([key, value]) {
                // console.log(`${key} ${value}`);
                cy.$id(key).data().label = value;
            });
            cy.style().update();
        }
        """,
        Output('viewport-container-edge-info', 'children'),
        Input('cytoscape-edge-info', 'data'),
        prevent_initial_call=True,
    )


@register
def edge_focus(app):
    app.clientside_callback(
        """
        function(clickData) {
            var e_id = clickData.points[0].text;
            var e = cy.$id( e_id );
            cy.$().unselect();
            cy.style().selector('node').style("opacity", 1)
                      .selector('edge').style("opacity", 1)
                      .update();
            e.select();
            // cy.$().flashClass("all-fade-flash", 600);
            e.flashClass("edge-flash", 1300);
            // cy.fit(e, 410);
            cy.center(e);
         return JSON.stringify(clickData);
        }
        """,
        Output('viewport-container', 'children'),
        Input('metrics-figure', 'clickData'),
        prevent_initial_call=True,
    )


@register
def sync_sample(app):
    app.clientside_callback(
        """
        function(value) {
           return value;
        }
        """,
        Output('figure-sample-display', 'children'),
        Input('figure-sample', 'value'),
        prevent_initial_call=True,
    )


# get windows size
# app.clientside_callback(
#     """
#     function(href) {
#         var w = window.innerWidth;
#         var h = window.innerHeight;
#         return JSON.stringify({'height': h, 'width': w}).toString();
#     }
#     """,
#     Output('viewport-container', 'children'),
#     Input('url', 'href')
# )
