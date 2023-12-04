import networkx as nx
import collections
import functools

import sys
sys.path.append('../..')
from utils.mlir_parser import MlirParser


class Graph():
    def __init__(self, mlir_file):
        self.parser = MlirParser(mlir_file)
        self.ops = self.parser.ops
        self.graph = nx.DiGraph()

        bottom_of_layer = collections.defaultdict(list)
        for idx, op in enumerate(self.ops):
            self.graph.add_node(idx)
            for opd in op.opds:
                bottom_of_layer[opd].append(idx)

        self.fake_node = [len(self.graph.nodes)]
        self.edge_name = dict()
        for idx, op in enumerate(self.ops):
            for opd in op.outputs:
                in_ = False
                for b_ in bottom_of_layer:
                    if opd in b_:
                        in_ = True
                        break
                ids = bottom_of_layer[op.name]
                if not in_ or len(ids) == 0:
                    fake_id = self.fake_node[-1]
                    self.fake_node.append(fake_id)
                    self.graph.add_node(fake_id)
                    self.edge_name[(idx, fake_id)] = opd
                    self.graph.add_edge(idx, fake_id)
                    self.fake_node.append(fake_id + 1)
                    continue
                for x in ids:
                    self.edge_name[(idx, x)] = opd
                    self.graph.add_edge(idx, x)
        self.edge_id = {v: k for k, v in self.edge_name.items()}
        self.__build_flow()

    def node(self, id):
        return self.ops[id]

    def edge(self, id):
        return self.edge_name[id]

    def edge_id_by_name(self, name):
        return self.edge_id[name]

    def cy_nodes(self):
        nodes = [{
            'data': {
                'id': str(id),
                'label': op.type,
                'width': len(op.type) * 8
            },
            'classes':
            ' '.join(map(str, self.de_flow_node(id))) + " " +
            ' '.join(map(lambda x: 'nb{}'.format(x), self.neighbors(id)))
        } for id, op in enumerate(self.ops)]

        f_node = [{
            'data': {
                'id': str(x),
                'label': "Fake Node",
                'width': 10
            },
            'classes': "fake-node"
        } for x in self.fake_node]

        for d in self.scc:
            if len(d) > 1:
                id = str(len(nodes))
                scc_name = '#SCC' + id
                nodes.append({'data': {'id': scc_name, 'lable': scc_name}})
                for s in d:
                    nodes[s]['data']['parent'] = scc_name
        return nodes + f_node

    def cy_edges(self):
        def de_flow_edge(s, t):
            return self.de_flow_node(s) | self.de_flow_node(t)

        return [{
            'data': {
                'id': str((s, t)),
                'source': str(s),
                'target': str(t),
                'label': self.edge((s, t))
            },
            'classes': ' '.join(map(str, de_flow_edge(s, t)))
        } for s, t in self.graph.edges]

    def save_wailea_g(self, out='out.txt'):
        with open(out, 'w') as f:
            f.write('NODES\n')
            for n in self.graph.nodes:
                f.write('{}\n'.format(n))
                f.write('\n')
                f.write('EDGES\n')
            for e in self.graph.edges:
                f.write('{} {} 1 1\n'.format(*e))

    def __build_flow(self):
        tps_g = list(nx.strongly_connected_components(self.graph))

        def base_add(tps, associate_f):
            col = collections.defaultdict(set)
            for d in tps:
                for s in d:
                    associate_node = set(associate_f(s))
                    col[s].update(associate_node)
                    for i in associate_node:
                        col[s].update(col[i])
                if len(d) > 1:  # SCC
                    scc = set()
                    for s in d:
                        scc.update(col[s])
                        scc.update(d)
                    for s in d:
                        col[s] = scc
            return col

        descendants = base_add(tps_g, self.graph.successors)
        ancestors = base_add(reversed(tps_g), self.graph.predecessors)
        flow_node = {
            n: descendants[n] | ancestors[n] | {n}
            for n in self.graph.nodes
        }
        self.descendants = lambda x: descendants[x]
        self.ancestors = lambda x: ancestors[x]
        self.flow_node = lambda x: flow_node[x]
        self.scc = tps_g

    def __ancestors(self, source):
        return nx.algorithms.dag.ancestors(self.graph, source)

    def __descendants(self, source):
        return nx.algorithms.dag.descendants(self.graph, source)

    def __flow(self, me):
        return self.__ancestors(me) | self.__descendants(me) | {me}

    @functools.lru_cache()
    def de_flow_node(self, me):
        return set(self.graph.nodes) - self.flow_node(me)

    def neighbors(self, me):
        from itertools import chain
        return chain(self.graph.successors(me), self.graph.predecessors(me))
