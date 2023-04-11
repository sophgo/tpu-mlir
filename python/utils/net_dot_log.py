#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import graphviz as gz

# Automatically generate visual logs according to the network structure,
# which can realize the association between node related logs and the
# network structure, and facilitate the analysis and positioning of problems
class net_dot_log:
    def __init__(self, dot_log_path, parser = None, logger = None):
        self.topo_idx = 0
        self.dot_log_path = dot_log_path
        self.node_info_map = {}
        self.node_attr_map = {}
        self.first_region_name = None
        self.logger = logger
        if parser is not None:
            for op in parser.ops:
                pre_ops = parser.get_pre_op_by_op_name(op.name)
                op_type = op.type.split('.')[-1]
                if op_type == 'Conv' and int(op.attrs['group'].split(':')[0].strip()) > 1:
                    op_type = f'{op_type}_depth'
                self.append_input_edge_and_node(pre_ops, op.name, op_type)

    def append_input_edge_and_node(self, input_edges, node:str, type:str, log_str:str = None):
        self.node_info_map[node] = [self.topo_idx, type, input_edges, log_str]
        self.node_attr_map[node] = {}
        self.topo_idx += 1

    def add_node_label(self, node:str, log_str:str):
        if self.logger is not None:
            self.logger.print_dbg(log_str)
        else:
            print(f'{node} : {log_str}')
        if node not in self.node_info_map:
            return
        old_str = self.node_info_map[node][-1]
        if old_str is None:
            old_str = ''
        old_str += '\l{}'.format(log_str)
        self.node_info_map[node][-1] = old_str

    def add_new_log_region(self, region_name = ''):
        if len(self.node_info_map) == 0:
            self.first_region_name = region_name if region_name != '' else 'first_log_region'
            return
        region_name = f'>>>{region_name}:' if region_name != '' else region_name
        for node in self.node_info_map:
            self.add_node_label(node, region_name)

    def add_node_attr(self, node, attr_name:str, attr_value:str):
        if node not in self.node_attr_map:
            return
        self.node_attr_map[node][attr_name] = attr_value

    def gen_dot_graph(self, file_name = None):
        dot = gz.Digraph()
        for node in self.node_info_map:
            info = self.node_info_map[node]
            input_edges = info[2]
            log_str = info[3]
            if log_str is None:
                log_str = '\lno log'
            tmp_str = f'idx:{info[0]}  name:{node}  type:{info[1]}\n'
            if self.first_region_name is not None:
                tmp_str += f'\l>>>{self.first_region_name}:'
            tmp_str += log_str
            #print(f'gen_dot_graph for {node}, tmp_str:{tmp_str}')
            url = self.node_attr_map[node]['URL'] if 'URL' in self.node_attr_map[node] else ''
            dot.node(node, f'{tmp_str}\l', URL=url, shape='box')
            if input_edges is not None:
                for input_edge in input_edges:
                    dot.edge(input_edge, node, label=input_edge)
        if len(self.node_info_map) > 0:
            if file_name is None:
                basename = os.path.basename(self.dot_log_path)
                dirname = os.path.dirname(self.dot_log_path)
                dot.render(filename=basename, directory=dirname, view=False)
                os.system(f'dot -Tsvg {self.dot_log_path} -o ./{self.dot_log_path}.svg')
            else:
                basename = os.path.basename(file_name)
                dirname = os.path.dirname(file_name)
                dot.render(filename=basename, directory=dirname, view=False)
                os.system(f'dot -Tsvg {file_name} -o ./{file_name}.svg')
