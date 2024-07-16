# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import re
import os
from collections import namedtuple
import json
import  logging

from utils.utils import *
from src.common import *

class GlobalProfileParser:
    def __init__(self):
        self.in_dir = None
        self.json_filename = "tensor_location.json"
        self.mlir_filename = "final.mlir"
        self.subnet_list = []
        self.layer_list = []
        self.file_line_dict = {}


    def parse(self, filename):
        json_file = filename + 'tensor_location.json'
        mlir_file = filename + 'final.mlir'
        if not os.path.exists(json_file):
            logging.fatal("'{}' does not exist".format(json_file))
            return None
        elif not os.path.exists(mlir_file):
            logging.fatal("'{}' does not exist".format(mlir_file))
            return None
        self.json_filename = json_file
        self.mlir_filename = mlir_file
        assert os.path.isfile(json_file) and os.path.isfile(mlir_file)

        with open(self.mlir_filename, 'r') as file:
            lines = file.readlines()
        group_indices = []
        yield_indices = []
        group_args = []
        result_dict = {}
        for i, line in enumerate(lines):
            if 'Group' in line and 'Parallel' not in line:
                group_indices.append(i)
                group_arg = re.search(r'%(\w+)', line)
                if group_arg:
                    group_args.append(group_arg.group(1))
            if 'Yield' in line:
                yield_indices.append(i)
        for group_index in group_indices:
            first_yield_found = False
            for yield_index in yield_indices:
                if yield_index > group_index:
                    first_yield_found = True
                    result_dict[group_args[group_indices.index(group_index)]] = list(range(group_index + 2, yield_index + 1))
                    break
            if not first_yield_found:
                result_dict[group_args[group_indices.index(group_index)]] = []
        self.file_line_dict = result_dict

        ginfo = GlobalInfo()
        subnet_list = ginfo.subnet_list
        subnet_info = None
        subnet_dict = {}
        layer_list = []
        layer_ins = {}  # tensor_id -> layer_id
        layer_outs = {} # tensor_id -> layer_id
        json_layer_list = []
        with open(self.json_filename, encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                json_layer = jsonObj()
                json_layer.file_line = d['file-line']
                json_layer.subnet_id = d['subnet_id']
                json_layer.opcode = d['opcode']
                json_layer.core_id = d['core_id']
                json_layer.bd_ids = [d['tiu_dma_id(before)'][0] + 1, d['tiu_dma_id(after)'][0] + 1]
                json_layer.dma_ids = [d['tiu_dma_id(before)'][1] + 1, d['tiu_dma_id(after)'][1] + 1]
                json_layer.operands = d['operands']
                json_layer.results = d['results']
                json_layer_list.append(json_layer)
        tensor_id = 1
        for index, j_layer in enumerate(json_layer_list):
            subnet_id = j_layer.subnet_id
            if subnet_id not in subnet_dict:
                subnet_dict[subnet_id] = SubnetInfo()
                subnet_dict[subnet_id].subnet_id = subnet_id #TOCHECK
                subnet_dict[subnet_id].layer_list = []
            layer_info = self.get_layer_info(index, j_layer,tensor_id,layer_ins,layer_outs)
            subnet_dict[subnet_id].layer_list.append(layer_info)
        for subnet_info in subnet_dict.values():
            ginfo.subnet_list.append(subnet_info)

        return ginfo, self.file_line_dict

    def get_layer_info(self, index, j_layer,tensor_id,layer_ins,layer_outs):
        layer_info = LayerInfo()
        layer_info.layer_id = index + 1
        layer_info.core_id = j_layer.core_id
        layer_info.file_line = j_layer.file_line

        for idx, in_tensor in enumerate(j_layer.operands):
            if len(in_tensor.keys()) <= 0:
                continue
            tensor = TensorInfo()
            tensor.tensor_id = tensor_id
            tensor.shape, tensor.dtype = get_memory_type(in_tensor['memory_type'])
            tensor.address = in_tensor['address']
            layer_ins[tensor_id] = layer_info.layer_id
            layer_info.in_tensors.append(tensor)
            tensor_id += 1

        for out_tensor in j_layer.results:
            if len(out_tensor.keys()) <= 0:
                continue
            tensor = TensorInfo()
            tensor.tensor_id = tensor_id
            tensor.shape, tensor.dtype = get_memory_type(out_tensor['memory_type'])
            tensor.address = out_tensor['address']
            layer_outs[tensor_id] = layer_info.layer_id
            layer_info.out_tensors.append(tensor)
            tensor_id += 1

        for b_id in range(j_layer.bd_ids[0], j_layer.bd_ids[1]): #before, after
            bd_node = StaticRunNode()
            bd_node.bd_id = b_id
            bd_node.core_id = j_layer.core_id
            layer_info.bd_nodes.append(bd_node)

        for g_id in range(j_layer.dma_ids[0], j_layer.dma_ids[1]): #before, after
            dma_node = StaticRunNode()
            dma_node.gdma_id = g_id
            dma_node.core_id = j_layer.core_id
            layer_info.gdma_nodes.append(dma_node)
        layer_info.engine_type, layer_info.layer_type = get_layer_info_by_opcode(j_layer.opcode)
        # layer_list.append(layer_info)
        return layer_info

def get_engine_layer(g_info):
    tiu_layer_map = dict()
    gdma_layer_map = dict()
    # if g_info is not None:
    if isinstance(g_info, GlobalInfo):
        for subnet_info in g_info.subnet_list:
            subnet_id = subnet_info.subnet_id
            for layer_info in subnet_info.layer_list:
                for tiu_node in layer_info.bd_nodes:
                    k = tiu_node.bd_id
                    c = tiu_node.core_id
                    if (k,c) in tiu_layer_map.keys():
                        print('ERROR! Tiu id is not unique.')
                        assert 0
                    else:
                        tiu_layer_map[(k,c)] = [layer_info.layer_id, layer_info.layer_type, subnet_id, layer_info.engine_type, layer_info.file_line]
                for gdma_node in layer_info.gdma_nodes:
                    k = gdma_node.gdma_id
                    c = gdma_node.core_id
                    if (k,c) in gdma_layer_map.keys():
                        print('ERROR! Gdma id is not unique.')
                        assert 0
                    else:
                        gdma_layer_map[(k,c)] = [layer_info.layer_id, layer_info.layer_type, subnet_id, layer_info.engine_type, layer_info.file_line]
    return tiu_layer_map, gdma_layer_map
