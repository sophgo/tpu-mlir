#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 11:55
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import re
import os
from collections import namedtuple

from definition.bm1684x_defs import Arch, LayerType, DataType, GDMAOpType
from include.layer import LayerInfo
from include.summary import GlobalInfo, SubnetInfo, TensorInfo, StaticRunNode
from utils.utils import enum_cast, re_key_value
from src.common.common import *

MemBlock = namedtuple("MemBlock", "addr size alloc_time free_time type desc")


class GlobalProfileParser:
    def __init__(self):
        self.in_dir = None
        self.global_filename = "global.profile"
        self.iter_prefix = "iter"
        self.archlib = None

    def match(self, pattern):
        self.m = pattern.match(self.line)
        return self.m is not None

    def str_val(self, name):
        return self.m.group(name)

    def int_val(self, name):
        new_name = self.str_val(name).replace(",", "")
        return int(new_name)

    def enum_val(self, name, enum_type):
        return enum_cast(self.int_val(name), enum_type)

    def shape_val(self, name):
        shape_str = self.str_val(name)
        shape_str = shape_str.replace("x", ",")
        return eval(shape_str)

    def parse(self, filename):
        filename = filename + 'global.profile'
        if not os.path.exists(filename):
            # logging.fatal("'{}' does not exist".format(filename))
            return None
        self.global_filename = filename
        assert os.path.isfile(filename)
        re_subnet_info_start = re.compile(r'\[bmprofile\] subnet_id (?P<subnet_id>\d+) start')
        re_group_start = re.compile(
            r'.*start local memory usage of group (?P<group_id>\d+) subnet_id (?P<subnet_id>\d+)')
        re_layer_local_info = re_key_value("", "layer_id total_size feature_size weight_size")
        re_start_parellel = re.compile(r".*start parallel.")
        re_end_parellel = re.compile(r".*end parallel.")
        re_gdma_node = re_key_value("gdma cmd_id", "bd_id gdma_id gdma_dir gdma_func")
        re_bd_node = re_key_value("bd cmd_id", "bd_id gdma_id bd_func")

        re_tensor_info = re_key_value("",
                                      "tensor_id is_in shape dtype is_const gaddr gsize loffset nslice hslice l2addr")
        re_local_layer = re_key_value("local_layer", "layer_id layer_type layer_name")
        re_global_layer = re_key_value("global_layer", "layer_id layer_type layer_name")
        re_gdma_layer = re_key_value("gdma_layer", "gdma_type")
        re_subnet_run_start = re_key_value("start to run", "subnet_id")
        re_subnet_run_end = re_key_value("end to run", "subnet_id")
        re_arch = re_key_value("", "arch")
        re_freq = re_key_value("", "freq")
        re_mlir = re_key_value("", "is_mlir")
        re_net_name = re_key_value("", "net_name")
        re_mem_info = re_key_value("", "mtype addr size alloc free desc")

        ginfo = GlobalInfo()
        mem_info = ginfo.mem_info
        subnet_list = ginfo.subnet_list
        subnet_info = None
        current_group_id = -1
        layer_list = None
        tensor_infos = []
        layer_info = None
        layer_ins = {}  # tensor_id -> layer_id
        layer_outs = {} # tensor_id -> layer_id
        is_parallel = False
        gdma_nodes = []
        bd_nodes = []
        is_mlir = 0
        bd_id = 1
        gdma_id = 1
        with open(filename) as f:
            for self.line in f:
                if len(self.line) == 0:
                    continue
                if self.match(re_arch) and self.archlib is None:
                    ginfo.set_arch(self.enum_val("arch", Arch))
                    if not ginfo.arch:
                        ginfo.arch = Arch.bm1684x
                        # using bm1684x arch for sg2260 single core
                    self.archlib = ginfo.archlib
                    # self.archlib.show_arch_info()
                elif self.match(re_freq):
                    ginfo.freq = self.int_val("freq")
                elif self.match(re_net_name):
                    ginfo.net_name = self.str_val("net_name")
                elif self.match(re_mlir):
                    is_mlir = self.int_val("is_mlir")
                elif self.match(re_subnet_info_start):
                    subnet_info = SubnetInfo()
                    subnet_info.subnet_id = self.int_val('subnet_id')
                    subnet_info.layer_info = []
                    layer_list = subnet_info.layer_list
                    subnet_list.append(subnet_info)
                elif self.match(re_layer_local_info):
                    layer_info = LayerInfo()
                    layer_info.is_local = True
                    layer_info.layer_id = self.int_val("layer_id")
                    layer_info.total_size = self.int_val("total_size")
                    layer_info.feature_size = self.int_val("feature_size")
                    layer_info.weight_size = self.int_val("weight_size")
                    layer_info.group_id = current_group_id
                    layer_list.append(layer_info)
                elif self.match(re_subnet_run_start):
                    subnet_info = None
                    subnet_id = self.int_val("subnet_id")
                    for s in subnet_list:
                        if s.subnet_id == subnet_id:
                            subnet_info = s
                            break
                    if subnet_info is None:
                        subnet_info = SubnetInfo()
                        subnet_info.subnet_id = subnet_id
                        subnet_list.append(subnet_info)

                    layer_list = subnet_info.layer_list
                    gdma_nodes = subnet_info.gdma_nodes
                    bd_nodes = subnet_info.bd_nodes
                elif self.match(re_group_start):
                    current_group_id = self.int_val("group_id")
                elif self.match(re_global_layer):
                    layer_info = LayerInfo()
                    layer_info.layer_id = self.int_val("layer_id")
                    if is_mlir:
                        layer_info.layer_type = self.str_val("layer_type")
                    else:
                        layer_info.layer_type = self.enum_val("layer_type", LayerType)
                    layer_info.layer_name = self.str_val("layer_name")
                    layer_list.append(layer_info)
                elif self.match(re_tensor_info):
                    tensor_id = self.int_val("tensor_id")
                    tensor = None
                    for t in tensor_infos:
                        if t.tensor_id == tensor_id:
                            tensor = t
                            break
                    if tensor is None:
                        tensor = TensorInfo()
                        tensor.tensor_id = self.int_val("tensor_id")
                        tensor.shape = self.shape_val("shape")
                        tensor.dtype = self.enum_val("dtype", DataType)
                        tensor.is_const = bool(self.int_val("is_const"))
                        tensor.gaddr = self.int_val("gaddr")
                        tensor.gsize = self.int_val("gsize")
                        tensor.laddr = self.int_val("loffset")
                        tensor.nslice = self.int_val("nslice")
                        tensor.hslice = self.int_val("hslice")
                        tensor.l2addr = self.int_val("l2addr")
                        tensor_infos.append(tensor)
                    is_in = bool(self.int_val("is_in"))
                    if layer_info.layer_id == -1:
                        layer_info.set_gdma_tensor(tensor)
                    elif is_in:
                        layer_info.add_input(tensor)
                        layer_ins[tensor_id] = layer_info.layer_id
                    else:
                        layer_info.add_output(tensor)
                        layer_outs[tensor_id] = layer_info.layer_id
                elif self.match(re_start_parellel):
                    is_parallel = True
                elif self.match(re_end_parellel):
                    is_parallel = False
                elif self.match(re_bd_node):
                    node = StaticRunNode()
                    node.bd_id = bd_id
                    node.gdma_id = gdma_id
                    bd_id += 1
                    node.type = self.archlib.EngineType.BD
                    # node.bd_func = self.enum_val("bd_func", self.archlib.BDFuncType)
                    if layer_info is not None:
                        node.layer = layer_info
                        layer_info.bd_nodes.append(node)
                    bd_nodes.append(node)
                elif self.match(re_gdma_node):
                    node = StaticRunNode()
                    node.bd_id = bd_id
                    node.gdma_id = gdma_id
                    gdma_id += 1
                    node.type = self.archlib.EngineType.GDMA
                    # node.gdma_func = self.enum_val("gdma_func", self.archlib.GDMAFuncType)
                    node.gdma_dir = self.enum_val("gdma_dir", self.archlib.GDMADirection)
                    if layer_info is not None:
                        node.layer = layer_info
                        layer_info.gdma_nodes.append(node)
                    gdma_nodes.append(node)
                elif self.match(re_local_layer):
                    layer_id = self.int_val("layer_id")
                    layer_info = LayerInfo()
                    layer_info.layer_id = layer_id
                    layer_list.append(layer_info)
                    layer_info.is_local = True
                    if is_mlir:
                        layer_info.layer_type = self.str_val("layer_type")
                    else:
                        layer_info.layer_type = self.enum_val("layer_type", LayerType)
                    layer_info.layer_name = self.str_val("layer_name")
                    layer_info.group_id = current_group_id
                elif self.match(re_gdma_layer):
                    layer_info = LayerInfo()
                    layer_info.gdma_op = self.enum_val("gdma_type", GDMAOpType)
                    layer_info.group_id = current_group_id
                    layer_info.is_local = True
                    layer_list.append(layer_info)
                elif self.match(re_mem_info):
                    mem_info.append(MemBlock(
                        addr=self.int_val("addr"),
                        size=self.int_val("size"),
                        alloc_time=self.int_val("alloc"),
                        free_time=self.int_val("free"),
                        type=self.int_val("mtype"),
                        desc=self.str_val("desc")
                    ))
                ginfo.tiu_period = ginfo.archlib.BDCyclePeriod if ginfo.freq is None or ginfo.archlib.PeriodFixed else 1.0 / ginfo.freq
                ginfo.gdma_period = ginfo.archlib.GDMACyclePeriod if ginfo.freq is None or ginfo.archlib.PeriodFixed else 1.0 / ginfo.freq
        for subnet in subnet_list:
            if subnet is None:
                continue
            layer_id_map = dict((l.layer_id, l) for l in subnet.layer_list)
            for layer_info in subnet.layer_list:
                if layer_info.layer_type == 'Load':
                    layer_id = layer_ins[layer_info.out_tensors[0].tensor_id]
                    layer = layer_id_map[layer_id]
                    layer.gdma_nodes.extend(layer_info.gdma_nodes)
                elif layer_info.layer_type == 'Store':
                    layer_id = layer_outs[layer_info.in_tensors[0].tensor_id]
                    layer = layer_id_map[layer_id]
                    layer.gdma_nodes.extend(layer_info.gdma_nodes)
                elif layer_info.is_local:
                    for tensor in layer_info.in_tensors:
                        load_layer_id = layer_outs[tensor.tensor_id]
                        load_layer = layer_id_map[load_layer_id]
                        tensor.is_const = load_layer.in_tensors[0].is_const
        return ginfo
