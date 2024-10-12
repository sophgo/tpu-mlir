#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import re
import logging
import struct as st
from collections import namedtuple
import glob

from bmprofile_common import *
from bmprofile_utils import *

import re

class SendInfo:
    def __init__(self, api, begin_usec, gdma_data, bdc_data, dyn_data, dyn_extra, info=""):
        self.api=api
        self.begin_usec = begin_usec
        self.gdma_data = gdma_data
        self.bdc_data = bdc_data
        self.dyn_data = dyn_data
        self.dyn_extra = dyn_extra
        self.info = info
class SyncInfo:
    def __init__(self, begin_usec, end_usec, info=""):
        self.begin_usec = begin_usec
        self.end_usec = end_usec
        self.info = info
class MarkInfo:
    def __init__(self, mark_id, begin_usec, end_usec, info=""):
        self.mark_id = mark_id
        self.begin_usec = begin_usec
        self.end_usec = end_usec
        self.info = info
class CopyInfo:
    def __init__(self, src_addr, dst_addr, dir, size, begin_usec, end_usec, info=""):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.dir = dir
        self.size = size
        self.begin_usec = begin_usec
        self.end_usec = end_usec
        self.info = info
class MemInfo:
    def __init__(self, device_addr, size, type, begin_usec, end_usec, info=""):
        self.device_addr = device_addr
        self.size = size
        self.type = type
        self.begin_usec = begin_usec
        self.end_usec = end_usec
        self.info = info


def parse_fixed_length_items(raw_data, FixedType):
    class FixedItemWrapper():
        def __str__(self):
            kv_list=[f"{k}:{getattr(self, k)}" for k in self._fields_]
            kv_list.sort()
            return ",".join(kv_list)
        def add_kv(self, k, v):
            self._fields_.append(k)
            setattr(self, k, v)
    tlen = ct.sizeof(FixedType)
    items = []
    while True:
        num_items = len(items)
        if len(raw_data) - num_items*tlen == 0:
            break
        if len(raw_data)<(num_items+1)*tlen:
            logging.warn("raw_data may be incomplete when parsing fixed length items: " + FixedType.__name__)
            break
        item_data = raw_data[num_items*tlen: (num_items+1)*tlen]
        raw_item = ct.cast(item_data, ct.POINTER(FixedType)).contents
        item = FixedItemWrapper()
        for key in raw_item._fields_:
            setattr(item, key[0], getattr(raw_item, key[0]))
        item._fields_ = list(x[0] for x in raw_item._fields_)
        items.append(item)
    return items

def parse_dyn_data(raw_data):
    return parse_fixed_length_items(raw_data, DynRecord)

def parse_summary(raw_data):
    return parse_fixed_length_items(raw_data, IterSummary)[0]

def parse_monitor_bd(raw_data, archlib):
    return parse_fixed_length_items(raw_data, archlib.BDProfileFormat)

def parse_monitor_gdma(raw_data, archlib):
    return parse_fixed_length_items(raw_data, archlib.GDMAProfileFormat)

def parse_data_blocks(filename):
    BlockItem=namedtuple("BlockItem", "type content")
    if not os.path.isfile(filename):
        return None
    blocks = []
    with open(filename, "rb") as f:
        while True:
            header_data = f.read(8)
            if len(header_data) != 8:
                break
            block_type, block_len = st.unpack("II", header_data)
            block_type = BlockType(block_type)
            block_content = f.read(block_len)
            assert len(block_content) == block_len
            blocks.append(BlockItem(block_type, block_content))
    return blocks

def parse_dyn_extra(raw_data):
    extra_data = dict()
    head_len = 12
    while True:
        if len(raw_data) == 0:
            break
        if len(raw_data)<head_len:
            logging.warn("raw_data may be incomplete when parsing extra header")
            break
        header_data = raw_data[:head_len]
        profile_id, extra_type, extra_len = st.unpack("III", header_data)
        raw_data = raw_data[head_len:]
        if len(raw_data)<extra_len:
            logging.warn("raw_data may be incomplete when parsing extra data: extra_len="+str(extra_len))
            break
        content = raw_data[:extra_len]
        raw_data = raw_data[extra_len:]
        extra_item = DynExtra(profile_id, DynExtraType(extra_type), content)
        if profile_id not in extra_data:
            extra_data[profile_id] = [extra_item]
        else:
            extra_data[profile_id].append(extra_item)
    return extra_data

def parse_bmlib_extra(raw_data):
    extra_data = []
    head_len = 12
    while True:
        if len(raw_data) == 0:
            break
        if len(raw_data)<head_len:
            logging.warn("raw_data may be incomplete when parsing extra header")
            break
        header_data = raw_data[:head_len]
        type, index, extra_len = st.unpack("III", header_data)
        if extra_len == 0:
            continue
        raw_data = raw_data[head_len:]
        if len(raw_data)<extra_len:
            logging.warn("raw_data may be incomplete when parsing extra data: extra_len="+str(extra_len))
            break
        content = str(raw_data[:extra_len], encoding="utf-8")
        raw_data = raw_data[extra_len:]
        type = BMLibExtraType(type)
        extra_data.append([type, index, content])
    return extra_data

class BMLibSummary:
    def merge(self, other):
        self.begin_usec = min(self.begin_usec, other.begin_usec)
        self.end_usec = max(self.end_usec, other.end_usec)
        self.send_info += other.send_info
        self.sync_info += other.sync_info
        self.mark_info += other.mark_info
        self.copy_info += other.copy_info
        self.mem_info += other.mem_info

        self.send_info = sorted(self.send_info, key=lambda i: i.begin_usec)
        self.sync_info = sorted(self.sync_info, key=lambda i: i.begin_usec)
        self.mark_info = sorted(self.mark_info, key=lambda i: i.begin_usec)
        self.copy_info = sorted(self.copy_info, key=lambda i: i.begin_usec)
        self.mem_info = sorted(self.mem_info, key=lambda i: i.begin_usec)

    def __init__(self, raw_data, infile, archlib):
        interval_len = 16
        self.begin_usec, self.end_usec = st.unpack(
            "QQ", raw_data[0:interval_len])
        raw_data = raw_data[interval_len:]
        self.send_info = []
        self.sync_info = []
        self.mark_info = []
        self.copy_info = []
        self.mem_info = []
        self.extra = None

        num_send = st.unpack("I", raw_data[0:4])[0]
        raw_data = raw_data[4:]
        for i in range(num_send):
            api_id = st.unpack("I", raw_data[0:4])[0]
            begin_usec = st.unpack("Q", raw_data[4:12])[0]
            raw_data = raw_data[12:]
            api = enum_cast(api_id, BMLibApi)
            if api == BMLibApi.SET_PROFILE_ENABLE or api == BMLibApi.GET_PROFILE_DATA:
                continue
            dyn_data = None
            gdma_data = None
            bdc_data = None
            dyn_extra = None
            self.send_info.append(SendInfo(api, begin_usec, gdma_data, bdc_data, dyn_data, dyn_extra))
        num_sync = st.unpack("I", raw_data[0:4])[0]
        raw_data = raw_data[4:]
        for i in range(num_sync):
            begin_usec, end_usec = st.unpack("QQ", raw_data[0:16])
            self.sync_info.append(SyncInfo(begin_usec, end_usec))
            raw_data = raw_data[16:]

        if len(raw_data) > 0:
            num_mark = st.unpack("I", raw_data[0:4])[0]
            raw_data = raw_data[4:]
            for i in range(num_mark):
                mark_id, begin_usec, end_usec = st.unpack("QQQ", raw_data[0:24])
                self.mark_info.append(MarkInfo(mark_id, begin_usec, end_usec))
                raw_data = raw_data[24:]


        if len(raw_data) > 0:
            num_copy = st.unpack("I", raw_data[0:4])[0]
            raw_data = raw_data[4:]
            for i in range(num_copy):
                self.copy_info.append(CopyInfo(*st.unpack("QQIIQQ", raw_data[0:40])))
                raw_data = raw_data[40:]

        if len(raw_data) > 0:
            num_mem = st.unpack("I", raw_data[0:4])[0]
            raw_data = raw_data[4:]
            for i in range(num_mem):
                self.mem_info.append(MemInfo(*st.unpack("QIIQQ", raw_data[0:32])))
                raw_data = raw_data[32:]

    def add_extra(self, extra):
        if extra is None:
            return
        data_dict = {
            BMLibExtraType.COPY_EXTRA: self.copy_info,
            BMLibExtraType.SYNC_EXTRA: self.sync_info,
            BMLibExtraType.SEND_EXTRA: self.send_info,
            BMLibExtraType.MARK_EXTRA: self.mark_info,
            BMLibExtraType.MEM_EXTRA: self.mem_info,
        }
        for t,i,v in extra:
            data_dict[t][i].info = v

class IterRecord():
    def __init__(self):
        self.dyn_data = []
        self.dyn_extra = dict()
        self.monitor_gdma = []
        self.monitor_sdma = []
        self.monitor_bd = []
        self.summary = None
        self.command_info = None
        self.subnet_info = None
        self.bmlib_extra = None
    def merge(self, other):
        self.summary.merge(other.summary)
        for key, value in other.dyn_extra.items():
            if key in self.dyn_data:
                self.dyn_data[key]+= value
            else:
                self.dyn_data[key] = value
        self.monitor_bd += other.monitor_bd
        self.monitor_gdma += other.monitor_gdma
        self.monitor_sdma += other.monitor_sdma
        self.dyn_data = sorted(self.dyn_data, key=lambda i: i.begin_cycle)
        self.monitor_bd= sorted(self.monitor_bd, key=lambda i: i.inst_start_time)
        self.monitor_gdma= sorted(self.monitor_gdma, key=lambda i: i.inst_start_time)
        self.monitor_sdma= sorted(self.monitor_sdma, key=lambda i: i.inst_start_time)

class NetStatParser:
    layer_prefix = "LAYER_"
    def __parse_gdma_item(self, line):
        # ST_5_N|GDMA_TENSOR|s:50216|b:3|g:5|e:52576|t:2361|dr:1|sz:96768|bw:38.17
        items = line.split("|")
        layer_id = -1
        tensor_id = -1
        info = items[0]
        if info.startswith(self.layer_prefix):
            layer_id = int(info[len(self.layer_prefix):])
        else:
            tensor_id = int(re.search(r"\d+", line).group())
        op_type = items[1]
        start_time = int(items[2].split(":")[-1])/1000.0 + self.time_offset
        bd_id = int(items[3].split(":")[-1])
        gdma_id = int(items[4].split(":")[-1])
        end_time = int(items[5].split(":")[-1])/1000.0 + self.time_offset
        cost_time = int(items[6].split(":")[-1])/1000.0
        direction = int(items[7].split(":")[-1])
        byte_size = int(items[8].split(":")[-1])
        if len(items)>9:
            bandwidth = float(items[9].split(":")[-1])
        else:
            bandwidth = float(byte_size)/(1000*cost_time)
        self.last_end_time = max(self.last_end_time, end_time)
        self.gdma_nodes.append(
            GDMASimRecord(
                layer_id=layer_id,
                tensor_id=tensor_id,
                info=info,
                op_type=op_type,
                start_time=start_time,
                bd_id=bd_id,
                gdma_id=gdma_id,
                end_time=end_time,
                cost_time=cost_time,
                direction=direction,
                byte_size=byte_size,
                bandwidth=bandwidth,
            )
        )

    def __parse_bd_item(self, line):
        # LAYER_0|CONV|s:50216|b:4|g:4|e:83974|t:33759
        items = line.split("|")
        layer_id = -1
        if items[0].startswith(self.layer_prefix):
            layer_id = int(items[0][len(self.layer_prefix):])
        if len(items)<7:
            return
        op_type = items[1]
        start_time = int(items[2].split(":")[-1])/1000.0 + self.time_offset
        bd_id = int(items[3].split(":")[-1])
        gdma_id = int(items[4].split(":")[-1])
        end_time = int(items[5].split(":")[-1])/1000.0 + self.time_offset
        cost_time = int(items[6].split(":")[-1])/1000.0
        self.last_end_time = max(self.last_end_time, end_time)
        self.bd_nodes.append(
            BDSimRecord(
                layer_id=layer_id,
                op_type=op_type,
                bd_id=bd_id,
                gdma_id=gdma_id,
                start_time=start_time,
                end_time=end_time,
                cost_time=cost_time)
            )

    def __parse_line(self, line):
        if "dr" in line:
            self.__parse_gdma_item(line)
        else:
            self.__parse_bd_item(line)

    def __update_global_data(self, global_info, update_time):
        bd_idx = 0
        gdma_idx = 0
        for subnet in global_info.subnet_list:
            start_time = self.gdma_nodes[gdma_idx].start_time
            end_time = self.gdma_nodes[gdma_idx].end_time
            for bd_node in subnet.bd_nodes:
                if bd_idx >= len(self.bd_nodes):
                    continue
                sim_node = self.bd_nodes[bd_idx]
                if bd_node.bd_id == sim_node.bd_id and bd_node.gdma_id == sim_node.gdma_id:
                    bd_node.sim_info = sim_node
                    if update_time and bd_node.layer is not None:
                        bd_node.layer.update_time(sim_node.start_time, sim_node.end_time)
                    start_time = min(start_time, sim_node.start_time)
                    end_time = max(end_time, sim_node.end_time)
                    bd_idx += 1
            for gdma_node in subnet.gdma_nodes:
                if gdma_idx >= len(self.gdma_nodes):
                    continue
                sim_node = self.gdma_nodes[gdma_idx]
                if gdma_node.bd_id == sim_node.bd_id and gdma_node.gdma_id == sim_node.gdma_id:
                    gdma_node.sim_info = sim_node
                    if update_time and gdma_node.layer is not None:
                        gdma_node.layer.update_time(sim_node.start_time, sim_node.end_time)
                    start_time = min(start_time, sim_node.start_time)
                    end_time = max(end_time, sim_node.end_time)
                    gdma_idx += 1
            if len(subnet.bd_nodes)>0 or len(subnet.gdma_nodes)>0:
                subnet.sim_info=(start_time, end_time)

    def parse(self, global_info, filename):
        # s: start_time, b: bd_id, g: gdma_id, e: end_time, t: cmd cost_time, dr: direction, sz: memsize, bw: bandwidth

        update_time = global_info.no_perf_data
        start = False
        self.gdma_nodes = []
        self.bd_nodes = []
        self.time_offset = 0
        self.last_end_time = 0
        if not os.path.exists(filename):
            return global_info
        with open(filename) as f:
            for line in f:
                line = line.strip()
                strings = re.split(r"\s+", line)
                if strings == ["ENGINE_BD", "ENGINE_GDMA"]:
                    start = True
                    self.time_offset = self.last_end_time
                    continue
                elif start and re.match(r"-+", strings[0]) and len(strings) == 1:
                    start = False
                    continue
                if start:
                    for s in strings:
                        self.__parse_line(s)
                if line.startswith("flops:"):
                    global_info.flops += int(re.search(r'\d+', line).group())
        self.__update_global_data(global_info, update_time)
        return global_info

class BMProfileParser:
    def __init__(self):
        self.global_filename = "global.profile"
        self.mlir_filename = "final.mlir"
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

    # def __parse_mlir_file(self, filename, ginfo):
    #     if not os.path.exists(filename):
    #         print(f'{filename} not found, please copy it here')
    #     import mlir.ir as ir
    #     with open(filename, encoding='utf-8') as f:
    #         content = f.read()
    #     ctx = ir.Context()
    #     ctx.allow_unregistered_dialects = True
    #     module = ir.Module.parse(content, ctx)
    #     attr = module.operation.attributes
    #     ginfo.flops = attr['module.FLOPs'].value
    #     ginfo.net_name = attr['sym_name'].value
    #     ginfo.quant_type = attr['module.mode'].value.lower()

    def __parse_global_file(self, filename):
        assert os.path.isfile(filename)
        re_subnet_info_start = re.compile(r'\[bmprofile\] subnet_id (?P<subnet_id>\d+) start')
        re_group_start = re.compile(r'.*start local memory usage of group (?P<group_id>\d+) subnet_id (?P<subnet_id>\d+)')
        re_layer_local_info = re_key_value("", "layer_id total_size feature_size weight_size")
        re_start_parellel = re.compile(r".*start parallel.")
        re_end_parellel = re.compile(r".*end parallel.")
        re_gdma_node = re_key_value("gdma cmd_id", "bd_id gdma_id gdma_dir gdma_func")
        re_bd_node = re_key_value("bd cmd_id", "bd_id gdma_id bd_func")

        re_tensor_info = re_key_value("", "tensor_id is_in shape dtype is_const gaddr gsize loffset nslice hslice l2addr")
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
        is_parallel = False
        gdma_nodes = []
        bd_nodes = []
        is_mlir = 0
        with open(filename) as f:
            for self.line in f:
                if len(self.line)==0:
                    continue
                if self.match(re_arch) and self.archlib is None:
                    ginfo.set_arch(self.enum_val("arch", Arch))
                    self.archlib = ginfo.archlib
                    self.archlib.show_arch_info()
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
                    else:
                        layer_info.add_output(tensor)
                elif self.match(re_start_parellel):
                    is_parallel = True
                elif self.match(re_end_parellel):
                    is_parallel = False
                elif self.match(re_bd_node):
                    node = StaticRunNode()
                    node.bd_id = self.int_val("bd_id")
                    node.gdma_id = self.int_val("gdma_id")
                    node.type = self.archlib.EngineType.BD
                    node.bd_func = self.enum_val("bd_func", self.archlib.BDFuncType)
                    if layer_info is not None:
                        node.layer = layer_info
                        layer_info.bd_nodes.append(node)
                    bd_nodes.append(node)
                elif self.match(re_gdma_node):
                    node = StaticRunNode()
                    node.bd_id = self.int_val("bd_id")
                    node.gdma_id = self.int_val("gdma_id")
                    node.type = self.archlib.EngineType.GDMA
                    node.gdma_func = self.enum_val("gdma_func", self.archlib.GDMAFuncType)
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
                        addr= self.int_val("addr"),
                        size = self.int_val("size"),
                        alloc_time = self.int_val("alloc"),
                        free_time = self.int_val("free"),
                        type = self.int_val("mtype"),
                        desc = self.str_val("desc")
                    ))
                ginfo.tiu_period = ginfo.archlib.BDCyclePeriod if ginfo.freq is None or ginfo.archlib.PeriodFixed else 1.0/ginfo.freq
                ginfo.gdma_period = ginfo.archlib.GDMACyclePeriod if ginfo.freq is None or ginfo.archlib.PeriodFixed else 1.0/ginfo.freq
        return ginfo

    def __parse_command_info(self, raw_data):
        if self.archlib.arch_name == "BM1690":
            header_len = 8*6+4
            gdma_base, gdma_offset, bd_base, bd_offset, sdma_base, sdma_offset, group_num = st.unpack("QQQQQQI", raw_data[0:header_len])
            # basename = "gdma %x bd %x sdma %x sdma_offset %d group_num %d"
            # print(basename % (gdma_base, bd_base, sdma_base, sdma_offset, group_num))
            CommandInfo = namedtuple("CommandInfo", "gdma_base gdma_offset bd_base bd_offset sdma_base sdma_offset")
            return CommandInfo(gdma_base, gdma_offset, bd_base, bd_offset, sdma_base, sdma_offset)
        else:
            header_len = 8*4+4
            gdma_base, gdma_offset, bd_base, bd_offset, group_num = st.unpack("QQQQI", raw_data[0:header_len])
            group = []
            for i in range(group_num):
                group.append(st.unpack("II", raw_data[header_len+i*8: header_len+(i+1)*8]))
            CommandInfo = namedtuple("CommandInfo", "gdma_base gdma_offset bd_base bd_offset group_num group")
            return CommandInfo(gdma_base, gdma_offset, bd_base, bd_offset, group_num, group)

    def __base_read_command_data(self, base, offset, engine_type, command_num, command_parser):
        basename = "cmd_%x_%d.dat"
        command_filename = os.path.join(self.in_dir, basename%(base, engine_type.value))
        if not os.path.isfile(command_filename):
            basename = "cmd_%x_0_%d.dat"
            command_filename = os.path.join(self.in_dir, basename%(base, engine_type.value))
        byte_len = command_parser.command_byte_len()
        if not os.path.isfile(command_filename):
            return []
        with open(command_filename, "rb") as f:
            f.seek(offset)
            raw_data = f.read(command_num*byte_len)
            command_list = command_parser.parse(raw_data, command_num)
        return command_list

    def __read_command_data(self, command_info):
        gdma_num = 0
        bd_num = 0
        for num_info in command_info.group:
            gdma_num += num_info[0]
            bd_num += num_info[1]
        gdma_parser = self.archlib.GDMACommandParser()
        bd_parser = self.archlib.BDCommandParser()
        gdma_command = self.__base_read_command_data(command_info.gdma_base,
                                                     command_info.gdma_offset,
                                                     self.archlib.EngineType.GDMA,
                                                     gdma_num, gdma_parser)
        print(f"parsed gdma command num={len(gdma_command)}")

        bd_command = self.__base_read_command_data(command_info.bd_base,
                                                     command_info.bd_offset,
                                                     self.archlib.EngineType.BD,
                                                     bd_num, bd_parser)
        print(f"parsed tiu command num={len(bd_command)}")
        return [gdma_command, bd_command]

    def match_node(self, commands, dyn_node, static_node, node_id_func):
        if not commands:
            return

        def next_id(raw_id, inc=1):
            return next_id_by_width(raw_id, inc, self.archlib.ID_WIDTH)

        # fix command
        if self.archlib.ID_WIDTH>16:
            delta_id = 0
            last_id = 0
            for c in commands:
                if last_id > 65000 and c.inst_id < 1000:
                    delta_id += 65536
                last_id = c.inst_id
                c.inst_id += delta_id
        for c in commands:
            c.static = None
            c.dynamic = None
            c.command = None

        dyn_idx = 0
        cmd_idx = 0
        if static_node:
            for i, d in enumerate(dyn_node):
                c = commands[cmd_idx]
                if c.inst_id == 0 and cmd_idx > 0:
                    dyn_idx = i
                    break
                if node_id_func(d) == next_id(c.inst_id):
                    c.dynamic = d
                    c.command = d.command
                    d.pmu_info = c
                    cmd_idx += 1
            for s in static_node:
                if cmd_idx >= len(commands):
                    break
                c = commands[cmd_idx]
                if node_id_func(s) == next_id(c.inst_id):
                    c.static = s
                    c.command = s.command
                    s.pmu_info = c
                    cmd_idx += 1
        for i in range(dyn_idx, len(dyn_node)):
            if cmd_idx >= len(commands):
                break
            d = dyn_node[i]        # firmware profile data
            c = commands[cmd_idx]  # perf_monitor data
            if node_id_func(d) == next_id(c.inst_id):
                c.dynamic = d
                c.command = d.command
                d.pmu_info = c
                cmd_idx += 1

    def update_relation(self, item:IterRecord, global_data):
        # relation between static set node and static command
        gdma_commands = []
        bd_commands = []
        if item.command_info is not None:
            command_data = self.__read_command_data(item.command_info)
            if item.subnet_info is not None and item.subnet_info.command_info is None:
                subnet_info = item.subnet_info
                subnet_info.command_info = item.command_info
                assert len(subnet_info.gdma_nodes) == len(command_data[0]) or len(command_data[0]) == 0
                assert len(subnet_info.bd_nodes) == len(command_data[1]) or len(command_data[1]) == 0
                for i in range(len(command_data[0])):
                    subnet_info.gdma_nodes[i].command = command_data[0][i]
                for i in range(len(command_data[1])):
                    subnet_info.bd_nodes[i].command = command_data[1][i]
            else:
                gdma_commands, bd_commands = command_data
                for c in gdma_commands:
                    c.gdma_id = c.cmd_id
                    c.bd_id = c.dep_id
                    c.command = c
                    c.layer = None
                    c.gdma_func = self.archlib.GDMAFuncType(c.cmd_type)
                    c.gdma_dir = self.archlib.GDMADirection(1)
                for c in bd_commands:
                    c.bd_id = c.cmd_id
                    c.gdma_id = c.dep_id
                    c.command = c
                    c.layer = None


        # relation between dynamic set id node and dynamic command
        dyn_gdma = []
        dyn_bd = []
        wait_dyn_gdma = []
        wait_dyn_bd = []
        wait_nodes = []
        monitor_start_time = 0
        if item.dyn_data:
            dyn_data = item.dyn_data
            assert len(dyn_data)>0
            dyn_cycle_ns = dyn_data[0].end_cycle
            dyn_base_cycle = dyn_data[0].begin_cycle
            if len(dyn_data) > 1:
                dyn_base_cycle = dyn_data[1].begin_cycle

            extra_data = item.dyn_extra
            gdma_parser = self.archlib.GDMACommandParser()
            bd_parser = self.archlib.BDCommandParser()
            for d in dyn_data[1:]:  # skip init record
                d.pmu_info = None
                d.sim_info = None
                d.begin_usec = (d.begin_cycle - dyn_base_cycle)*dyn_cycle_ns/1000
                if d.end_cycle == 0xFFFFFFFFFFFFFFFF or d.end_cycle == 0:
                    d.end_usec = d.begin_usec + 0.01
                else:
                    d.end_usec = (d.end_cycle - dyn_base_cycle)*dyn_cycle_ns/1000
                d.extra = extra_data.get(d.profile_id, [])
                d.info = []
                d.command = None
                dyn_type = self.archlib.DynRecordType(d.type & 0xFF)
                if dyn_type == self.archlib.DynRecordType.NODE_WAIT:
                    wait_nodes.append(d)
                    d.gdma_nodes = wait_dyn_gdma
                    d.bd_nodes = wait_dyn_bd
                    for w in wait_dyn_bd + wait_dyn_gdma:
                        w.wait = d
                    wait_dyn_bd = []
                    wait_dyn_gdma = []
                elif dyn_type == self.archlib.DynRecordType.NODE_SET:
                    if monitor_start_time == 0:
                        monitor_start_time = d.end_usec
                    d.engine, d.gdma_id, d.bd_id = self.archlib.get_node_set_info(d)
                    if d.engine == self.archlib.EngineType.GDMA:
                        dyn_gdma.append(d)
                        wait_dyn_gdma.append(d)
                    elif d.engine == self.archlib.EngineType.BD:
                        dyn_bd.append(d)
                        wait_dyn_bd.append(d)

                def handle_error(reason, etype, d, e):
                    print(d.profile_id, reason)
                    error_dir = "error_data"
                    if not os.path.exists(error_dir):
                        os.mkdir(error_dir)
                    with open(os.path.join(error_dir, "{}_extra_{}.bin".format(etype, d.profile_id)), "wb") as f:
                        f.write(e.content)

                for e in d.extra:
                    if e.type == DynExtraType.STRING:
                        d.info.append(str(e.content))
                    elif dyn_type == self.archlib.DynRecordType.NODE_SET:
                        if len(e.content) == 0:
                           continue
                        if d.engine == self.archlib.EngineType.GDMA:
                            try:
                                d.command = gdma_parser.parse(e.content, 1)[0]
                            except Exception as reason:
                                handle_error(reason, "gdma", d, e)
                                continue
                        elif d.engine == self.archlib.EngineType.BD:
                            try:
                                d.command = bd_parser.parse(e.content, 1)[0]
                            except Exception as reason:
                                handle_error(reason, "tiu", d, e)
                                continue

            if hasattr(item.summary, "send_info"):
                run_api = [s for s in item.summary.send_info if BMLibApi.will_run_kernel(s.api)]
                def match_layer(layer, wait_node):
                    if hasattr(layer, "gdma_nodes") and layer.gdma_nodes:
                        layer.gdma_nodes += wait_node.gdma_nodes
                    else:
                        layer.gdma_nodes = wait_node.gdma_nodes
                    if hasattr(layer, "bd_nodes") and layer.bd_nodes:
                        layer.bd_nodes += wait_node.bd_nodes
                    else:
                        layer.bd_nodes = wait_node.bd_nodes
                    if hasattr(layer, "wait_nodes") and layer.wait_nodes:
                        layer.wait_nodes += [wait_node]
                    else:
                        layer.wait_nodes = [wait_node]

                    wait_node.layer = layer
                    for n in wait_node.gdma_nodes + wait_node.bd_nodes:
                        n.layer = layer

                if run_api:
                    if len(run_api) == len(wait_nodes):
                        layer_id = 0
                        for layer, w in zip(run_api, wait_nodes):
                            layer.layer_id = layer_id
                            layer_id += 1
                            match_layer(layer, w)
                    elif len(run_api) == 1:
                        for w in wait_nodes:
                            layer = run_api[0]
                            layer.layer_id = 0
                            match_layer(layer, w)

        # relation set id node
        static_gdma = item.subnet_info.gdma_nodes if item.subnet_info else gdma_commands
        static_bd = item.subnet_info.bd_nodes if item.subnet_info else bd_commands
        self.match_node(item.monitor_gdma, dyn_gdma, static_gdma, lambda d: d.gdma_id)
        self.match_node(item.monitor_bd, dyn_bd, static_bd, lambda d: d.bd_id)

        # calibrate time
        def reset_monitor_time(monitor_data, start_cycle=0):
            monitor_start = 0
            if len(monitor_data)>0:
                monitor_start = monitor_data[0].inst_start_time
            last_start_time = 0
            last_end_time = 0
            fixed_offset = start_cycle
            for n in monitor_data:
                if n.inst_start_time < last_start_time or n.inst_end_time < last_end_time:
                    fixed_offset += 1<<32
                last_start_time = n.inst_start_time
                last_end_time = n.inst_end_time
                n.add_kv("raw_inst_start_time", n.inst_start_time)
                n.add_kv("raw_inst_end_time", n.inst_end_time)
                n.inst_start_time = int(n.inst_start_time - monitor_start + fixed_offset)
                n.inst_end_time = int(n.inst_end_time - monitor_start + fixed_offset)

        reset_monitor_time(item.monitor_gdma, monitor_start_time/global_data.gdma_period)
        reset_monitor_time(item.monitor_bd, monitor_start_time/global_data.tiu_period)

        if len(item.monitor_bd) == 0 or len(item.monitor_gdma) == 0:
            return

        # assert gdma is always first
        if len(static_gdma) > 0 and len(static_bd) > 0: # use static info
            for gdma_node in static_gdma:
                if gdma_node.pmu_info is None:
                    continue
                gdma_start = int(gdma_node.pmu_info.inst_end_time * global_data.gdma_period/global_data.tiu_period)
                bd_id = gdma_node.bd_id
                offset = 0
                for n in static_bd:
                    if n.bd_id == bd_id+1 and n.pmu_info and n.gdma_id == gdma_node.gdma_id:
                        offset = gdma_start - n.pmu_info.inst_start_time
                        break
                if offset != 0:
                    for n in item.monitor_bd:
                        n.inst_start_time += offset
                        n.inst_end_time += offset
                    break
        elif len(dyn_gdma)>0 and len(dyn_bd)>0:
            for gdma_node in dyn_gdma:
                if gdma_node.pmu_info is None:
                    continue
                gdma_start = int(gdma_node.pmu_info.inst_end_time * global_data.gdma_period/global_data.tiu_period)
                bd_id = gdma_node.bd_id
                offset = 0
                for n in dyn_bd:
                    if n.bd_id == bd_id+1 and n.pmu_info and n.gdma_id == gdma_node.gdma_id:
                        offset = max(gdma_start - n.pmu_info.inst_start_time, int(n.end_usec/global_data.tiu_period))
                        break
                if offset != 0:
                    for n in item.monitor_bd:
                        n.inst_start_time += offset
                        n.inst_end_time += offset
                    break
        else:
            print("WARNING: Cannot determine tiu start time, use 0 instead")
            return

    def parse(self, in_dir, sim_only=False):
        self.in_dir = in_dir
        if not os.path.exists(in_dir):
            logging.fatal("'{}' does not exist".format(in_dir))
            exit(-1)
        global_file_path = os.path.join(in_dir,self.global_filename)
        mlir_file_path = os.path.join(in_dir,self.mlir_filename)
        iter_data = []
        global_info = self.__parse_global_file(global_file_path)
        # self.__parse_mlir_file(mlir_file_path, global_info)

        no_perf_data = True
        iter_count = 0
        while True:
            block_filename = self.iter_prefix+str(iter_count)+".profile"
            iter_count += 1
            block_filename = os.path.join(in_dir, block_filename)
            blocks = parse_data_blocks(block_filename)
            if blocks is None:
                break
            item = IterRecord()
            for block in blocks:
                if block.type == BlockType.DYN_DATA:
                    item.dyn_data = parse_dyn_data(block.content)
                elif block.type == BlockType.DYN_EXTRA:
                    item.dyn_extra = parse_dyn_extra(block.content)
                elif block.type == BlockType.MONITOR_BD and not sim_only:
                    item.monitor_bd = parse_monitor_bd(block.content, self.archlib)
                elif block.type == BlockType.MONITOR_GDMA and not sim_only:
                    item.monitor_gdma = parse_monitor_gdma(block.content, self.archlib)
                elif block.type == BlockType.SUMMARY:
                    item.summary = parse_summary(block.content)
                    for s in global_info.subnet_list:
                        if s.subnet_id == item.summary.subnet_id:
                            item.subnet_info = s
                            break
                elif block.type == BlockType.COMMAND:
                    item.command_info = self.__parse_command_info(block.content)
            if len(item.monitor_bd)>0 or len(item.monitor_gdma)>0:
                no_perf_data = False

            assert item.summary is not None
            self.update_relation(item, global_info)
            iter_data.append(item)

        bmlib_data = []
        iteration=0
        for infile in glob.glob(in_dir + "/bmlib*.profile"):
            print("reading " + infile)
        # while True:
            # infile = os.path.join(in_dir, "bmlib_*.profile".format(iteration))
            iteration += 1
            if not os.path.exists(infile):
                break
            blocks = parse_data_blocks(infile)
            if blocks is None:
                break
            item = IterRecord()
            bmlib_extra = None
            for block in blocks:
                if block.type == BlockType.DYN_DATA:
                    item.dyn_data = parse_dyn_data(block.content)
                elif block.type == BlockType.DYN_EXTRA:
                    item.dyn_extra = parse_dyn_extra(block.content)
                elif block.type == BlockType.MONITOR_BD:
                    item.monitor_bd = parse_monitor_bd(block.content, self.archlib)
                elif block.type == BlockType.MONITOR_GDMA:
                    item.monitor_gdma = parse_monitor_gdma(block.content, self.archlib)
                elif block.type == BlockType.BMLIB:
                    item.summary = BMLibSummary(block.content, infile, self.archlib)
                elif block.type == BlockType.BMLIB_EXTRA:
                    bmlib_extra = parse_bmlib_extra(block.content)
            assert item.summary is not None
            item.summary.iteration = os.path.basename(infile).split(".")[0]
            item.summary.add_extra(bmlib_extra)
            if len(item.monitor_bd)>0 or len(item.monitor_gdma)>0:
                no_perf_data = False
            self.update_relation(item, global_info)
            bmlib_data.append(item)
        bmlib_data = sorted(bmlib_data, key = lambda v: v.summary.begin_usec)
        global_info.no_perf_data = no_perf_data
        self.__parse_simulate(global_info, in_dir)
        return global_info, iter_data, bmlib_data

    def parse_static_command(self, input_dir:str, output_dir:str, mark_str: str, arch="bm1684"):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for infile in glob.glob(input_dir + "/cmd_*.dat"):
            outfile = os.path.join(output_dir, os.path.basename(infile).replace(".dat", ".log"))
            self.__parse_static_command_file(infile, outfile, mark_str, arch)

    def __parse_static_command_file(self, infile:str, outfile=None, mark_condition:str="", arch="bm1684"):
        ginfo = GlobalInfo()
        ginfo.set_arch(Arch[arch.lower()])
        engine_type = infile.replace(".dat", "").split("_")[-1]
        command_list = []
        gdma_parser = ginfo.archlib.GDMACommandParser()
        bd_parser = ginfo.archlib.BDCommandParser()
        command_parser = gdma_parser if engine_type == "1" else bd_parser
        byte_len = command_parser.command_byte_len()
        out = sys.stdout if outfile is None else open(outfile, "w")
        with open(infile, "rb") as f:
            while True:
                raw_data = f.read(byte_len)
                if len(raw_data) != byte_len:
                    break
                command = command_parser.parse(raw_data)
                command_list.append(command)
        for i, command in enumerate(command_list):
            if mark_condition != "" and command.test(mark_condition):
                print("--- Command #{}, mark for '{}' ---".format(i, mark_condition))
                print(str(command))
                print("--- Command #{}, mark for '{}' ---".format(i, mark_condition),file = out)
            else:
                print("--- Command #{}---".format(i),file = out)
            print(str(command), file=out)
        if outfile is not None:
            out.close()
        return command_list

    def check_static_command(self, input_dir:str, output_dir:str, mark_str: str, arch="bm1684"):
        _, iter_info, _ = self.parse(input_dir)

        static_subnets = []
        for item in iter_info:
            if item.subnet_info is None or item.subnet_info.command_info is None:
                continue
            subnet = item.subnet_info
            if subnet not in static_subnets:
                static_subnets.append(subnet)

        for subnet in static_subnets:
            command_info, gdma_nodes, bd_nodes = subnet.command_info, subnet.gdma_nodes, subnet.bd_nodes
            print("Analysing subnet_id={}, gdma_num={}, bd_num={}, group_num={}".format(subnet.subnet_id, len(gdma_nodes), len(bd_nodes), command_info.group_num))
            gdma_idx = 0
            bd_idx = 0
            for group_index in range(command_info.group_num):
                gdma_num=command_info.group[group_index][0]
                bd_num=command_info.group[group_index][1]
                print("  group {}: gdma={} bd={}".format(group_index, gdma_num, bd_num))
                self.analyse_mem(gdma_nodes[gdma_idx:(gdma_idx+gdma_num)], bd_nodes[bd_idx:(bd_idx+bd_num)])
                gdma_idx = gdma_idx + gdma_num
                bd_idx = bd_idx + bd_num

    def __check_rw_mem(self, identity, read_mems, write_mems):
        rw_pair = []
        for wi, wmem in enumerate(write_mems):
            begin_waddr = wmem.addr
            end_waddr = begin_waddr + wmem.cover_size - 1
            for ri, rmem in enumerate(read_mems):
                begin_raddr = rmem.addr
                end_raddr = begin_raddr + rmem.cover_size - 1
                if begin_raddr == begin_waddr and end_raddr == end_waddr:
                    pass
                    # print("{} has inplace operation: [{},{})".format(node, begin_waddr, end_waddr))
                elif max(begin_waddr, begin_raddr) < min(end_waddr, end_raddr):
                    rw_pair.append((ri, wi))
                    print("{} partial override: write=[{:x},{:x}), read=[{:x},{:x})".format(identity, begin_waddr,end_waddr+1, begin_raddr, end_raddr+1))
        return rw_pair

    def __print_pair_mem(self, rw_pairs, read_mems, read_nodes, write_mems, write_nodes):
        for ri, wi in rw_pairs:
            read_mem= read_mems[ri]
            write_mem= write_mems[wi]
            read_node = None
            write_node = None
            for rnode in read_nodes:
                if read_mem in rnode.command.mem_records:
                    read_node = rnode
            for wnode in write_nodes:
                if write_mem in wnode.command.mem_records:
                    write_node = wnode
            assert read_node is not None and write_node is not None
            print("  --> read_node: {}, read[{:x},{:x}), {}".format(read_node, read_mem.addr, read_mem.addr+read_mem.cover_size, read_mem.desc))
            print("  --> write_node: {} write[{:x},{:x}), {}".format(write_node, write_mem.addr, write_mem.addr+write_mem.cover_size, write_mem.desc))
            print("")

    def analyse_mem(self, gdma_nodes, bd_nodes):
        gdma_num = len(gdma_nodes)
        bd_num = len(bd_nodes)
        gdma_pos=0
        bd_pos=0
        time_point=0
        time_step = 1
        mem_read_ranges=[]
        mem_write_ranges=[]
        current_gdma_id = 0
        current_bd_id = 0
        run_groups = []
        while not (gdma_pos >= gdma_num and bd_pos >= bd_num):
            gdma_run = []
            bd_run = []

            if gdma_pos<gdma_num and gdma_nodes[gdma_pos].bd_id <= current_bd_id:
                gdma_run.append(gdma_nodes[gdma_pos])
                gdma_pos = gdma_pos + 1

            if bd_pos<bd_num and bd_nodes[bd_pos].gdma_id <= current_gdma_id:
                bd_run.append(bd_nodes[bd_pos])
                bd_pos = bd_pos + 1

            if len(gdma_run)>0 and len(bd_run)>0:
                while gdma_pos<gdma_num and gdma_nodes[gdma_pos].bd_id <= current_bd_id:
                    gdma_run.append(gdma_nodes[gdma_pos])
                    gdma_pos = gdma_pos + 1

                while bd_pos<bd_num and bd_nodes[bd_pos].gdma_id <= current_gdma_id:
                    bd_run.append(bd_nodes[bd_pos])
                    bd_pos = bd_pos + 1

            if len(gdma_run):
                current_gdma_id = gdma_run[-1].gdma_id
                # print("gdma: {} {}".format(gdma_run[-1].gdma_id, gdma_run[-1].bd_id))
            if len(bd_run)>0:
                current_bd_id = bd_run[-1].bd_id
                # print("  bd: {} {}".format(bd_run[-1].gdma_id, bd_run[-1].bd_id))
            # print(len(gdma_run), len(bd_run), current_gdma_id, current_bd_id)
            run_groups.append([gdma_run, bd_run])

        for group in run_groups:
            gdma_group, bd_group = group
            for node in gdma_group + bd_group:
                read_mems = [record for record in node.command.mem_records if not record.is_out]
                write_mems = [record for record in node.command.mem_records if record.is_out]
                self.__check_rw_mem(str(node), read_mems, write_mems)

            if len(gdma_group) > 0 and len(bd_group) > 0:
                gdma_read_mems = []
                bd_write_mems = []
                gdma_write_mems = []
                bd_read_mems = []
                for node in gdma_group:
                    gdma_read_mems = gdma_read_mems + [record for record in node.command.mem_records if not record.is_out]
                    gdma_write_mems = gdma_write_mems + [record for record in node.command.mem_records if record.is_out]
                for node in bd_group:
                    bd_read_mems = bd_read_mems + [record for record in node.command.mem_records if not record.is_out]
                    bd_write_mems = bd_write_mems + [record for record in node.command.mem_records if record.is_out]

                rw_pair = self.__check_rw_mem("", gdma_read_mems, bd_write_mems)
                self.__print_pair_mem(rw_pair, gdma_read_mems, gdma_group, bd_write_mems, bd_group)

                rw_pair = self.__check_rw_mem("", bd_read_mems, gdma_write_mems)
                self.__print_pair_mem(rw_pair, bd_read_mems, bd_group, gdma_write_mems, gdma_group)

    def __parse_simulate(self, global_info, in_dir, update_time=False):
        stat_parser = NetStatParser()
        return stat_parser.parse(global_info, os.path.join(in_dir, "net_stat.sim"))


if __name__ == "__main__":
    parser = BMProfileParser()
    parser.parse("/data/work/nntoolchain/net_compiler/bmnetp/bmprofile_data-1")
