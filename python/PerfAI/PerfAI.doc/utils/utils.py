#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# @Time    : 2023/7/18 10:55
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import logging
import os
import re
import sys
import math
from decimal import Decimal

from definition.bm1684x_defs import dma_func_name_dict, DataType


def get_simulator_total_cycle(simulator_cycle_file):
    simulator_total_cycle = 0
    if not os.path.exists(simulator_cycle_file):
        print('Error, simulatorTotalCycle.txt do not exist.')
        assert 0
    with open(simulator_cycle_file) as f:
        rows = f.readlines()
        for row in rows:
            if ':' in row:
                simulator_total_cycle = int(row.split(': ')[1])

    return simulator_total_cycle


def get_total_time(tius, gdmas, sdmas=None, cdmas=None):
    start, end = sys.maxsize, 0
    for i in range(len(tius)):
        start, end = (min(start, tius[i].start_time), max(end, tius[i].end_time)) if len(tius) > 0 else (start, end)
        start, end = (min(start, gdmas[i].start_time), max(end, gdmas[i].end_time)) if len(gdmas) > 0 else (start, end)
        if sdmas is not None:
            start, end = (min(start, sdmas[i].start_time), max(end, sdmas[i].end_time)) if len(sdmas) > 0 else (start, end)
        if cdmas is not None:
            start, end = (min(start, cdmas[i].start_time), max(end, cdmas[i].end_time)) if len(cdmas) > 0 else (start, end)
    total_time = end - start
    return total_time if total_time > 0 else 0


def get_time_by_cycle(cycle, frequency):
    frequency = int(frequency)
    cycle = int(cycle)
    # cycle to ns
    return int(cycle / frequency * 1000)


def get_active_cores(file_prefix, core_num):
    active_core_num = 0
    for core_id in range(int(core_num)):
        current_file = f"{file_prefix}_{core_id}.txt"
        if os.path.exists(current_file) and os.path.getsize(current_file):
            active_core_num += 1
    return active_core_num


def int2Hex(data_list):
    new_data_list = []
    for data in data_list:
        if not str(data).isnumeric():
            new_data_list.append('')
        else:
            new_data_list.append(str(hex(data)))
    return new_data_list


def get_dma_trace(addr):
    CHIP_ARCH = 'bm1684x'
    if CHIP_ARCH == 'bm1684x' or CHIP_ARCH == 'mars3':
        addr = int(addr)
        if addr & 0x1 == 0x1:
            return 'DDR'
        elif addr & 0x1 == 0x0:
            return 'LMEM'
        else:
            return 'None'
    elif CHIP_ARCH == "sg2260":
        addr = int(addr)
        if (addr >> 5) & 0x4 == 0x4:
            return 'DDR'
        elif (addr >> 5) & 0x7 == 0x2:
            return 'L2'
        elif (addr >> 5) & 0x7 == 0x0:
            return 'LMEM'
        else:
            return 'None'
    else:
        print("error CHIP_ARCH!")
        assert 0


def get_short_burst_length(reg_dict, data_size_dict, bl_threshold):
    src_trace = reg_dict['Direction'].split('->')[0]
    dst_trace = reg_dict['Direction'].split('->')[1]
    data_size = data_size_dict[reg_dict['src_data_format']]
    short_bl = 0
    if src_trace == 'DDR':
        if ((int(reg_dict['src_hstride']) - int(reg_dict['src_wsize'])) * data_size > bl_threshold or
                (int(reg_dict['src_cstride']) - int(reg_dict['src_hsize']) * int(
                    reg_dict['src_wsize'])) * data_size > bl_threshold):
            short_bl = 1
    elif dst_trace == 'DDR':
        if ((int(reg_dict['dst_hstride']) - int(reg_dict['dst_wsize'])) * data_size > bl_threshold or
                (int(reg_dict['dst_cstride']) - int(reg_dict['dst_hsize']) * int(
                    reg_dict['dst_wsize'])) * data_size > bl_threshold):
            short_bl = 1
    return short_bl


def get_start_addr(addr_h8, addr_l32):
    addr_h8 = str(hex(int(addr_h8)))[2:]
    addr_l32 = str(hex(int(addr_l32)))[2:]
    while len(addr_h8) < 2:
        addr_h8 = '0' + addr_h8
    while len(addr_l32) < 8:
        addr_l32 = '0' + addr_l32
    return '0x' + addr_h8 + addr_l32


def get_profile_cycle(file_path):
    total_cycle = -1
    with open(file_path, 'r') as file:
        lines = reversed(file.readlines())
        for line in lines:
            if 'total_cycle' in line:
                total_cycle = re.findall(r"\d+\.?\d*", line)[0]
                break
    return total_cycle


def remove_duplicate_path(paths):
    if len(paths) < 2:
        return paths[0]
    res = []
    idx = -1
    _pattern1 = paths[0].split('/')
    _pattern2 = paths[1].split('/')
    for i in range(len(_pattern1)):
        if _pattern1[i] != _pattern2[i]:
            idx = i
            break
    if idx >= 0:
        for path in paths:
            res.append(path.split('/')[idx])
    return res


def get_dma_func_name(reg_dict):
    dma_func_name = ''
    dma_cmd_type, dma_spec_func_name = int(reg_dict['cmd_type']), int(reg_dict['cmd_special_function'])
    if (dma_cmd_type, dma_spec_func_name) in dma_func_name_dict.keys() and \
            dma_func_name_dict[(dma_cmd_type, dma_spec_func_name)] != reg_dict['Function Type']:
        dma_func_name = dma_func_name_dict[(dma_cmd_type, dma_spec_func_name)]
    else:
        direction = ''
        if 'DDR' not in reg_dict['Direction']:
            direction = 'Mv'
        else:
            dirs = reg_dict['Direction'].split('->')
            if dirs[0] == 'DDR':
                direction = 'Ld'
            elif dirs[1] == 'DDR':
                direction = 'St'
        dma_func_name += reg_dict['Function Type'].split('_')[1] + direction
    return dma_func_name


def get_instr_cols(tiu_cols, dma_cols):
    tiu_col_set = set(tiu_cols)
    for col in dma_cols:
        if col not in tiu_col_set:
            tiu_cols.append(col)
    return tiu_cols


def get_instr_reg_list(reg_list, reg_cols):
    instr_reg_list = []
    for reg_dict in reg_list:
        total_ref_dict = dict.fromkeys(reg_cols, '')
        total_ref_dict.update(reg_dict)
        instr_reg_list.append(total_ref_dict)
    instr_reg_list.sort(
        key=lambda x: (int(x['Start Cycle']), int(x['End Cycle']), int(x['Cmd Id']), int(x['Engine Id'])))
    return instr_reg_list


def load_module(filename, name=None):
    if not filename.endswith(".py"):
        return None
    if name is None:
        name = os.path.basename(filename).split(".")[0]
    if not name.isidentifier():
        return None
    with open(filename, "r", encoding="utf8") as f:
        code = f.read()
        module = type(sys)(name)
        sys.modules[name] = module
        try:
            exec(code, module.__dict__)
            return module
        except (EnvironmentError, SyntaxError) as err:
            sys.modules.pop(name, None)
            print(err)
    return sys.modules.get(name, None)


def load_arch_lib(arch):
    # Fix me, need parse CHIP_ARCH
    archlib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        arch.name + "_defs.py")
    archlib_path = archlib_path.replace('utils', 'definition')
    return load_module(archlib_path)


def re_key_value(prefix, key_str: str):
    keys = key_str.split(" ")
    segs = [".*" + prefix + ".*"]
    for key in keys[:-1]:
        if key == "":
            continue
        seg = r"{}=(?P<{}>\S+)".format(key, key)
        segs.append(seg)
    seg = r"{}=(?P<{}>.*)".format(keys[-1], keys[-1])
    segs.append(seg)
    return re.compile(r"\s*".join(segs))


def lcs_dp(input_x, input_y):
    dp = [([0] * (len(input_y) + 1)) for i in range(len(input_x) + 1)]
    max_len = max_index = 0
    for i in range(1, len(input_x) + 1):
        for j in range(1, len(input_y) + 1):
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_index = i - max_len
            else:
                dp[i][j] = 0
    return input_x[max_index:max_index + max_len]


def get_dtype_size(dtype):
    if dtype in [DataType.FP32, DataType.INT32, DataType.UINT32]:
        return 4
    elif dtype in [DataType.UINT16, DataType.INT16, DataType.FP16, DataType.BF16]:
        return 2
    return 1


def get_dtype_size(dtype):
    if dtype in [DataType.FP32, DataType.INT32, DataType.UINT32]:
        return 4
    elif dtype in [DataType.UINT16, DataType.INT16, DataType.FP16, DataType.BF16]:
        return 2
    return 1


def enum_cast(value, enum_type, default_val=-1):
    try:
        return enum_type(value)
    except:
        logging.warning(
            "{} is not a valid {} value, using default({}) instead. ".format(value, enum_type.__name__, default_val))
        return enum_type(default_val)


def get_memory_type(s):
    dtype_map = {
        'f32': DataType.FP32,
        'f16': DataType.FP16,
        'bf16': DataType.BF16,
        'si8': DataType.INT8,
        'u8': DataType.UINT8,
        'i16': DataType.INT16,
        'u16': DataType.UINT16,
        'i32': DataType.INT32,
        'u32': DataType.UINT32
    }

    s = s[1:-1]
    shape = s.split('x')[:-1]
    data_type = s.split('x')[-1]
    shape = [int(num) for num in shape]
    import pdb
    if data_type not in dtype_map:
        pdb.set_trace()
    data_type = dtype_map[data_type]
    return shape, data_type


def get_layer_info_by_opcode(s):
    e_type = s.split('.')[0].upper()
    l_name = s.split('.')[1]
    return e_type, l_name


def calc_bandwidth(num_bytes, dur_usec):
    bandwidth = num_bytes / dur_usec * 1e6
    if bandwidth > 1e9:
        return "%.2fGB/s" % (bandwidth / 1e9)
    elif bandwidth > 1e6:
        return "%.2fMB/s" % (bandwidth / 1e6)
    elif bandwidth > 1e3:
        return "%.2fKB/s" % (bandwidth / 1e3)
    return "%.2fB/s" % bandwidth


def get_ratio_str_2f(x, y):
    x = int(x)
    y = int(y)
    return '%.2f%%' % (x / y * 100) if y != 0 else "--"


def get_ratio_str_2f_zero(x, y):
    x = int(x)
    y = int(y)
    return '%.2f%%' % (x / y * 100) if y != 0 else "0.00%"

def get_ratio_str_2f_zero_f(x, y):
    x = float(x)
    y = float(y)
    return '%.2f%%' % (x / y * 100) if y != 0 else "0.00%"

def get_ratio_float_2f(x, y):
    x = int(x)
    y = int(y)
    return round(x / y, 2) if y != 0 else 0


def get_ratio_str_3f(x, y):
    x = int(x)
    y = int(y)
    return '%.3f%%' % (x / y * 100) if y != 0 else "--"



def get_ratio_float_6f(x, y):
    x = int(x)
    y = int(y)
    return round(x / y, 6) * 1000 if y != 0 else 0


def cycle_to_us(cycles, frequency, with_unit=False):
    return str((Decimal(cycles / frequency)).quantize(Decimal("0.000"))) \
            + ('(us)' if with_unit else '')


def ops_to_gops(ops):
    return Decimal(ops / 1e9).quantize(Decimal("0.000000"))


def cycle_to_fps(cycles):
    return Decimal(1e9 / cycles).quantize(Decimal("0.00")) if cycles > 0 else 0


def datasize_to_MB(datasize):
    return str((Decimal(datasize / math.pow(2, 20))).quantize(Decimal("0.00"))) + 'MiB'
