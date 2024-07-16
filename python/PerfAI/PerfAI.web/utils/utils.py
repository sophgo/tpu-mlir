# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# Utility functions
import os

data_type_dict = {
    '0': 'INT8',
    '1': 'FP16',
    '2': 'FP32',
    '3': 'INT16',
    '4': 'INT32',
    '5': 'BFP16',
    '6': 'INT4',
    '': 'None',
    '-': 'None'
}


data_size_dict = { #prec_map
    '0': 1,
    '1': 2,
    '2': 4,
    '3': 2,
    '4': 4,
    '5': 2,
    '6': 0.5,
}

ip_base_power = {
    'TPU': 0.18,
    'GDMA': 0.118,
    'SDMA': 0.05,
    'CDMA': 0.05,
    'LMEM': 0.11,
    'NOC': 8,
    'DDR': 7.2,
    'DDR MICORN': 0.88,
    'L2M': 0.11
}


def intToHex(dataList):
    newDataList = []
    for data in dataList:
        if not data.isnumeric():
            newDataList.append('')
        else:
            newDataList.append(str(hex(int(data))))
    return newDataList


def get_realtime_from_cycle(cycle, frequency):
    return round(cycle / frequency * 1000,2) #ns


def get_memory_type(s):
    s = s[1:-1]
    shape = s.split('x')[:-1]
    data_type = s.split('x')[-1]
    shape = [int(num) for num in shape]
    return shape, data_type


def load_arch_lib(arch):
    archlib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        arch.name + "_defs.py")
    return load_module(archlib_path)

def get_layer_info_by_opcode(s):
    subnet_type = s.split('.')[0].upper()
    layer_name = s.split('.')[1]
    return subnet_type, layer_name

def get_time_by_cycle(cycle, frequency):
    frequency = int(frequency)
    cycle = int(cycle)
    # cycle to ns
    return int(cycle / frequency * 1000)
