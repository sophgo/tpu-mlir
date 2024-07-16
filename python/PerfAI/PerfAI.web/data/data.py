# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

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
