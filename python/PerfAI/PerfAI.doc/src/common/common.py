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
# @Time    : 2023/8/22 9:59
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
import ctypes as ct
from enum import Enum
import logging


class IterRecord:
    def __init__(self):
        self.dyn_data = []
        self.dyn_extra = dict()
        self.monitor_gdma = []
        self.monitor_bd = []
        self.summary = None
        self.command_info = None
        self.subnet_info = None
        self.bmlib_extra = None

    def merge(self, other):
        self.summary.merge(other.summary)
        for key, value in other.dyn_extra.items():
            if key in self.dyn_data:
                self.dyn_data[key] += value
            else:
                self.dyn_data[key] = value
        self.monitor_bd += other.monitor_bd
        self.monitor_gdma += other.monitor_gdma
        self.dyn_data = sorted(self.dyn_data, key=lambda i: i.begin_cycle)
        self.monitor_bd = sorted(self.monitor_bd, key=lambda i: i.inst_start_time)
        self.monitor_gdma = sorted(self.monitor_gdma, key=lambda i: i.inst_start_time)


class DynRecord(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("profile_id", ct.c_uint32),
        ("type", ct.c_uint32),
        ("id", ct.c_uint64),
        ("begin_cycle", ct.c_uint64),
        ("end_cycle", ct.c_uint64),
    ]


class IterSummary(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("iteration", ct.c_uint32),
        ("subnet_id", ct.c_uint32),
        ("subnet_type", ct.c_uint32),
        ("begin_usec", ct.c_uint64),
        ("end_usec", ct.c_uint64),
        ("extra_data", ct.c_uint64),
    ]


class BMLibExtraType(Enum):
    SEND_EXTRA = 0
    SYNC_EXTRA = 1
    MARK_EXTRA = 2
    COPY_EXTRA = 3
    MEM_EXTRA = 4


class SendInfo:
    def __init__(self, api, begin_usec, gdma_data, bdc_data, dyn_data, dyn_extra, info=""):
        self.api = api
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


class BMLibMemDir(Enum):
    UNKNOWN = -1
    HOST2CHIP = 0
    CHIP2HOST = 1
    CHIP2CHIP = 2


class BMLibMemOpType(Enum):
    UNKNOWN = -1
    ALLOC = 0
    FREE = 1
    INVALIDATE = 2
    FLUSH = 3


class BMLibApi(Enum):
    UNKNOWN = -1
    RESERVED = 0
    MEM_SET = 1
    MEM_CPY = 2
    MD_SCALAR = 3
    MD_SUM = 4
    MD_LINEAR = 5
    MD_CMP = 6
    MD_SFU = 7
    IMG_SUM = 8
    FLOAT2INT8 = 9
    DROPOUT_FWD = 10
    DROPOUT_BWD = 11
    ACCURACY_LAYER = 12
    LOG_FWD = 13
    LOG_BWD = 14
    SIGMOID_CROSS_ENTROPY_LOSS_FWD = 15
    SIGMOID_CROSS_ENTROPY_LOSS_BWD = 16
    CONTRASTIVE_LOSS_FWD = 17
    CONTRASTIVE_LOSS_BWD = 18
    FILTER_FWD = 19
    FILTER_BWD = 20
    SPLIT_BWD = 21
    PRELU_FWD = 22
    PRELU_BWD = 23
    SCALE_FWD = 24
    SCALE_BWD = 25
    THRESHOLD_FWD = 26
    EXP_FWD = 27
    EXP_BWD = 28
    POWER_FWD = 29
    POWER_BWD = 30
    EUCLIDEAN_LOSS_FWD = 31
    EUCLIDEAN_LOSS_BWD = 32
    SILENCE_BWD = 33
    LSTM_UNIT_FWD = 34
    LSTM_UNIT_BWD = 35
    ELTWISE_FWD = 36
    ELTWISE_BWD = 37
    BIAS_FWD = 38
    BIAS_BWD = 39
    ELU_FWD = 40
    ELU_BWD = 41
    ABSVAL_FWD = 42
    ABSVAL_BWD = 43
    BNLL_FWD = 44
    BNLL_BWD = 45
    PERMUTE = 46
    ROI_POOLING_FWD = 47
    NORMALIZE_FWD = 48
    CONV_FWD_PARALLEL = 49
    DECONV_FWD_PARALLEL = 50
    CONV_BWD_BIAS_PARALLEL = 51
    POOLING_FWD_PARALLEL = 52
    POOLING_BWD_PARALLEL = 53
    FC_BWD_PARALLEL = 55
    LRN_FWD_PARALLEL = 56
    LRN_BWD_PARALLEL = 57
    BN_FWD_INF_PARALLEL = 58
    BN_FWD_TRAIN_PARALLEL = 59
    BN_BWD_PARALLEL = 60
    SIGMOID_BWD_PARALLEL = 62
    TANH_BWD_PARALLEL = 64
    RELU_FWD_PARALLEL = 65
    RELU_BWD_PARALLEL = 66
    SOFTMAX_FWD_PARALLEL = 67
    SOFTMAX_BWD_PARALLEL = 68
    SOFTMAX_LOSS_FWD_PARALLEL = 69
    SOFTMAX_LOSS_BWD_PARALLEL = 70
    SOFTMAX_LOSS_BIDIR_PARALLEL = 71
    COEFF_UPDATE_SGD_PARALLEL = 72
    UPSAMPLE_FWD_PARALLEL = 73
    UPSAMPLE_BWD_PARALLEL = 74
    MULTIREGION_FWD_PARALLEL = 75
    MULTIREGION_BWD_PARALLEL = 76
    CONV_CORRELATION_PARALLEL = 77
    WINOGRAD_BWD_BOTTOM_DIFF_PARALLEL = 78
    REGULARIZATION_L1_PARALLEL = 79
    LSTM_FWD = 82
    CONV_FWD_FIX8B_PARALLEL = 83
    ELTWISE_FWD_FIX8B_PARALLEL = 84
    FC_FWD_FIX8B_PARALLEL = 85
    BN_FWD_FIX8B_PARALLEL = 86
    SCALE_FWD_FIX8B_PARALLEL = 87
    POOLING_FWD_FIX8B_PARALLEL = 88
    FLOAT_TO_INT_PARALLEL = 89
    INT_TO_FLOAT_PARALLEL = 90
    CONCAT = 91
    CONCAT_FIX8B = 92
    GLOBAL_MEMCPY = 93
    CROP = 94
    POOLING_FWD_TRAIN_PARALLEL = 95
    BNSCALE_FWD_FIX8B_PARALLEL = 96
    DEPTHWISE_FIX8B_FWD_PARALLEL = 97
    SCALE = 98
    AXPY = 99
    DOT = 100
    ASUM = 101
    NRM2 = 102
    ROT = 103
    ROTM = 104
    GEMM = 105
    SYMM = 106
    TRMM = 107
    TPMV = 108
    TBMV = 109
    MULTI_FULLNET = 110
    DYNAMIC_FULLNET = 111
    CV_WARP = 112
    CV_RESIZE = 113
    CV_YUV2RGB = 114
    SLICE_FWD = 115
    TILE_FWD = 116
    TRANSPOSE = 117
    TRANSPOSE_FIX8B = 118
    TILE_FIX8B_FWD = 119
    BATCH2SPACE_FIX8B = 120
    SPACE2BATCH_FIX8B = 121
    POOLING_FWD_TRAIN_INDEX_PARALLEL = 122
    DIM1_SCALAR = 123
    DIM1_SFU = 124
    CV_GEN_PROPOSAL = 126
    WORD2VEC = 127
    SUM_X2N = 128
    MD_OPS = 129
    DECONV_FIX8B_FWD_PARALLEL = 130
    CV_NMS = 131
    MEMCPY_BYTE = 136
    MEMCPY_WSTRIDE = 137
    MEMCPY_TENSOR = 138
    F32_IS = 139
    CTCLOSS = 140
    NORMALIZE_FIX8B_FWD = 141
    PERMUTE_FIX8B_FWD = 142
    SLICE_FIX8B_FWD = 143
    PAD = 144
    PAD_FIX8B = 145
    SPLIT_TF_FIX8B = 146
    UNARY = 147
    BINARY = 148
    SIMPLE_BINARY = 149
    SELECT = 150
    SIMPLE_SELECT = 151
    DEPTHWISE_FWD = 152
    DEPTHWISE_BWD_INPUT = 153
    DEPTHWISE_BWD_FILTER = 154
    PSROIPOOLING_FWD = 155
    DYNAMIC_FULLNET_EX = 156
    ARG = 157
    ARG_FIX8B = 158
    SHUFFLE_CHANNEL_FWD = 159
    SHUFFLE_CHANNEL_FIX8B_FWD = 160
    STRIDE_SLICE = 161
    FC_WEIGHT_DECOMPRESS = 162
    INTERP_FWD_PARALLEL = 163
    BIAS_FIX8B_FWD = 164
    STRIDE_SLICE_FIX8B = 165
    ADAPTIVE_POOLING_FWD = 166
    ADAPTIVE_POOLING_FIX8B_FWD = 167
    REDUCE_FIX8B = 168
    ELTWISE_BINARY_FWD_FIX8B_PARALLEL = 169
    CONST_BINARY_FWD_FIX8B_PARALLEL = 170
    BROADCAST_BINARY_FWD_FIX8B_PARALLEL = 171
    SIGMOID_FWD_PARALLEL_FIX8B = 172
    TANH_FWD_PARALLEL_FIX8B = 173
    ACTIVE_FWD = 174
    ACTIVE_FWD_FIX8B = 175
    SSD_DETECT_OUT = 176
    MEMSET_BYTE = 177
    YOLOV3_DETECT_OUT = 178
    SORT_PER_DIM = 180
    INDEX_SELECT = 181
    CONV3D_FWD_PARALLEL = 182
    MEMCPY_TENSORS = 183
    REDUCE = 200
    SEGMENT_REDUCE = 201
    LRN_FIX8B_FWD_PARALLEL = 202
    UNFOLD = 203
    CV_CONVERT_TO = 500
    CV_GEN_PROP_AND_NMS = 501
    CV_CONVERT_TO_INTERGRATED = 502
    CV_SROT_TEST = 503
    CV_FEATURE_MATCH = 504
    CV_STORAGE_CONVERT = 505
    CV_CORRECT_LAYOUT = 506
    CV_COPY_TO = 507
    CV_SORT = 508
    CV_FEATURE_MATCH_FIX8B = 509
    CV_WARP_BILINEAR = 510
    CV_SOFT_NMS = 511
    CV_SPLIT = 512
    CV_TRANSPOSE = 513
    CV_FILTER = 514
    CV_ADD_WEIGHTED = 515
    CV_BATCH_TOPK = 516
    CV_YUV2HSV = 517
    CV_DCT_COEFF = 518
    CV_DCT = 519
    FLASH_UPDATE = 600
    TOPK = 700
    CUMSUM = 701
    WHERE = 702
    MASKED_SELECT = 703
    YOLO = 800
    FC_FWD_PARALLEL = 900
    SIGMOID_FWD_PARALLEL = 901
    TANH_FWD_PARALLEL = 902
    CONV_FWD_FIX16B_PARALLEL = 983
    DEPTHWISE_FIX16B_FWD_PARALLEL = 984
    PERCHANNEL_SHIFT = 985
    SET_PROFILE_ENABLE = 986
    GET_PROFILE_DATA = 987
    START_CPU = 0x80000001
    OPEN_PROCESS = 0x80000002
    LOAD_LIBRARY = 0x80000003
    EXEC_FUNCTION = 0x80000004
    MAP_PHY_ADDR = 0x80000005
    CLOSE_PROCESS = 0x80000006
    SET_LOG = 0x80000007
    GET_LOG = 0x80000008
    SET_TIME = 0x80000009
    TPUKERNEL_1684 = 0x3f995233
    TPUKERNEL_1684X = 0xffffffe
    BMKERNEL = 0x3f995ec4
    UNLOAD_LIBRARY = 0x8000000b
    A53LITE_LOAD_LIB = 0x90000001
    A53LITE_GET_FUNC = 0x90000002
    A53LITE_LAUNCH_FUNC = 0x90000003
    A53LITE_UNLOAD_LIB = 0x90000004

    @staticmethod
    def will_run_kernel(id):
        return id not in [
            BMLibApi.RESERVED,
            BMLibApi.SET_PROFILE_ENABLE,
            BMLibApi.GET_PROFILE_DATA,
            BMLibApi.START_CPU,
            BMLibApi.OPEN_PROCESS,
            BMLibApi.LOAD_LIBRARY,
            BMLibApi.EXEC_FUNCTION,
            BMLibApi.MAP_PHY_ADDR,
            BMLibApi.CLOSE_PROCESS,
            BMLibApi.SET_LOG,
            BMLibApi.GET_LOG,
            BMLibApi.SET_TIME,
            BMLibApi.UNLOAD_LIBRARY,
            BMLibApi.A53LITE_LOAD_LIB,
            BMLibApi.A53LITE_GET_FUNC,
            BMLibApi.A53LITE_UNLOAD_LIB,
        ]


def enum_cast(value, enum_type, default_val=-1):
    try:
        return enum_type(value)
    except:
        logging.warn(
            "{} is not a valid {} value, using default({}) instead. ".format(value, enum_type.__name__, default_val))
        return enum_type(default_val)


class dictStructure(ct.Structure):
    _alias_ = {}

    def items(self):
        return zip(self.keys(), self.values())

    def __property(self):
        attr = self.__class__.__dict__
        return filter(lambda key: isinstance(attr[key], property), attr)

    def keys(self):
        from itertools import chain

        return chain((field[0] for field in self._fields_), self.__property())

    def values(self):
        return (getattr(self, key) for key in self.keys())

    def __getattr__(self, key):
        if key in self._alias_:
            return getattr(self, self._alias_[key])
        if key in self.keys():
            return getattr(self, key)
        return self.__dict__[key]

    def __setattr__(self, key, value):
        if key in self._alias_:
            super().__setattr__(self._alias_[key], value)
            return

        if key in self.keys():
            super().__setattr__(key, value)
            return

        self.__dict__[key] = value

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(dict(self.items()))

def calc_bandwidth(num_bytes, dur_usec):
    bandwidth = num_bytes/dur_usec*1e6
    if bandwidth>1e9:
        return "%.2fGB/s"%(bandwidth/1e9)
    elif bandwidth>1e6:
        return "%.2fMB/s"%(bandwidth/1e6)
    elif bandwidth>1e3:
        return "%.2fKB/s"%(bandwidth/1e3)
    return "%.2fB/s"%bandwidth
