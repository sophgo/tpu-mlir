# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from enum import Enum
import ctypes as ct


class Arch(Enum):
    UNKNOWN = -1
    BM1684 = 1
    BM1684X = 3


class BlockType(Enum):
    UNKNOWN = -1
    SUMMARY = 1
    COMPILER_LOG = 2
    MONITOR_TIU = 3
    MONITOR_DMA = 4
    MCU_DATA = 5
    MCU_EXTRA = 6
    FIRMWARE_LOG = 7
    COMMAND = 8
    BMLIB = 9
    BMLIB_EXTRA = 10


class MCUExtraType(Enum):
    STRING = 0
    BINARY = 1
    CUSTOM = 100


class BMLibExtraType(Enum):
    SEND_EXTRA = 0
    SYNC_EXTRA = 1
    MARK_EXTRA = 2
    COPY_EXTRA = 3
    MEM_EXTRA = 4


class SubnetType(Enum):
    UNKNOWN = -1
    TPU = 0
    CPU = 1
    MERGE = 2
    SWITCH = 3


# used in static profile info
class DataType(Enum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    BF16 = 8
    UNKNOWN = -1


def get_dtype_size(dtype):
    if dtype in [DataType.FP32, DataType.INT32, DataType.UINT32]:
        return 4
    elif dtype in [DataType.UINT16, DataType.INT16, DataType.FP16, DataType.BF16]:
        return 2
    return 1


class LayerType(Enum):
    UNKNOWN = -1
    CONV = 0
    POOL = 1
    LRN = 2
    FC = 3
    SPLIT = 4
    DATA = 5
    RELU = 6
    CONCAT = 7
    BATCHNORM = 8
    SCALE = 9
    ELTWISE = 10
    PRELU = 11
    LSTM = 12
    PERMUTE = 13
    REORG = 14
    REDUCTION = 15
    SLICE = 16
    RESHAPE = 17
    LSTM_UNIT = 18
    NORMALIZE = 19
    PRIORBOX = 20
    FLATTEN = 21
    SOFTMAX = 22
    DROPOUT = 23
    ACTIVE = 24
    UPSAMPLE = 25
    MULTIREGION = 26
    DECONV = 27
    CROP = 28
    ROIPOOLING = 29
    MULSHIFT = 30
    PAD = 31
    ARG = 32
    POOLTF = 33
    STRIDESLICE = 34
    INTERP = 35
    COMPARE = 36
    UPSAMPLEMASK = 37
    TOPK = 38
    SPLIT_TF = 39
    RPN = 40
    SHUFFLECHANNEL = 41
    SELECT = 42
    TILE = 43
    ELTWISE_BINARY = 44
    CONST_BINARY = 45
    BROADCAST_BINARY = 46
    REDUCE = 47
    BIASADD = 48
    PSROIPOOLING = 49
    REVERSE = 50
    ADAPTIVEPOOLING = 51
    OUTPUT = 52
    SHAPE_REF = 53
    SHAPE_CONST = 54
    SHAPE_OP = 55
    SHAPE_SLICE = 56
    SHAPE_PACK = 57
    SHAPE_ASSIGN = 58
    SHAPE_REORDER = 59
    EXPAND_DIM = 60
    SQUEEZE_DIM = 61
    REF_PAD = 62
    REF_CROP = 63
    TRANSPOSE = 64
    REDUCE_FULL = 65
    COEFF_LAYER = 66
    BATCH2SPACE = 67
    SPACE2BATCH = 68
    EXPAND = 69
    EMBEDDING = 70
    CUMSUM = 71
    SLICELIKE = 72
    STRIDECALC = 73
    SHAPE_ADDN = 74
    CHSHIFT = 75
    CPU = 76
    CONSTANT_FILL = 77
    SIMPLE_CROP = 78
    DTYPE_CONVERT = 79
    BATCH_MATMUL = 80
    INTERLEAVE = 81
    SHAPE_RANGE = 82
    SHAPE_TILE = 83
    SHAPE_REVERSE = 84
    SHAPE_EXPAND_NDIMS = 85
    SHAPE_CAST = 86
    SHAPE_RESHAPE = 87
    SHAPE_REDUCE = 88
    TA_SIZE = 89
    TA_SCATTER = 90
    TA_GATHER = 91
    TA_READ = 92
    TA_WRITE = 93
    SWITCH = 94
    MERGE = 95
    IDENTITY = 96
    TENSOR_ARRAY = 97
    HOST2DEVICE = 98
    DEVICE2HOST = 99
    YOLO = 100
    SHAPE_UNARY = 101
    TA_SPLIT = 102
    TA_CONCAT = 103
    SSD_DETECT_OUT = 104
    SHAPE_SPLIT = 105
    SHAPE_SQUEEZE = 106
    BITWISE = 107
    RANK = 108
    ARITH_SHIFT = 109
    WHERE = 110
    YOLOV3_DETECT_OUT = 111
    MASKED_SELECT = 112
    SORT_PER_DIM = 113
    INDEX_SELECT = 114
    LUT = 115
    NMS = 116
    SHAPE_SIZESLICE = 117
    COEFF2NEURON = 118
    CONV3D = 119
    SHAPE_SELECT = 120
    DEPTH2SPACE = 121
    WHERE_SQUEEZE_GATHER = 122
    UNFOLD = 123
    BROADCAST_LIKE = 124
    MATRIX_BAND_PART = 125
    ELTWISE_BINARY_EX = 126
    BINARY_SHIFT = 127
    POOL3D = 128
    CONV3D_ADD = 129
    BMNET_GRU = 130
    PYTORCH_LSTM = 131
    ENTER = 132
    MULTI_MASKED_SELECT = 133
    TPU = 134
    SEQUENCE_GEN = 135
    DECONV3D = 136
    DEFORM_CONV = 137
    QUANT_DIV = 138
    INT32_REQUANT = 139
    FP32_REQUANT = 140
    DEQUANT_FP32 = 141
    DEQUANT_INT = 142
    ROUND = 143
    SHIFT = 144
    GATHER_ND_TF = 145
    ROIALIGN = 146
    GROUP_NORM = 147
    TEMPORAL_SHIFT = 148
    MASKED_FILL = 149
    TRIANGULARIZE = 150
    IM2SEQUENCE = 151
    GRID_SAMPLE = 152
    AXES_SLICE = 153
    PAD3D = 154
    EMBEDDING_BAG = 155
    INDEX_PUT = 156
    GLU = 157
    DEFORM_GATHER = 158
    SCATTER_ND = 159
    LAYER_NORM = 160
    IM2COL = 161
    COL2IM = 162
    NORMAL_SPARSE_CONV3D = 163
    SUBM_SPARSE_CONV3D = 164


# used in dynamic profile info
class FWDataType(Enum):
    UNKNOWN = -1
    FP32 = 0
    WORD = 1
    BYTE = 2
    INT32 = 3


class FWGDMAType(Enum):
    DEFAULT = -1
    LD_INPUT_NEURON = 0
    ST_OUTPUT_NEURON = 1
    LD_ITM_NEURON = 2
    ST_ITM_NEURON = 3
    LD_COEFF = 4
    LD_COEFF_NERUON = 5
    LD_COEFF_WINOGRAD = 6
    MV_ITM_NEURON = 7
    MV_OUTPUT_NEURON = 8
    MV_ITM_EXTEND_NEURON = 9
    ST_ITM_EXTEND_NEURON = 10
    LD_G2L2 = 11
    ST_OUTPUT_EXTEND_NEURON = 12
    LD_ITM_EXTEND_NEURON = 13


class FWLayerType(Enum):
    UNKNOWN = -1
    CONV = 0
    POOL = 1
    LRN = 2
    FC = 3
    SPLIT = 4
    DATA = 5
    RELU = 6
    CONCAT = 7
    BATCHNORM = 8
    SCALE = 9
    ELTWISE = 10
    PRELU = 11
    LSTM = 12
    PERMUTE = 13
    NORMALIZE = 14
    REORG = 15
    PRIORBOX = 16
    FLATTEN = 17
    RESHAPE = 18
    SOFTMAX = 19
    INTERP = 20
    RPN = 21
    DECONV = 22
    CROP = 23
    MULSHIFT = 24
    POOLTF = 25
    PAD = 26
    ARG = 27
    TILE = 28
    SELECT = 29
    REDUCE = 30
    BROADCAST_BINARY = 31
    CONST_BINARY = 32
    ELTWISE_BINARY = 33
    BIASADD = 34
    ACTIVE = 35
    TENSOR_ARITHMETIC = 36
    SPLIT_TF = 37
    SHAPE_REF = 38
    SHAPE_CONST = 39
    SHAPE_OP = 40
    SHAPE_SLICE = 41
    SHAPE_PACK = 42
    SHAPE_ASSIGN = 43
    SHAPE_REORDER = 44
    EXPAND_DIM = 45
    SQUEEZE_DIM = 46
    REF_PAD = 47
    REF_CROP = 48
    TRANSPOSE = 49
    REDUCE_FULL = 50
    STRIDESLICE = 51
    SPACE2BATCH = 52
    BATCH2SPACE = 53
    OUTPUT = 54
    EXPAND = 55
    EMBEDDING = 56
    TOPK = 57
    CUMSUM = 58
    SHAPE_ADDN = 59
    ROIPOOLING = 60
    CONSTANT_FILL = 61
    SIMPLE_CROP = 62
    SLICELIKE = 63
    ADAPTIVEPOOLING = 64
    BATCH_MATMUL = 65
    UPSAMPLE = 66
    SHAPE_RANGE = 67
    SHAPE_TILE = 68
    SHAPE_REVERSE = 69
    SHAPE_EXPAND_NDIMS = 70
    SHAPE_CAST = 71
    SHAPE_RESHAPE = 72
    SHAPE_REDUCE = 73
    DTYPE_CONVERT = 74
    YOLO = 75
    SSD_DETECT_OUT = 76
    HOST2DEVICE = 77
    DEVICE2HOST = 78
    TENSOR_ARRAY = 79
    TA_WRITE = 80
    TA_READ = 81
    TA_SIZE = 82
    TA_SCATTER = 83
    TA_GATHER = 84
    TA_SPLIT = 85
    TA_CONCAT = 86
    PSROIPOOLING = 87
    SHAPE_UNARY = 88
    SHAPE_SPLIT = 89
    SHAPE_SQUEEZE = 90
    RANK = 91
    WHERE = 92
    YOLOV3_DETECT_OUT = 93
    MASKED_SELECT = 94
    SORT_PER_DIM = 95
    INDEX_SELECT = 96
    NMS = 97
    SLICE = 98
    SHAPE_SIZESLICE = 99
    COEFF2NEURON = 100
    SHAPE_SELECT = 101
    DEPTH2SPACE = 102
    WHERE_SQUEEZE_GATHER = 103
    REVERSE = 104
    BROADCAST_LIKE = 105
    LUT = 106
    MATRIX_BAND_PART = 107
    ARITH_SHIFT = 108
    CONV3D = 109
    POOL3D = 110
    STRIDECALC = 111
    INTERLEAVE = 112
    BITWISE = 113
    BINARY_SHIFT = 114
    GRU = 115
    PYTORCH_LSTM = 116
    MULTI_MASKED_SELECT = 117
    TPU = 118
    SEQUENCE_GEN = 119
    UPSAMPLEMASK = 120


class CPULayerType(Enum):
    UNKNOWN = -1
    SSD_DETECTION_OUTPUT = 0
    ANAKIN_DETECT_OUTPUT = 1
    RPN = 2
    USER_DEFINED = 3
    ROI_POOLING = 4
    ROIALIGN = 5
    BOXNMS = 6
    YOLO = 7
    CROP_AND_RESIZE = 8
    GATHER = 9
    NON_MAX_SUPPRESSION = 10
    ARGSORT = 11
    GATHERND = 12
    YOLOV3_DETECTION_OUTPUT = 13
    WHERE = 14
    ADAPTIVE_AVERAGE_POOL = 15
    ADAPTIVE_MAX_POOL = 16
    TOPK = 17
    RESIZE_INTERPOLATION = 18
    GATHERND_TF = 19
    SORT_PER_DIM = 20
    WHERE_SQUEEZE_GATHER = 21
    MASKED_SELECT = 22
    UNARY = 23
    EMBEDDING = 24
    TOPK_MX = 25
    INDEX_PUT = 26
    SCATTER_ND = 27
    RANDOM_UNIFORM = 28
    GATHER_PT = 29
    BINARY = 30
    TENSORFLOW_NMS_V5 = 31
    GENERATE_PROPOSALS = 32
    BBOX_TRANSFORM = 33
    BOX_WITH_NMS_LIMIT = 34
    COLLECT_RPN_PROPOSALS = 35
    DISTRIBUTE_FPN_PROPOSALS = 36
    DISTRIBUTE_FPN_PROPOSALS_ROI_ALIGN_CONCAT = 37
    PYTORCH_ROI_ALIGN = 38
    AFFINE_GRID_GENERATOR = 39
    GRID_SAMPLER = 40
    AFFINE_GRID_SAMPLER = 41
    RANDOM_UNIFORM_INT = 42
    TOPK_ASCENDING = 43
    PYTORCH_INDEX = 44
    EMBEDDING_BAG = 45
    ONNX_NMS = 46
    DEFORM_GATHER = 47
    DEFORM_PSROIPOOLING = 48
    PADDLE_YOLO_BOX = 49
    PADDLE_MULTICLASS_NMS = 50
    PADDLE_DEFORM_CONV = 51
    PADDLE_MATRIX_NMS = 52
    REVERSE_SEQUENCE = 53
    FULL_INDEX = 54
    ADAPTIVE_AVERAGE_POOL_3D = 55
    TENSOR_SCATTER_OP = 56
    REPEAT_INTERLEAVE = 57
    PADDLE_DENSITY_PRIOR_BOX = 58
    PADDLE_BOX_CODER = 59
    DEBUG = 88888


class GDMAOpType(Enum):
    LOAD = 0
    STORE = 1
    MOVE = 2
    LD_G2L2 = 3
    UNKNOWN = -1


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


# BModel profile information
class IterSummary(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("iteration", ct.c_uint32),
        ("subnet_id", ct.c_uint32),
        ("subnet_type", ct.c_uint32),
        ("begin_usec", ct.c_uint64),
        ("end_usec", ct.c_uint64),
        ("extra_data", ct.c_uint64),
    ]


class MCURecord(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("profile_id", ct.c_uint32),
        ("type", ct.c_uint32),
        ("id", ct.c_uint64),
        ("begin_cycle", ct.c_uint64),
        ("end_cycle", ct.c_uint64),
    ]


class TIUProfile(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint64, 16),
        ("computation_load", ct.c_uint64, 48),
        ("num_read", ct.c_uint32),
        ("num_read_stall", ct.c_uint32),
        ("num_write", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class DMAProfile(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint16),
        ("reserved", ct.c_uint16),
        ("d0_aw_bytes", ct.c_uint32),
        ("d0_wr_bytes", ct.c_uint32),
        ("d0_ar_bytes", ct.c_uint32),
        ("d1_aw_bytes", ct.c_uint32),
        ("d1_wr_bytes", ct.c_uint32),
        ("d1_ar_bytes", ct.c_uint32),
        ("gif_aw_bytes", ct.c_uint32),
        ("gif_wr_bytes", ct.c_uint32),
        ("gif_ar_bytes", ct.c_uint32),
        ("d0_wr_valid_cyc", ct.c_uint32),
        ("d0_rd_valid_cyc", ct.c_uint32),
        ("d1_wr_valid_cyc", ct.c_uint32),
        ("d1_rd_valid_cyc", ct.c_uint32),
        ("gif_wr_valid_cyc", ct.c_uint32),
        ("gif_rd_valid_cyc", ct.c_uint32),
        ("d0_wr_stall_cyc", ct.c_uint32),
        ("d0_rd_stall_cyc", ct.c_uint32),
        ("d1_wr_stall_cyc", ct.c_uint32),
        ("d1_rd_stall_cyc", ct.c_uint32),
        ("gif_wr_stall_cyc", ct.c_uint32),
        ("gif_rd_stall_cyc", ct.c_uint32),
        ("d0_aw_end", ct.c_uint32),
        ("d0_aw_st", ct.c_uint32),
        ("d0_ar_end", ct.c_uint32),
        ("d0_ar_st", ct.c_uint32),
        ("d0_wr_end", ct.c_uint32),
        ("d0_wr_st", ct.c_uint32),
        ("d0_rd_end", ct.c_uint32),
        ("d0_rd_st", ct.c_uint32),
        ("d1_aw_end", ct.c_uint32),
        ("d1_aw_st", ct.c_uint32),
        ("d1_ar_end", ct.c_uint32),
        ("d1_ar_st", ct.c_uint32),
        ("d1_wr_end", ct.c_uint32),
        ("d1_wr_st", ct.c_uint32),
        ("d1_rd_end", ct.c_uint32),
        ("d1_rd_st", ct.c_uint32),
        ("gif_aw_reserved1", ct.c_uint32),
        ("gif_aw_reserved2", ct.c_uint32),
        ("gif_ar_end", ct.c_uint32),
        ("gif_ar_st", ct.c_uint32),
        ("gif_wr_end", ct.c_uint32),
        ("gif_wr_st", ct.c_uint32),
        ("gif_rd_end", ct.c_uint32),
        ("gif_rd_st", ct.c_uint32),
    ]
