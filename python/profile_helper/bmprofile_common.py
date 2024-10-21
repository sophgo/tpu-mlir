#!/usr/bin/python3
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
from collections import namedtuple
import struct as st

from bmprofile_utils import *

class Arch(Enum):
    UNKNOWN = -1
    bm1684 = 1
    bm1684x = 3
    bm1688 = 4
    bm1690 = 5
class BlockType(Enum):
    UNKNOWN = -1
    SUMMARY = 1
    COMPILER_LOG = 2
    MONITOR_BD = 3
    MONITOR_GDMA = 4
    DYN_DATA = 5
    DYN_EXTRA = 6
    FIRMWARE_LOG = 7
    COMMAND = 8
    BMLIB = 9
    BMLIB_EXTRA = 10
    MONITOR_SDMA = 11
    MONITOR_CDMA = 12

class DynExtraType(Enum):
    STRING=0
    BINARY=1
    CUSTOM=100
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

class DynRecord(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("profile_id", ct.c_uint32),
        ("type", ct.c_uint32),
        ("id", ct.c_uint64),
        ("begin_cycle", ct.c_uint64),
        ("end_cycle", ct.c_uint64),
    ]

DynExtra = namedtuple("DynExtra", "profile_id type content")

class BMLibApi(Enum):
    UNKNOWN  = -1
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
class BMLibMemDir(Enum):
    UNKNOWN = -1
    HOST2CHIP=0
    CHIP2HOST=1
    CHIP2CHIP=2

class BMLibMemOpType(Enum):
    UNKNOWN = -1
    ALLOC = 0
    FREE = 1
    INVALIDATE = 2
    FLUSH = 3
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
    if dtype in [ DataType.FP32, DataType.INT32, DataType.UINT32]:
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
    QUANT_DIV = 121
    GROUP_NORM = 122
    ROI_ALIGN = 123
    EMBEDDING_BAG = 124
    TRIANGULARIZE = 125
    INDEX_PUT = 126
    MASKED_FILL = 127
    GLU = 128
    DEFORM_GATHER = 129
    SCATTERND = 130
    LAYER_NORM = 131
    REQUANT_FP32 = 132
    REQUANT_INT = 133
    DEQUANT_FP32 = 134
    CLIP = 135
    DEQUANT_INT = 136
    SWAP_DIM_INNER = 137
    SWAP_CHANNEL = 138
    SCALE_LUT = 139
    PIXEL_NORM = 140
    NORMAL_SPARSE_CONV3D = 141
    SUBM_SPARSE_CONV3D = 142
    SHAPE_UNSQUEEZE= 143
    UNSQUEEZE = 144
    DECONV3D = 145
    YOLOV5_DETECT_OUT = 146
    ONNX_NMS = 147
    YOLOV8_DETECT_OUT = 148
    SHAPE_ARITH = 149
    RANGE = 150
    RELATIVE_POSITION = 151
    INSTANCENORM = 152
    SCATTERELEMENTS = 153
    RMSNORM = 154
    A16_MATMUL = 155
    FATTENTION = 156
    GATHERND = 157
    GRIDSAMPLER = 158
    GATHERELEMENTS = 159
    SHAPE_CONSTANT_FILL = 160
    MASKRCNNRPNGETBBOXES = 161
    MASKRCNNBBOXPOOLER = 162
    MASKRCNNGETBBOXB = 163
    MASKRCNNMASKPOOLER = 164
    SHAPE_CLIP = 165
    SHAPE_POW = 166
    RANDNLIKE = 167
    SHAPE_TRANSPOSE = 168

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

class TensorInfo():
    def __init__(self):
        self.tensor_id = -1
        self.shape = None
        self.dtype = DataType.UNKNOWN
        self.is_const = False
        self.gaddr = -1
        self.gsize = 0
        self.loffset = -1
        self.nslice = 0
        self.hslice = 0
        self.l2addr = 0
        self.in_layer = None
        self.out_layers = []

    def __str__(self):
        shape_str = "x".join([str(s) for s in self.shape])
        const_str = "CONST" if self.is_const else ""
        slice_str = ""
        if self.nslice > 0 and self.hslice > 0:
            slice_str = "nslice={} hslice={}".format(self.nslice, self.hslice)
        return "tensor_id={} [{}] {} {} {}".format(self.tensor_id, shape_str, self.dtype.name, const_str, slice_str)

class LayerInfo():
    def __init__(self):
        self.layer_id = -1
        self.layer_type = None
        self.layer_name = ""
        self.is_local = False
        self.in_tensors = []
        self.out_tensors = []
        self.group_id = -1
        self.total_size = 0
        self.feature_size = 0
        self.weight_size = 0
        self.gdma_op = None
        self.gdma_tensor = None
        self.begin_usec = None
        self.end_usec = None
        self.gdma_nodes = []
        self.bd_nodes = []

    def add_input(self, tensor):
        if tensor in self.in_tensors:
            return
        self.in_tensors.append(tensor)
        tensor.out_layers.append(self)

    def add_output(self, tensor):
        if tensor in self.out_tensors:
            return
        self.out_tensors.append(tensor)
        tensor.in_layer = self

    def set_gdma_tensor(self, tensor):
        self.gdma_tensor = tensor

    def update_time(self, begin_usec, end_usec):
        if self.begin_usec is None:
            self.begin_usec = begin_usec
        else:
            self.begin_usec = min(begin_usec, self.begin_usec)
        if self.end_usec is None:
            self.end_usec = end_usec
        else:
            self.end_usec = max(end_usec, self.end_usec)

    def io_info(self):
         ins_str ="ins=[" + (",".join([str(t) for t in self.in_tensors]))+"]"
         outs_str = "outs=[" + (",".join([str(t) for t in self.out_tensors])) + "]"
         return ins_str+"," + outs_str

    def info(self, sep='<br>'):
        all_info = []
        prefix = "local_layer" if self.is_local else "global_layer"
        if self.layer_name != "":
            prefix += ":{}".format(self.layer_name)
        all_info.append(prefix)
        if self.layer_type is not None:
            all_info.append("==ins==" + sep +
                            (sep.join([str(t) for t in self.in_tensors])))
            all_info.append("==outs==" + sep +
                            (sep.join([str(t) for t in self.out_tensors])))
        if self.gdma_op is not None:
            all_info.append("==gdma==")
            all_info.append(str(self.gdma_tensor))
        all_info.append("========")
        if self.group_id >= 0:
            all_info.append("group_id={}".format(self.group_id))
        if self.feature_size >= 0:
            all_info.append("feature_size={}".format(self.feature_size))
        if self.weight_size >= 0:
            all_info.append("weight_size={}".format(self.weight_size))
        if self.total_size >= 0:
            all_info.append("total_size={}".format(self.total_size))
        return sep + (sep.join(all_info))
    def __str__(self):
        prefix = "local" if self.is_local else "global"
        prefix += "-{}".format(str(self.layer_type).split(".")[-1])
        return prefix

def enum_name(val):
    return str(val).split(".")[-1]

BDSimRecord = namedtuple("BDSimRecord", " ".join(["layer_id", "op_type", "bd_id", "gdma_id", "start_time", "end_time", "cost_time"]))
GDMASimRecord = namedtuple("GDMASimRecord", " ".join(["layer_id", "tensor_id", "op_type", "bd_id", "gdma_id", "start_time", "end_time", "cost_time", "byte_size", "direction", "bandwidth", "info"]))
class StaticRunNode():
    __run_id = -1

    def __init__(self):
        self.__class__.__run_id += 1
        self.run_id = self.__class__.__run_id
        self.type = None
        self.bd_id = -1
        self.gdma_id = -1
        self.gdma_dir = None
        self.gdma_func = None
        self.bd_func = None
        self.layer = None
        self.command = None
        self.sim_info = None
        self.pmu_info = None
    def __str__(self):
        prefix = ""
        if self.layer is not None:
            prefix = str(self.layer)+", "

        if str(self.type) == "EngineType.BD":
            return prefix + "type={}, bd_id={}, gdma_id={}, func={}" .format(
                enum_name(self.type), self.bd_id, self.gdma_id, enum_name(self.bd_func))
        else:
            return prefix + "type={}, bd_id={}, gdma_id={}, func={}" .format(
                enum_name(self.type), self.bd_id, self.gdma_id, enum_name(self.gdma_func)+"-"+enum_name(self.gdma_dir))

class SubnetInfo():
    def __init__(self):
        self.subnet_id = -1
        self.layer_list = []
        self.command_info = None
        self.gdma_nodes = []
        self.bd_nodes = []
        self.sim_info = None

    def __str__(self):
        str_info = "subnet_id={} layer_num={} run_nodes={}".format(
            self.subnet_id, len(self.layer_list), len(self.gdma_nodes), len(self.bd_nodes))
        return str_info

def normal_fw_layer_type(fw_type: FWLayerType):
    name = fw_type.name
    return LayerType.__members__[name]

MemBlock = namedtuple("MemBlock", "addr size alloc_time free_time type desc")

MemRecord = namedtuple("MemRecord", "mem_type op_type addr size usage desc")

class GlobalInfo():
    def __init__(self):
        self.subnet_list = []
        self.arch = Arch.UNKNOWN
        self.mem_info = []
        self.archlib = None
        self.freq = None
        self.net_name = None
        self.flops = 0
        self.no_perf_data = False

    def set_arch(self, arch):
        self.arch = arch
        if self.archlib is None:
            self.archlib = load_arch_lib(arch)
        assert self.archlib is not None


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
