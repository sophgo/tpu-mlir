#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 11:47
# @Author  : chongqing.zeng@sophgo.com
# @Project: PerfAI
from enum import Enum
import ctypes as ct
from src.common.common import dictStructure

GDMACyclePeriod = 1.0 / 1000
BDCyclePeriod = 1.0 / 1000
PeriodFixed = False


class EngineType(Enum):
    BD = 0
    GDMA = 1
    GDE = 2
    SORT = 3
    NMS = 4
    CDMA = 5
    UNKNOWN = -1


class Arch(Enum):
    UNKNOWN = -1
    bm1684 = 1
    bm1684x = 3


class DynRecordType(Enum):
    FUNC = 0
    NODE_SET = 1
    NODE_WAIT = 2
    CDMA = 3
    GDE = 4
    SORT = 5
    NMS = 6
    CUSTOM = 100
    UNKNOWN = -1


class BDFuncType(Enum):
    CONV = 0
    PD = 1
    MM = 2
    AR = 3
    RQDQ = 4
    TRANS_BC = 5
    SG = 6
    LAR = 7
    SFU = 9
    LIN = 10
    CMP = 13
    VC = 14
    SYS = 15
    UNKNOWN = -1


class CONV(Enum):
    CONV_NORMAL = 0
    CONV_WRQ = 1
    CONV_WRQ_RELU = 2
    UNKNOWN = -1


class PD(Enum):
    PD_DEPTHWISE = 0
    PD_AVG = 1
    PD_DEPTHWISE_RELU = 2
    PD_MAX = 4
    PD_ROI_DEPTHWISE = 5
    PD_ROI_AVG = 6
    PD_ROI_MAX = 7
    UNKNOWN = -1


class MM(Enum):
    MM_NORMAL = 1
    MM_WRQ = 2
    MM_WRQ_RELU = 3
    MM_NN = 4
    MM_NT = 5
    MM_TT = 6
    UNKNOWN = -1


class AR(Enum):
    AR_MUL = 0
    AR_NOT = 1
    AR_ADD = 2
    AR_SUB = 3
    AR_MAX = 4
    AR_MIN = 5
    AR_LOGIC_SHIFT = 6
    AR_AND = 7
    AR_OR = 8
    AR_XOR = 9
    AR_SG = 10
    AR_SE = 11
    AR_DIV = 12
    AR_SL = 13
    AR_DATA_CONVERT = 14
    AR_ADD_SATU = 15
    AR_SUB_SATU = 16
    AR_CLAMP = 17
    AR_MAC = 18
    AR_COPY = 19
    AR_MUL_SATU = 20
    AR_ARITH_SHIFT = 21
    AR_ROTATE_SHIFT = 22
    AR_MULDHR = 23
    AR_EU_IDX_GEN = 24
    AR_NPU_IDX_GEN = 25
    AR_ABS = 26
    AR_FSUBABS = 27
    AR_COPY_MB = 28
    AR_GET_FIRST_ONE = 29
    AR_GET_FIRST_ZERO = 30
    UNKNOWN = -1


class RQDQ(Enum):
    RQ_0 = 0
    RQ_1 = 1
    DQ_0 = 3
    DQ_1 = 4
    UNKNOWN = -1


class TRANS_BC(Enum):
    TRAN_C_W_TRANSPOSE = 0
    TRAN_W_C_TRANSPOSE = 1
    LANE_COPY = 2
    LANE_BROAD = 3
    STATIC_BROAD = 4
    STATIC_DISTRIBUTE = 5
    UNKNOWN = -1


class SG(Enum):
    PL_gather_d1coor = 0
    PL_gather_d2coor = 1
    PL_gather_rec = 2
    PL_scatter_d1coor = 3
    PL_scatter_d2coor = 4
    PE_S_gather_d1coor = 5
    PE_S_scatter_d1coor = 6
    PE_M_gather_d1coor = 7
    PE_S_mask_select = 8
    PE_S_nonzero = 9
    PE_S_scatter_pp_d1coor = 10
    PE_S_gather_hzd = 13
    PE_S_scatter_hzd = 14
    PE_S_mask_selhzd = 15
    PE_S_nonzero_hzd = 16
    PE_S_gather_line = 17
    PE_S_scatter_line = 18
    PE_S_mask_seline = 19
    UNKNOWN = -1


class SFU(Enum):
    SFU_TAYLOR_4X = 12
    SFU_TAYLOR = 13
    SFU_NORM = 15
    SFU_RSQ = 17
    UNKNOWN = -1


class LIN(Enum):
    LIN_MAC = 1
    LIN_ADD_SQR = 20
    LIN_SUB_SQR = 21
    UNKNOWN = -1


class CMP(Enum):
    CMP_GT_AND_SG = 22
    CMP_SG = 23
    CMP_SE = 24
    CMP_LT_AND_SL = 25
    CMP_SL = 26
    UNKNOWN = -1


class VC(Enum):
    VC_MUL = 0
    VC_ADD = 2
    VC_SUB = 3
    VC_MAX = 4
    VC_MIN = 5
    VC_AND = 7
    VC_OR = 8
    VC_XOR = 9
    VC_SG = 10
    VC_SE = 11
    VC_DIV = 12
    VC_SL = 13
    VC_ADD_SATU = 15
    VC_SUB_SATU = 16
    VC_MUL_SATU = 20
    VC_MULDHR = 23
    UNKNOWN = -1


class SYS(Enum):
    INSTR_BARRIER = 0
    SPB = 1
    SWR = 2
    SWR_FROM_LMEM = 3
    SWR_COL_FROM_LMEM = 4
    SYNC_ID = 5
    DATA_BARRIER = 6
    SYS_END = 31
    UNKNOWN = -1


class GDMAFuncType(Enum):
    TENSOR = 0
    MATRIX = 1
    MASKED_SEL = 2
    GENERAL = 3
    CW_TRANS = 4
    NONZERO = 5
    SYS = 6
    GATHER = 7
    SCATTER = 8
    UNKNOWN = -1


class GDMASubFunc(Enum):
    NONE = 0
    TRANS = 1
    COLLECT = 2
    BROADCAST = 3
    DISTRIBUTE = 4
    BANK4_COPY = 5
    BANK4_BDC = 6
    UNKNOWN = -1


class GDMADirection(Enum):
    S2L = 0
    L2S = 1
    S2S = 2
    L2L = 3
    UNKNOWN = -1


class GDMAFormat(Enum):
    UNKNOWN = -1
    INT8 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    INT16 = 3
    INT32 = 4
    BFLOAT16 = 5


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


class DynExtraType(Enum):
    STRING = 0
    BINARY = 1
    CUSTOM = 100


class SubnetType(Enum):
    UNKNOWN = -1
    TPU = 0
    CPU = 1
    MERGE = 2
    SWITCH = 3


def show_arch_info():
    print("BM1684X")


class BDProfileFormat(dictStructure):
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


class GDMAProfileFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint32, 20),
        ("reserved", ct.c_uint32, 12),
        ("d0_aw_bytes", ct.c_uint32),
        ("d0_wr_bytes", ct.c_uint32),
        ("d0_ar_bytes", ct.c_uint32),
        ("d1_aw_bytes", ct.c_uint32),
        ("d1_wr_bytes", ct.c_uint32),
        ("d1_ar_bytes", ct.c_uint32),
        ("gif_aw_bytes", ct.c_uint32),
        ("gif_wr_bytes", ct.c_uint32),
        ("gif_ar_bytes", ct.c_uint32),
        ("gif_12sram_aw_bytes", ct.c_uint32),
        ("gif_l2sram_w_bytes", ct.c_uint32),
        ("gif_l2sram_ar_bytes", ct.c_uint32),
        ("reserved1", ct.c_uint32),
        ("axi_d0_wr_valid_bytes", ct.c_uint32),
        ("axi_d0_rd_valid_bytes", ct.c_uint32),
        ("axi_d1_wr_valid_bytes", ct.c_uint32),
        ("axi_d1_rd_valid_bytes", ct.c_uint32),
        ("gif_fmem_wr_valid_bytes", ct.c_uint32),
        ("gif_fmem_rd_valid_bytes", ct.c_uint32),
        ("gif_l2sram_wr_valid_bytes", ct.c_uint32),
        ("gif_l2sram_rd_valid_bytes", ct.c_uint32),
        ("axi_d0_wr_stall_bytes", ct.c_uint32),
        ("axi_d0_rd_stall_bytes", ct.c_uint32),
        ("axi_d1_wr_stall_bytes", ct.c_uint32),
        ("axi_d1_rd_stall_bytes", ct.c_uint32),
        ("gif_fmem_wr_stall_bytes", ct.c_uint32),
        ("gif_fmem_rd_stall_bytes", ct.c_uint32),
        ("gif_l2sram_wr_stall_bytes", ct.c_uint32),
        ("gif_12sram_rd_stall_bytes", ct.c_uint32),
        ("axi_d0_aw_end", ct.c_uint32),
        ("axi_d0_aw_st", ct.c_uint32),
        ("axi_d0_ar_end", ct.c_uint32),
        ("axi_d0_ar_st", ct.c_uint32),
        ("axi_d0_wr_end", ct.c_uint32),
        ("axi_d0_wr_st", ct.c_uint32),
        ("axi_d0_rd_end", ct.c_uint32),
        ("axi_d0_rd_st", ct.c_uint32),
        ("axi_d1_aw_end", ct.c_uint32),
        ("axi_d1_aw_st", ct.c_uint32),
        ("axi_d1_ar_end", ct.c_uint32),
        ("axi_d1_ar_st", ct.c_uint32),
        ("axi_d1_wr_end", ct.c_uint32),
        ("axi_d1_wr_st", ct.c_uint32),
        ("axi_d1_rd_end", ct.c_uint32),
        ("axi_d1_rd_st", ct.c_uint32),
    ]

    def _cycles(self):
        return self.inst_end_time - self.inst_start_time

    @property
    def read_bw(self):
        d_read = self.d0_ar_bytes + self.d1_ar_bytes
        gif_read = self.gif_ar_bytes
        l2_read = self.gif_l2sram_ar_bytes
        return (d_read + gif_read + l2_read) / self._cycles()

    @property
    def write_bw(self):
        d_write = self.d0_aw_bytes + self.d1_aw_bytes
        gif_write = self.gif_wr_bytes
        l2_write = self.gif_l2sram_aw_bytes
        return (d_write + gif_write + l2_write) / self._cycles()

    @property
    def direction(self):
        _from = _to = ""
        if self.d0_ar_bytes + self.d1_ar_bytes:
            _from = "ddr"
        elif self.gif_ar_bytes:
            _from = "lmem"

        if self.d0_aw_bytes + self.d1_aw_bytes:
            _to = "ddr"
        elif self.gif_aw_bytes:
            _to = "lmem"

        return {
            "ddr->ddr": GDMADirection.S2S,
            "ddr->lmem": GDMADirection.S2L,
            "lmem->ddr": GDMADirection.L2S,
            "lmem->lmem": GDMADirection.L2L,
            "->": GDMADirection.UNKNOWN,
        }[f"{_from}->{_to}"]


tiu_func_name_dict = {
    (0, 0): 'conv_normal',
    (0, 1): 'conv_wrq',
    (0, 2): 'conv_wrq_relu',
    (2, 1): 'mm_normal',
    (2, 2): 'mm_wrq',
    (2, 3): 'mm_wrq_relu',
    (2, 4): 'mm2_nn',
    (2, 5): 'mm2_nt',
    (2, 6): 'mm2_tt',
    (13, 22): 'cmp_gt_and_cmp_sel_gt',
    (13, 23): 'cmp_sel_gt',
    (13, 24): 'cmp_sel_eq',
    (13, 25): 'cmp_lt_and_cmp_sel_lt',
    (13, 26): 'cmp_sel_lt',
    (9, 12): 'tailor_4x',
    (9, 13): 'tailor',
    (9, 15): 'normalize',
    (9, 17): 'rsqrt',
    (14, 0): 'mul',
    (14, 2): 'add',
    (14, 3): 'sub',
    (14, 4): 'max',
    (14, 5): 'min',
    (14, 7): 'and',
    (14, 8): 'or',
    (14, 9): 'xor',
    (14, 10): 'select_great',
    (14, 11): 'select_equal',
    (14, 12): 'div',
    (14, 13): 'select_less',
    (14, 15): 'add_satu',
    (14, 16): 'sub_satu',
    (14, 20): 'mul_satu',
    (14, 23): 'mul_dhr',
    (10, 1): 'mac',
    (10, 20): 'a_plus_b_square',
    (10, 21): 'a_minus_b_square',
    (3, 0): 'mul',
    (3, 1): 'not',
    (3, 2): 'add',
    (3, 3): 'sub',
    (3, 4): 'max',
    (3, 5): 'min',
    (3, 6): 'logic_shift',
    (3, 7): 'and',
    (3, 8): 'or',
    (3, 9): 'xor',
    (3, 10): 'select_great',
    (3, 11): 'select_equal',
    (3, 12): 'div',
    (3, 13): 'select_less',
    (3, 14): 'data_convert',
    (3, 15): 'add_satu',
    (3, 16): 'sub_satu',
    (3, 17): 'clamp',
    (3, 18): 'mac',
    (3, 19): 'copy',
    (3, 20): 'mul_satu',
    (3, 21): 'arith_shift',
    (3, 22): 'rotate_shift',
    (3, 23): 'mul_dhr',
    (3, 26): 'abs',
    (3, 27): 'fsubabs',
    (3, 28): 'copy_mb',
    (3, 29): 'get_first_one',
    (3, 30): 'get_first_zero',
    (3, 24): 'eu_idx_gen',
    (3, 25): 'npu_idx_gen',
    (1, 0): 'depthwise',
    (1, 1): 'avg_pooling',
    (1, 2): 'depthwise_relu',
    (1, 4): 'max_pooling',
    (1, 5): 'roi_depthwise',
    (1, 6): 'roi_avg_pooling',
    (1, 7): 'roi_max_pooling',
    (4, 0): 'rq_0',
    (4, 1): 'rq_1',
    (4, 2): 'rq_2',
    (4, 3): 'dq_0',
    (4, 4): 'dq_1',
    (4, 5): 'dq_2',
    (6, 0): 'pl_gather_d1coor',
    (6, 1): 'pl_gather_d2coor',
    (6, 2): 'pl_gather_rec',
    (6, 3): 'pl_scatter_d1coor',
    (6, 4): 'pl_scatter_d2coor',
    (6, 5): 'pe_s_gather_d1coor',
    (6, 6): 'pe_s_scatter_d1coor',
    (6, 7): 'pe_m_gather_d1coor',
    (6, 8): 'pe_s_mask_select',
    (6, 9): 'pe_s_nonzero',
    (6, 10): 'pe_s_scatter_pp_d1coor',
    (6, 11): 'pl_gather_perw',
    (6, 12): 'pl_scatter_perw',
    (6, 13): 'pe_s_gather_hzd',
    (6, 14): 'pe_s_scatter_hzd',
    (6, 15): 'pe_s_mask_selhzd',
    (6, 16): 'pe_s_nonzero_hzd',
    (6, 17): 'pe_s_gather_line',
    (6, 18): 'pe_s_scatter_line',
    (5, 0): 'c_w_transpose',
    (5, 1): 'w_c_transpose',
    (5, 2): 'lane_copy',
    (5, 3): 'lane_broad',
    (5, 4): 'static_broad',
    (5, 5): 'static_distribute',
    (15, 8): 'send_msg',
    (15, 9): 'wait_msg'
}

dma_func_type_dict = {
    0: 'DMA_tensor',
    1: 'DMA_matrix',
    2: 'DMA_masked_select',
    3: 'DMA_general',
    4: 'DMA_cw_transpose',
    5: 'DMA_nonzero',
    6: 'DMA_sys',
    7: 'DMA_gather',
    8: 'DMA_scatter'
}

dma_func_name_dict = {
    (0, 0): 'DMA_tensor', (0, 1): 'NC transpose', (0, 2): 'collect', (0, 3): 'broadcast',
    (0, 4): 'distribute', (0, 5): 'lmem 4 bank copy', (0, 6): 'lmem 4 bank broadcast',
    (1, 0): 'DMA_matrix', (1, 1): 'matrix transpose',
    (2, 0): 'DMA_masked_select', (2, 1): 'ncw mode',
    (3, 0): 'DMA_general', (3, 1): 'broadcast',
    (4, 0): 'cw transpose',
    (5, 0): 'DMA_nonzero',
    (6, 0): 'chain end', (6, 1): 'nop', (6, 2): 'sys_tr_wr', (6, 3): 'sys_send', (6, 4): 'sys_wait',
    (7, 0): 'DMA_gather',
    (8, 0): 'DMA_scatter',
    (9, 0): 'w reverse', (9, 1): 'h reverse', (9, 2): 'c reverse', (9, 3): 'n reverse',
    (10, 0): 'non-random-access', (10, 1): 'random-access',
    (11, 0): 'non-random-access', (11, 1): 'random-access'
}

data_type_dict = {
    0: 'INT8',
    1: 'FP16',
    2: 'FP32',
    3: 'INT16',
    4: 'INT32',
    5: 'BFP16',
    6: 'INT4',
    '': 'None',
    '-': 'None'
}

data_size_dict = {
    '0': 1,
    '1': 2,
    '2': 4,
    '3': 2,
    '4': 4,
    '5': 2,
    '6': 0.5,
    '': 'None',
    '-': 'None'
}
