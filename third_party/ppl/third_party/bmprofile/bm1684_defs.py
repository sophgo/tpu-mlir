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
from bmprofile_common import FWGDMAType, FWLayerType, dictStructure
from bmprofile_utils import enum_cast

arch_name="BM1684"
class EngineType(Enum):
    BD = 0
    GDMA = 1
    CDMA = 2
    GDE = 3
    UNKNOWN = -1

class DynRecordType(Enum):
    UNKNOWN = -1
    FUNC = 0
    NODE_SET = 1
    NODE_WAIT = 2
    CDMA = 3
    GDE = 4
    CUSTOM = 100

class StoreMode(Enum):
    UNKNOWN    = -1
    FP32_1N    = 0
    INT8_1N    = 1
    INT16_1N   = 2
    INT16_2N   = 3
    INT8_4N    = 4
    FP32_2IC   = 5
    IC4_OC4_4N = 6
    INT16_4N   = 7

class BDFuncType(Enum):
    UNKNOWN = -1
    CONV_NEURON = 0
    DEPTHWISE_OR_POOLING = 1
    FC = 2
    TENSOR_ARITHMETIC = 3
    FC2 = 4
    CONV_CORRELATION = 5
    TABLE_LOOKUP = 6
    MD_SUM = 7
    MD_SCALAR = 8
    MD_SFU = 9
    MD_LINEAR = 10
    LOCALMEM_ARANGE = 11
    DECOMPRESS = 12
    MD_CMP = 13
    VECTOR_CORRELATION = 14

class BDEUType(Enum):
    EU_MUL = 0
    EU_MAC = 1
    EU_ADD = 2
    EU_SUB = 3
    EU_MAX = 4
    EU_MIN = 5
    EU_SHIFT = 6
    EU_AND = 7
    EU_OR = 8
    EU_XOR = 9
    EU_SELECT_GT = 10
    EU_SELECT_EQ = 11
    EU_DIVIDE = 12
    EU_TAYLOR = 13
    EU_FP32_TO_INT = 14
    EU_INT_NORMALIZE = 15
    EU_FP32_NORMALIZE = 16
    EU_RSQRT = 17
    EU_ADD_TREE = 18
    EU_COPY = 19
    EU_AB_SQUARE = 20
    EU_UNKNOWN = -1


class GDMAFuncType(Enum):
    NORMAL = 0
    TRANS = 1
    LRN_SHIFT = 2
    FORMAT = 3
    CONSTANT = 4
    CW_TRANS = 5
    WINOGRAD = 6
    FILTER = 7
    FILTER_RES_COUNTER = 8
    UNKNOWN = -1

class GDMADirection(Enum):
    S2L = 0
    L2S = 1
    S2S = 2
    L2L = 3
    UNKNOWN = -1

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

GDMACyclePeriod= 1.0/575
BDCyclePeriod= 1.0/550
PeriodFixed = True

class MemType(Enum):
    GLOBAL = 0
    DTCM = 1
    L2MEM = 2
    LOCAL = 3
    UNKNOWN = -1

class GDMAFormat(Enum):
    UNKNOWN=-1
    FLOAT32 = 0
    INT16 = 1
    UINT8 = 2
    INT8 = 3
    FLOAT16 = 4

class BDFormat(Enum):
    UNKNOWN = -1
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT16 = 3
    INT32 = 4

def __get_format_len(f):
    if isinstance(f, GDMAFormat):
        if f == GDMAFormat.FLOAT32:
            return 4
        elif f == GDMAFormat.INT16 or f == GDMAFormat.FLOAT16:
            return 2
        else:
            return 1
    else:
        if f == BDFormat.FP32 or f == BDFormat.INT32:
            return 4
        elif f == BDFormat.FP16 or f == BDFormat.INT16:
            return 2
        else:
            return 1

def __get_global_mem_size_info(addr, n, c, h, w, ns, cs, hs, ws, f, is_out, desc="", result_add=False):
    format_len = __get_format_len(f)
    data_size = n*c*h*w*format_len
    cover_size = ((n-1)*ns+ (c-1)*cs+ (h-1)*hs+ (w-1)*ws+1)*format_len
    # print("--global-->", addr, n, c, h, w, ns, cs, hs, ws, f)
    desc += ", dtype={}, shape=({},{},{},{}), stride=({},{},{},{})".format(f.name, n, c, h, w, ns, cs, hs, ws)
    if result_add:
        desc+= ", res_add=True"
    return [addr, data_size, cover_size, True, is_out, desc]

ID_WIDTH=16
EU_NUM = 32
NPU_NUM = 64
LOCAL_MEM_SHIFT=19
LOCAL_MEM_SIZE=1<<LOCAL_MEM_SHIFT #512k
TOTAL_LOCAL_MEM_SIZE = LOCAL_MEM_SIZE * NPU_NUM
LOCAL_MEM_ADDR_MASK = TOTAL_LOCAL_MEM_SIZE -1
LOCAL_BANK_NUM=8
LOCAL_BANK_SIZE=LOCAL_MEM_SIZE//LOCAL_BANK_NUM

def local_mem_partition():
    partitions = []
    for i in range(LOCAL_BANK_NUM):
        partitions.append([i*LOCAL_BANK_SIZE, LOCAL_BANK_SIZE, "BANK[{}]".format(i)])
    return partitions

MemRecord=namedtuple("MemRecord", "addr data_size cover_size is_global is_out desc")
def __get_local_mem_size_info(addr, n, c, h, w, ns, cs, hs, ws, f, is_aligned, enable_stride, is_out, desc, format_len=None, result_add=False, store_mode=StoreMode.UNKNOWN):
    local_addr = addr & LOCAL_MEM_ADDR_MASK
    local_offset = local_addr & (LOCAL_MEM_SIZE-1)
    cidx = (local_addr & (~(LOCAL_MEM_SIZE-1)))>>LOCAL_MEM_SHIFT
    cnum = (cidx + c+NPU_NUM-1)//NPU_NUM
    real_type_len = __get_format_len(f)
    is_normal = store_mode == StoreMode.UNKNOWN
    if format_len is None:
        format_len = real_type_len
    divide_num = 1
    if store_mode == StoreMode.INT16_2N or store_mode == StoreMode.INT8_4N:
        format_len = 4
    if store_mode == StoreMode.INT16_2N:
        divide_num = 2
    if store_mode == StoreMode.INT8_4N:
        divide_num = 4

    data_size = n*c*h*w*format_len
    if not enable_stride:
        ws = 1
        hs = w
        if is_aligned:
            align_num = EU_NUM*4//format_len
            cs = (w*h + align_num-1)//align_num * align_num
        else:
            cs = w*h
        ns = cnum * cs
    cover_size = ((n-1)//divide_num* ns + (cnum-1)*cs + (h-1)*hs + (w-1)*ws)*format_len + ((n-1)%divide_num*real_type_len) + 1
    # print("--local-->", addr, n, c, h, w, ns, cs, hs, ws, f, cnum)
    stride_name = "user_stride" if enable_stride else "auto_stride"
    desc += ", {}, shape=({},{},{},{}), {}=({},{},{},{}), cidx={}, cnum={}".format(
        f.name, n, c, h, w, stride_name, ns, cs, hs, ws, cidx, cnum)
    if real_type_len != format_len:
        desc+= ", elem_size={}".format(format_len)
    if not is_normal:
        desc+= ", store={}".format(store_mode.name)
    if result_add:
        desc+= ", res_add=True"
    return [local_offset, data_size, cover_size, False, is_out, desc]

def matrix_desc(rows, cols, transposed, row_stride, desc=""):
    new_desc = desc
    if new_desc != "":
        new_desc += ", "
    new_desc += "Matrix({},{})".format(rows, cols)
    if transposed:
        new_desc += "-T"
    new_desc += ", stride={}".format(row_stride)
    return new_desc

def get_gdma_mem_record(cmd):
    cmd.DST_START_ADDR = cmd.DST_START_ADDR_H8<<32 | cmd.DST_START_ADDR_L32
    cmd.SRC_START_ADDR = cmd.SRC_START_ADDR_H8<<32 | cmd.SRC_START_ADDR_L32
    gdma_func = GDMAFuncType(cmd.SPECIAL_FUNC)
    gdma_dir = GDMADirection(cmd.DIRECTION)
    src_format = GDMAFormat(cmd.SRC_DATA_FORMAT)
    need_convert = gdma_func == GDMAFuncType.FORMAT
    dst_format = GDMAFormat(cmd.DST_DATA_FORMAT) if need_convert else src_format
    src_is_global = gdma_dir == GDMADirection.S2S or gdma_dir == GDMADirection.S2L
    dst_is_global = gdma_dir == GDMADirection.S2S or gdma_dir == GDMADirection.L2S
    is_lrn = gdma_func == GDMAFuncType.LRN_SHIFT
    is_winograd = gdma_func == GDMAFuncType.WINOGRAD
    cwtrans_used = gdma_func == GDMAFuncType.CW_TRANS
    need_transpose = gdma_func == GDMAFuncType.TRANS
    is_filter_move = gdma_func == GDMAFuncType.FILTER
    is_aligned = True
    result_add = cmd.ACC_WRITE_ENABLE == 1
    enable_stride = cmd.STRIDE_ENABLE == 1

    src_mem_info = None
    dst_mem_info = None
    mem_info = []
    desc = "{}-{}:id={}".format(gdma_func.name, gdma_dir.name, cmd.CMD_ID)
    format_len = __get_format_len(src_format)
    if cmd.COMMON_MODE:
        move_len = cmd.SRC_NSTRIDE * format_len
        if gdma_func != GDMAFuncType.CONSTANT:
            mem_info.append([cmd.SRC_START_ADDR, move_len, move_len, True, 0, desc])
        mem_info.append([cmd.DST_START_ADDR, move_len, move_len, True, 1, desc])
    else:
        src_nsize = cmd.SRC_NSIZE
        src_csize = cmd.SRC_CSIZE
        src_hsize = cmd.SRC_HSIZE
        src_wsize = cmd.SRC_WSIZE
        dst_nsize = cmd.DST_NSIZE
        dst_csize = cmd.DST_CSIZE
        dst_hsize = cmd.DST_HSIZE
        dst_wsize = cmd.DST_WSIZE

        if cmd.CHW_COPY:
            dst_nsize = src_nsize
            dst_csize = src_csize
            dst_hsize = src_hsize
            dst_wsize = src_wsize

        src_nstride = cmd.SRC_NSTRIDE
        src_cstride = cmd.SRC_CSTRIDE
        src_hstride = cmd.SRC_HSTRIDE
        src_wstride = cmd.SRC_WSTRIDE
        dst_nstride = cmd.DST_NSTRIDE
        dst_cstride = cmd.DST_CSTRIDE
        dst_hstride = cmd.DST_HSTRIDE
        dst_wstride = cmd.DST_WSTRIDE

        if is_lrn:
            # input local mem
            info = __get_local_mem_size_info(
                cmd.SRC_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride, src_format,
                is_aligned, enable_stride, 0, desc)
            mem_info.append(info)

            # output local mem
            info =  __get_local_mem_size_info(
                cmd.DST_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride, src_format,
                is_aligned, enable_stride, 1, desc, result_add=result_add)
            mem_info.append(info)

        elif is_filter_move:
            mask_local_addr = dst_nstride
            info = __get_local_mem_size_info(
                cmd.SRC_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride,
                src_format, is_aligned, enable_stride, 0, desc)
            mem_info.append(info)

            info = __get_local_mem_size_info(
                mask_local_addr, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride,
                src_format, is_aligned, enable_stride, 0, desc)
            mem_info.append(info)

            info = __get_local_mem_size_info(
                cmd.DST_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride,
                src_format, is_aligned, enable_stride, 1, desc, result_add=result_add)
            mem_info.append(info)

        elif cwtrans_used or is_winograd:
            dst_nsize = 1
            dst_hsize = 1
            dst_nstride = 0
            dst_hstride = 0
            src_nsize = 1
            src_hsize = 1
            src_nstride = 0
            src_hstride = 0
            info = __get_local_mem_size_info(
                cmd.SRC_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                src_nstride, src_cstride, src_hstride, src_wstride,
                src_format, is_aligned, enable_stride, 0, desc)
            mem_info.append(info)
            info = __get_local_mem_size_info(
                cmd.DST_START_ADDR, dst_nsize, dst_csize, dst_hsize, dst_wsize,
                dst_nstride, dst_cstride, dst_hstride, dst_wstride,
                dst_format,is_aligned, enable_stride, 1, desc, result_add=result_add)
            mem_info.append(info)
        elif cmd.SYS_MEM_TYPE == 0: # neuron
            if need_transpose:
                dst_nsize = src_csize
                dst_csize = src_nsize
            if gdma_func != GDMAFuncType.CONSTANT:
                if src_is_global:
                    info = __get_global_mem_size_info(cmd.SRC_START_ADDR,
                        src_nsize, src_csize, src_hsize, src_wsize,
                        src_nstride, src_cstride, src_hstride, src_wstride,
                        src_format, 0, desc)
                    mem_info.append(info)
                else:
                    info = __get_local_mem_size_info(
                        cmd.SRC_START_ADDR, src_nsize, src_csize, src_hsize, src_wsize,
                        src_nstride, src_cstride, src_hstride, src_wstride, src_format,
                        is_aligned, enable_stride, 0, desc)
                    mem_info.append(info)

            if dst_is_global:
                info = __get_global_mem_size_info(cmd.DST_START_ADDR,
                    dst_nsize, dst_csize, dst_hsize, dst_wsize,
                    dst_nstride, dst_cstride, dst_hstride, dst_wstride,
                    dst_format, 1, desc, result_add=result_add)
                mem_info.append(info)
            else:
                info = __get_local_mem_size_info(
                    cmd.DST_START_ADDR, dst_nsize, dst_csize, dst_hsize, dst_wsize,
                    dst_nstride, dst_cstride, dst_hstride, dst_wstride, dst_format,
                    is_aligned, enable_stride, 1, desc, result_add=result_add)
                mem_info.append(info)

        elif cmd.SYS_MEM_TYPE == 1: # matrix
            if gdma_dir == GDMADirection.S2L:
                cols = src_wsize
                rows = src_hsize
                row_stride = src_hstride
                mem_size = rows*cols*format_len
                cover_size = cols*row_stride*format_len if need_transpose else rows*row_stride*format_len
                desc = matrix_desc(rows, cols, need_transpose, row_stride, desc)
                if is_aligned:
                    cover_size = mem_size
                mem_info.append([cmd.SRC_START_ADDR, mem_size, cover_size, True, 0, desc])

                sec_len = dst_wsize
                sec_nstride = dst_nstride
                sec_cstride = dst_cstride
                sec_wstride = dst_wstride

                lm_rows = cols if need_transpose else rows
                lm_cols = rows if need_transpose else cols
                local_num = (lm_cols + sec_len -1)//sec_len

                info = __get_local_mem_size_info(
                    cmd.DST_START_ADDR, lm_rows, local_num, 1, sec_len,
                    sec_nstride, sec_cstride, 0, sec_wstride, dst_format,
                    is_aligned, enable_stride, 1, desc, result_add=result_add)
                mem_info.append(info)

            elif gdma_dir == GDMADirection.L2S:
                cols = dst_wsize
                rows = dst_hsize
                row_stride = dst_hstride
                sec_len = src_wsize
                sec_nstride = src_nstride
                sec_cstride = src_cstride
                sec_wstride = src_wstride
                desc = matrix_desc(rows, cols, need_transpose, row_stride, desc)

                lm_rows = cols if need_transpose else rows
                lm_cols = rows if need_transpose else cols
                local_num = (lm_cols + sec_len -1)//sec_len

                info = __get_local_mem_size_info(
                    cmd.SRC_START_ADDR, lm_rows, local_num, 1, sec_len,
                    sec_nstride, sec_cstride, 0, sec_wstride,
                    src_format, is_aligned, enable_stride, 0, desc)
                mem_info.append(info)

                mem_size = rows*cols*format_len
                cover_size = cols*row_stride*format_len if need_transpose else rows*row_stride*format_len
                if is_aligned:
                    cover_size = mem_size
                if result_add:
                    desc += ", res_add=True"
                mem_info.append([cmd.DST_START_ADDR, mem_size, cover_size, True, 1, desc])
        # 0:addr 1:size 2:cover_size 3:is_global 4:rw_type(1-w,0-r) 5:desc
    return [MemRecord(*m) for m in mem_info]


def get_bd_mem_record(cmd):
    bd_func = BDFuncType(cmd.TSK_TYP)
    in_num = cmd.TSK_OPD_NUM
    in0_is_const = cmd.OPT_OPD0_CONST
    in1_is_const = cmd.OPT_OPD1_CONST
    in2_is_const = cmd.OPT_OPD2_CONST
    in0_format = BDFormat(cmd.OPT_OPD0_PREC)
    in1_format = BDFormat(cmd.OPT_OPD1_PREC)
    in2_format = BDFormat(cmd.OPT_OPD2_PREC)
    res0_prec = BDFormat(cmd.OPT_RES0_PREC)
    int8_mode = res0_prec == BDFormat.INT8 or res0_prec == BDFormat.INT16
    int32_mode = res0_prec == BDFormat.INT32
    result_add = cmd.OPT_RES_ADD
    mem_list = []
    desc = "{}:id={}".format(bd_func.name, cmd.CMD_ID_TPU)
    if bd_func == BDFuncType.CONV_NEURON:
        input_addr = cmd.OPD0_ADDR
        output_addr = cmd.RES0_ADDR
        weight_addr = cmd.OPD1_ADDR
        bias_addr = cmd.OPD2_ADDR
        input_n = cmd.RES0_N
        input_c = cmd.OPD0_C
        input_h = cmd.OPD0_H
        input_w = cmd.OPD0_W
        output_c = cmd.RES0_C
        output_h = cmd.RES0_H
        output_w = cmd.RES0_W
        kh = cmd.OPD1_H
        kw = cmd.OPD1_W
        dh = cmd.OPD1_Y_INS0 + 1
        dw = cmd.OPD1_X_INS0 + 1
        kernel_stride_enable = cmd.SHORT_OPD1_STR == 3
        input_stride_enable = cmd.SHORT_OPD0_STR == 3
        kernel_stride_enable = cmd.SHORT_OPD1_STR == 3
        bias_stride_enable = cmd.SHORT_OPD1_STR == 3
        pad_h_t = cmd.OPD0_UP_PAD
        pad_h_b =cmd.OPD0_DN_PAD
        pad_w_l =cmd.OPD0_LF_PAD
        pad_w_r =cmd.OPD0_RT_PAD
        stride_w =cmd.RES_OP_X_STR
        stride_h =cmd.RES_OP_Y_STR
        ins_h = cmd.OPD0_Y_INS0
        ins_w = cmd.OPD0_X_INS0
        kernel_flip = cmd.OPT_KERNEL_ROTATE
        with_bias = cmd.TSK_OPD_NUM == 3
        kernel_is_const = cmd.OPT_OPD1_CONST
        input_n_stride = cmd.OPD0_N_STR
        input_c_stride = cmd.OPD0_C_STR
        input_h_stride = cmd.OPD0_H_STR
        input_w_stride = cmd.OPD0_W_STR
        bias_stride_enable = cmd.SHORT_OPD2_STR == 3
        output_stride_enable = cmd.SHORT_RES0_STR == 3
        store_mode = StoreMode.UNKNOWN
        if in0_format == BDFormat.INT8:
            store_mode = StoreMode.INT8_4N
        input_desc = desc+", input0, pad({},{},{},{}), str({},{}), dilate({},{}), ins0({},{})".format(
                      pad_h_b, pad_h_t, pad_w_l, pad_w_r,
                      stride_h, stride_w, dh, dw, ins_h, ins_w)
        m = __get_local_mem_size_info(input_addr, input_n, input_c, input_h, input_w,
                input_n_stride, input_c_stride, input_h_stride, input_w_stride, in0_format,
                True, input_stride_enable, 0, input_desc, store_mode = store_mode)
        mem_list.append(m)

        if not kernel_is_const:
            weight_n_stride = cmd.OPD1_N_STR
            weight_c_stride = cmd.OPD1_C_STR
            weight_h_stride = cmd.OPD1_H_STR
            weight_w_stride = cmd.OPD1_W_STR
            weight_n = input_c
            weight_c = output_c
            weight_h = kh
            weight_w = kw
            weight_desc = desc + ", kernel"
            m = __get_local_mem_size_info(weight_addr, weight_n, weight_c, weight_h, weight_w,
                    weight_n_stride, weight_c_stride, weight_h_stride, weight_w_stride, in1_format,
                    False, kernel_stride_enable, 0, weight_desc, store_mode = store_mode)
            mem_list.append(m)
        if with_bias:
            bias_n = 1
            bias_c = output_c
            bias_h = 1
            bias_w = 1
            bias_desc = desc + ", bias"
            bias_n_stride = cmd.OPD2_N_STR
            bias_c_stride = cmd.OPD2_C_STR
            bias_h_stride = cmd.OPD2_H_STR
            bias_w_stride = cmd.OPD2_W_STR
            bias_format = BDFormat.INT16
            if in0_format == BDFormat.FP32:
                bias_stride_enable = False
                bias_format = in0_format
            m = __get_local_mem_size_info(bias_addr, bias_n, bias_c, bias_h, bias_w,
                    bias_n_stride, bias_c_stride, bias_h_stride, bias_w_stride,
                    bias_format, False, bias_stride_enable, 0, bias_desc)
            mem_list.append(m)
        if in0_format == BDFormat.FP32:
            output_stride_enable = False
            res_format = in0_format
        output_desc = desc + ", output"
        output_n_stride = cmd.RES0_N_STR
        output_c_stride = cmd.RES0_C_STR
        output_h_stride = cmd.RES0_H_STR
        output_w_stride = cmd.RES0_W_STR
        m = __get_local_mem_size_info(output_addr, input_n, output_c, output_h, output_w,
                output_n_stride, output_c_stride, output_h_stride, output_w_stride,
                res0_prec, True, output_stride_enable, 1, output_desc, store_mode=store_mode)
        mem_list.append(m)
    elif bd_func == BDFuncType.DEPTHWISE_OR_POOLING:
        # parse as pooling
        input_addr = cmd.OPD0_ADDR
        output_addr = cmd.RES0_ADDR
        weight_addr = cmd.OPD1_ADDR
        bias_addr = cmd.OPD2_ADDR
        input_n = cmd.RES0_N
        input_c = cmd.RES0_C
        input_h = cmd.OPD0_H
        input_w = cmd.OPD0_W
        output_h = cmd.RES0_H
        output_w = cmd.RES0_W
        kh = cmd.OPD1_H
        kw = cmd.OPD1_W
        pad_h_t = cmd.OPD0_UP_PAD
        pad_h_b =cmd.OPD0_DN_PAD
        pad_w_l =cmd.OPD0_LF_PAD
        pad_w_r =cmd.OPD0_RT_PAD
        stride_h =cmd.RES_OP_Y_STR
        stride_w =cmd.RES_OP_X_STR
        ins_h = cmd.OPD0_Y_INS0
        ins_w = cmd.OPD0_X_INS0
        dh = cmd.OPD1_Y_INS0 + 1
        dw = cmd.OPD1_X_INS0 + 1
        is_avg = cmd.TSK_EU_TYP == 1
        with_bias = is_avg and cmd.TSK_OPD_NUM == 3
        kernel_is_const = cmd.OPT_OPD1_CONST
        in_store_mode = StoreMode.UNKNOWN
        weight_store_mode = StoreMode.UNKNOWN
        bias_store_mode = StoreMode.UNKNOWN
        if in0_format != BDFormat.FP32:
            in_store_mode = StoreMode.INT8_4N
            weight_store_mode = StoreMode.INT8_1N
            bias_store_mode = StoreMode.INT16_1N
        m = __get_local_mem_size_info(input_addr, input_n, input_c, input_h, input_w,
                0, 0, 0, 0, in0_format, True, False, 0, desc, store_mode = in_store_mode)
        mem_list.append(m)
        if is_avg and not kernel_is_const:
            weight_n = 1
            weight_c = input_c
            weight_h = kh
            weight_w = kw
            weight_desc = desc + ", kernel"
            m = __get_local_mem_size_info(weight_addr, weight_n, weight_c, weight_h, weight_w,
                    0, 0, 0, 0, in0_format, False, False, 0, weight_desc, store_mode=weight_store_mode)
            mem_list.append(m)
        if with_bias:
            bias_n = 1
            bias_c = input_c
            bias_h = 1
            bias_w = 1
            bias_desc = desc + ", bias"
            m = __get_local_mem_size_info(bias_addr, bias_n, bias_c, bias_h, bias_w,
                    0, 0, 0, 0, in0_format, False, False, 0, bias_desc, store_mode = bias_store_mode)
        output_desc = desc + ", output"
        m = __get_local_mem_size_info(output_addr, input_n, input_c, output_h, output_w,
                0, 0, 0, 0, res0_prec, True, False, 1, output_desc, result_add=result_add, store_mode=in_store_mode)
        mem_list.append(m)
    elif bd_func == BDFuncType.FC:
        with_bias = cmd.TSK_OPD_NUM == 3
        is_L_trans = cmd.OPT_LEFT_TRAN
        add_result = cmd.OPT_RES_ADD
        R_tensor_C = cmd.RES0_C
        R_tensor_W = cmd.RES0_W
        L_tensor_C = cmd.OPD0_C
        L_tensor_W = cmd.OPD0_W
        R_col_num = R_tensor_C * R_tensor_W
        L_last_W = cmd.OPD1_W
        L_col_num = (L_tensor_C-1) * L_tensor_W + L_last_W
        img_num = cmd.OPD0_N

        if is_L_trans:
            img_num, L_col_num = L_col_num, img_num
        L_tensor_start_addr = cmd.OPD0_ADDR
        R_tensor_start_addr = cmd.OPD1_ADDR
        Y_tensor_start_addr = cmd.RES0_ADDR
        bias_tensor_start_addr = cmd.OPD2_ADDR

        is_L_const = cmd.OPT_OPD0_CONST
        if not is_L_const:
            L_desc = desc + ", LMatrix({},{}), T={}".format(img_num, L_col_num, is_L_trans)
            m = __get_local_mem_size_info(L_tensor_start_addr, cmd.OPD0_N, L_tensor_C, 1, L_tensor_W,
                    0, 0, 0, 0, in0_format, True, False, 0, L_desc)
            mem_list.append(m)

        R_desc = desc + ", RMatrix({},{})".format(L_col_num, R_col_num)
        m = __get_local_mem_size_info(R_tensor_start_addr, L_col_num, R_tensor_C, 1, R_tensor_W,
                0, 0, 0, 0, in0_format, True, False, 0, R_desc)
        mem_list.append(m)
        if with_bias:
            bias_desc = desc + ", bias"
            m = __get_local_mem_size_info(bias_tensor_start_addr, 1, R_col_num, 1, 1,
                    0, 0, 0, 0, in0_format, True, False, 0, bias_desc)
            mem_list.append(m)
        out_desc = desc + ", OMatrix({},{})".format(img_num, R_col_num)
        m = __get_local_mem_size_info(Y_tensor_start_addr, img_num, R_col_num, 1, R_tensor_W,
                0, 0, 0, 0, in0_format, True, False, 1, out_desc, result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.TENSOR_ARITHMETIC:
        tensorA_addr = cmd.OPD0_ADDR
        tensorB_addr = cmd.OPD1_ADDR
        tensorC_addr = cmd.RES0_ADDR

        n = cmd.RES0_N
        c = cmd.RES0_C
        h = cmd.RES0_H
        w = cmd.RES0_W
        b_n_is1 = cmd.OPD1_N_STR
        b_h_is1 = cmd.OPD1_H_STR
        b_w_is1 = cmd.OPD1_W_STR

        b_n = 1 if b_n_is1 else n
        b_c = c
        b_h = 1 if b_h_is1 else h
        b_w = 1 if b_w_is1 else w

        a_is_const = cmd.OPT_OPD0_CONST
        b_is_const = cmd.OPT_OPD1_CONST
        op = cmd.TSK_EU_TYP
        op_names={
            0: "TENSOR_MUL",
            1: "TENSOR_MAC",
            2: "TENSOR_ADD",
            3: "TENSOR_SUB",
            4: "TENSOR_MAX",
            5: "TENSOR_MIN",
            6: "TENSOR_SHIFT",
            10: "TENSOR_SG",
            11: "TENSOR_SE",
            12: "TENSOR_DIV",
            19: "TENSOR_CPY",
            7: "TENSOR_AND",
            8: "TENSOR_OR",
            9: "TENSOR_XOR",
        }
        op_name=op_names[op]

        a_n_stride = cmd.OPD0_N_STR
        a_c_stride = cmd.OPD0_C_STR
        a_h_stride = cmd.OPD0_H_STR
        a_w_stride = cmd.OPD0_W_STR

        b_n_stride = cmd.OPD1_N_STR
        b_c_stride = cmd.OPD1_C_STR
        b_h_stride = cmd.OPD1_H_STR
        b_w_stride = cmd.OPD1_W_STR

        c_n_stride = cmd.RES0_N_STR
        c_c_stride = cmd.RES0_C_STR
        c_h_stride = cmd.RES0_H_STR
        c_w_stride = cmd.RES0_W_STR

        a_stride_enable = cmd.SHORT_OPD0_STR
        b_stride_enable = cmd.SHORT_OPD1_STR
        c_stride_enable = cmd.SHORT_RES0_STR
        format_map = {
            BDFormat.FP32: StoreMode.UNKNOWN,
            BDFormat.INT8: StoreMode.INT8_4N,
            BDFormat.INT16: StoreMode.INT16_2N,
            BDFormat.INT32: StoreMode.UNKNOWN
        }
        a_store_mode = format_map[in0_format]
        b_store_mode = format_map[in1_format]
        c_store_mode = format_map[res0_prec]
        if not a_is_const:
            a_desc = desc+", A"
            m = __get_local_mem_size_info(tensorA_addr, n, c, h, w,
                a_n_stride, a_c_stride, a_h_stride, a_w_stride, in0_format,
                True, a_stride_enable, 0, a_desc, store_mode = a_store_mode)
            mem_list.append(m)
        if not b_is_const and op_name != "TENSOR_CPY":
            b_desc = desc+", B"
            m = __get_local_mem_size_info(tensorB_addr, b_n, b_c, b_h, b_w,
                b_n_stride, b_c_stride, b_h_stride, b_w_stride, in1_format,
                True, b_stride_enable, 0, b_desc, store_mode = b_store_mode)
            mem_list.append(m)
        c_desc = desc+", C, op={}".format(op_name)
        m = __get_local_mem_size_info(tensorC_addr, n, c, h, w,
            c_n_stride, c_c_stride, c_h_stride, c_w_stride, res0_prec,
            True, c_stride_enable, 1, c_desc, result_add=result_add, store_mode = c_store_mode)
        mem_list.append(m)
    elif bd_func == BDFuncType.FC2:
        L_addr = cmd.OPD0_ADDR
        R_addr = cmd.OPD1_ADDR
        RES_addr = cmd.RES0_ADDR
        opd0_sign = cmd.OPT_OPD0_SIGN
        opd1_sign = cmd.OPT_OPD1_SIGN
        op_format = BDFormat.INT8
        m = __get_local_mem_size_info(L_addr, 1, 256, 1, 256,
            0, 0, 0, 0, op_format,
            True, False, 0, desc+", LMatrix, sign=".format(opd0_sign), 4)
        mem_list.append(m)
        m = __get_local_mem_size_info(R_addr, 1, 256, 1, 256,
            0, 0, 0, 0, op_format,
            True, False, 0, desc+", RMatrix, sign=".format(opd1_sign), 4)
        mem_list.append(m)
        m = __get_local_mem_size_info(RES_addr, 1, 256, 1, 256,
            0, 0, 0, 0, op_format, True, False, 1, desc+", result", 4)
        mem_list.append(m)
    elif bd_func == BDFuncType.TABLE_LOOKUP:
        N = cmd.RES0_N
        C = cmd.RES0_C
        H = cmd.RES0_H
        W = cmd.RES0_W
        B_addr = cmd.OPD0_ADDR
        C_addr = cmd.RES0_ADDR
        table_size = cmd.OPD1_H
        table_addr = cmd.OPD1_ADDR
        m = __get_local_mem_size_info(B_addr, N, C, H, W,
                        0, 0, 0, 0, in0_format, True, False, 0, desc + ", index", 4)
        mem_list.append(m)
        m = [table_addr, table_size*4, table_size*4, True, 0, desc+ ", table_size={}".format(table_size)]
        mem_list.append(m)
        m = __get_local_mem_size_info(C_addr, N, C, H, W,
                        0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", result", 4, result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.MD_SUM:
        A_addr = cmd.OPD0_ADDR
        Y_addr = cmd.RES0_ADDR
        H = cmd.OPD0_H
        W = cmd.OPD0_W
        C = cmd.RES0_C
        N = cmd.OPD0_N
        m = __get_local_mem_size_info(A_addr, N, C, H, W,
                        0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc + ", input")
        mem_list.append(m)
        m = __get_local_mem_size_info(Y_addr, 1, C, 1, 1,
                        0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc + ", output", result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.MD_SCALAR:
        A_addr = cmd.OPD0_ADDR
        B_addr = cmd.OPD1_ADDR
        R_addr = cmd.RES0_ADDR
        H = cmd.RES0_H
        W = cmd.RES0_W
        C = cmd.RES0_C
        N = cmd.RES0_N
        op = cmd.TSK_EU_TYP
        A_is_constant = cmd.OPT_OPD0_CONST
        B_is_constant = cmd.OPT_OPD1_CONST
        op_format = BDFormat.FP32
        op_names = {
            2: "ALIGN_TENSOR_ADD",
            3: "ALIGN_TENSOR_SUB",
            0: "ALIGN_TENSOR_MUL",
            12: "ALIGN_TENSOR_DIV",
        }
        op_name = op_names[op]
        if not A_is_constant:
            m = __get_local_mem_size_info(A_addr, N, C, H, W,
                0, 0, 0, 0, op_format, True, False, 0, desc+", A")
            mem_list.append(m)
        if not B_is_constant:
            m = __get_local_mem_size_info(B_addr, N, C, H, W,
                0, 0, 0, 0, op_format, True, False, 0, desc+", B")
            mem_list.append(m)
        m = __get_local_mem_size_info(R_addr, N, C, H, W,
            0, 0, 0, 0, op_format, True, False, 1, desc+", result", result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.MD_SFU:
        A_addr = cmd.OPD0_ADDR
        Y_addr = cmd.RES0_ADDR
        H = cmd.RES0_H
        W = cmd.RES0_W
        C = cmd.RES0_C
        N = cmd.RES0_N
        taylor_num = cmd.OPD1_N
        taylor_addr = cmd.OPD1_ADDR
        op = cmd.TSK_EU_TYP
        op_names =  {
            13: "TAYLOR",
            14: "SFU_F2I",
            15: "SFU_NORMB",
            16: "SFU_NORMA",
            17: "SFU_RSQ",
        }
        op_name = op_names[op]
        m = __get_local_mem_size_info(A_addr, N, C, H, W,
                        0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc + ", input", 4)
        mem_list.append(m)
        if op == 13:
            m = [taylor_addr, taylor_num*4, taylor_num*4, True, 0, desc+ ", taylor_num={}".format(taylor_num)]
            mem_list.append(m)
        m = __get_local_mem_size_info(Y_addr, N, C, H, W,
                        0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", result, op={}".format(op_name), result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.MD_LINEAR:
        op = cmd.TSK_EU_TYP
        S_is_const = cmd.OPT_OPD2_CONST
        B_is_const = cmd.OPT_OPD1_CONST
        op_name = "MAC" if op == 1 else "POWER_ADD"
        H = cmd.RES0_H
        W = cmd.RES0_W
        C = cmd.RES0_C
        N = cmd.RES0_N

        A_addr = cmd.OPD0_ADDR
        B_addr = cmd.OPD1_ADDR
        S_addr = cmd.OPD2_ADDR
        R_addr = cmd.RES0_ADDR
        a_desc = desc+", A"
        op_format = BDFormat.FP32
        m = __get_local_mem_size_info(A_addr, N, C, H, W,
            0, 0, 0, 0, op_format, True, False, 0, a_desc)
        mem_list.append(m)
        if not S_is_const:
            m = __get_local_mem_size_info(S_addr, 1, C, 1, 1,
                0, 0, 0, 0, op_format, False, False, 0, desc+", S")
            mem_list.append(m)
        if not B_is_const:
            m = __get_local_mem_size_info(B_addr, 1, C, 1, 1,
                0, 0, 0, 0, op_format, False, False, 0, desc+", B")
            mem_list.append(m)
        r_desc = desc+", result, op_name={}".format(op_name)
        m = __get_local_mem_size_info(A_addr, N, C, H, W,
            0, 0, 0, 0, op_format, True, False, 1, r_desc, result_add=result_add)
        mem_list.append(m)
    elif bd_func == BDFuncType.MD_CMP:
        A_addr = cmd.OPD0_ADDR
        B_addr = cmd.OPD1_ADDR
        C_addr = cmd.OPD2_ADDR
        D_addr = cmd.OPD3_ADDR
        Y_addr = cmd.RES0_ADDR
        R_addr = cmd.RES1_ADDR

        N = cmd.RES0_N
        C = cmd.RES0_C
        H = cmd.RES0_H
        W = cmd.RES0_W

        b_n_is1 = cmd.OPD0_N_STR == 0
        b_h_is1 = cmd.OPD0_H_STR == 0
        b_w_is1 = cmd.OPD0_W_STR == 0
        BN = 1 if b_n_is1 else N
        BC = C
        BH = 1 if b_h_is1 else H
        BW = 1 if b_w_is1 else W
        a_is_const = cmd.OPT_OPD0_CONST
        b_is_const = cmd.OPT_OPD1_CONST
        c_is_const = cmd.OPT_OPD2_CONST
        d_is_const = cmd.OPT_OPD3_CONST
        eu_type = cmd.TSK_EU_TYP

        if not a_is_const:
            m = __get_local_mem_size_info(A_addr, N, C, H, W,
                0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc+", A")
            mem_list.append(m)
        if not b_is_const:
            m = __get_local_mem_size_info(B_addr, BN, BC, BH, BW,
                0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc+", B")
            mem_list.append(m)
        if not c_is_const:
            m = __get_local_mem_size_info(C_addr, BN, BC, BH, BW,
                0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc+", C")
            mem_list.append(m)
        if not d_is_const:
            m = __get_local_mem_size_info(D_addr, BN, BC, BH, BW,
                0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc+", D")
            mem_list.append(m)
        if eu_type == 24:
            m = __get_local_mem_size_info(Y_addr, N, C, H, W,
                0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", Y")
            mem_list.append(m)
            m = __get_local_mem_size_info(R_addr, N, C, H, W,
                0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", R")
            mem_list.append(m)
        elif eu_type == 22:
            m = __get_local_mem_size_info(Y_addr, N, C, H, W,
                0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", Y")
            mem_list.append(m)
        else:
            m = __get_local_mem_size_info(R_addr, N, C, H, W,
                0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc+", R")
            mem_list.append(m)
    elif bd_func == BDFuncType.CONV_CORRELATION:
        input_addr = cmd.OPD0_ADDR
        output_addr = cmd.RES0_ADDR
        weight_addr = cmd.OPD1_ADDR
        pad_h_t = cmd.OPD0_UP_PAD
        pad_h_b = cmd.OPD0_DN_PAD
        pad_w_l = cmd.OPD0_LF_PAD
        pad_w_r = cmd.OPD0_RT_PAD
        stride_w = cmd.RES_OP_X_STR
        stride_h = cmd.RES_OP_Y_STR
        ic = cmd.RES0_N
        oc = cmd.RES0_C
        oh = cmd.RES0_H
        ow = cmd.RES0_W
        ih = cmd.OPD0_H
        iw = cmd.OPD0_W
        kh = cmd.OPD1_H
        kw = cmd.OPD1_W
        op_format = BDFormat.FP32
        m = __get_local_mem_size_info(input_addr, 1, ic, ih, iw,
            0, 0, 0, 0, op_format, True, False, 0, desc + ", input")
        mem_list.append(m)
        m = __get_local_mem_size_info(weight_addr, 1, oc, kh, kw,
            0, 0, 0, 0, op_format, True, False, 0, desc + ", weight")
        mem_list.append(m)
        m = __get_local_mem_size_info(output_addr, ic, oc, oh, ow,
            0, 0, 0, 0, op_format, False, False, 1, desc + ", output, pad=({},{},{},{}), conv_stride=({},{})".format(
                pad_h_t, pad_h_t, pad_w_l, pad_w_r, stride_h, stride_w))
        mem_list.append(m)
    elif bd_func == BDFuncType.VECTOR_CORRELATION:
        R_tensor_C = cmd.RES0_C
        R_tensor_W = cmd.RES0_W
        L_tensor_C = cmd.OPD0_C
        L_tensor_W = cmd.OPD0_W
        R_col_num = R_tensor_C * R_tensor_W
        L_last_W = cmd.OPD1_W
        L_col_num = (L_tensor_C - 1) * L_tensor_W + L_last_W

        L_tensor_start_addr = cmd.OPD0_ADDR
        R_tensor_start_addr = cmd.OPD1_ADDR
        Y_tensor_start_addr = cmd.RES0_ADDR
        op = cmd.TSK_EU_TYP
        op_names = {
            2:  "ADD",
            3:  "SUB",
            0:  "MUL",
            12: "DIV",
            4:  "MAX",
            5:  "MIN",
            7:  "AND",
            8:  "OR",
            9:  "XOR",
        }
        op_format = res0_prec
        op_name = op_names[op]
        m = __get_local_mem_size_info(L_tensor_start_addr, 1, L_tensor_C, 1, L_tensor_W,
            0, 0, 0, 0, op_format, True, False, 0, desc + ", LMatrix({},{})".format(1, L_col_num))
        mem_list.append(m)
        m = __get_local_mem_size_info(R_tensor_start_addr, 1, R_tensor_C, 1, R_tensor_W,
            0, 0, 0, 0, op_format, True, False, 0, desc + ", RMatrix({},{})".format(1, R_col_num))
        mem_list.append(m)
        m = __get_local_mem_size_info(Y_tensor_start_addr, L_col_num, R_tensor_C, 1, R_tensor_W,
            0, 0, 0, 0, op_format, True, False, 1, desc + ", op={}, OMatrix({},{})".format(op_name, L_col_num, R_col_num))
        mem_list.append(m)

    elif bd_func == BDFuncType.DECOMPRESS:
        I_addr = cmd.OPD0_ADDR
        C_addr = cmd.OPD1_ADDR
        D_addr = cmd.RES0_ADDR
        op_format = cmd.OPT_RES0_PREC
        N = 1
        C = NPU_NUM
        H = 1
        W = cmd.OPD0_W
        m = __get_local_mem_size_info(I_addr, N, C, H, W,
            0, 0, 0, 0, op_format, True, False, 0, desc + ", Compressed", 4)
        mem_list.append(m)
        m = __get_local_mem_size_info(C_addr, N, C, H, W,
            0, 0, 0, 0, BDFormat.INT16, True, False, 0, desc + ", Coeff", 4)
        mem_list.append(m)
        m = __get_local_mem_size_info(D_addr, N, C, H, W,
            0, 0, 0, 0, op_format, True, False, 1, desc + ", Decompressed", 4)
        mem_list.append(m)
    elif bd_func == BDFuncType.LOCALMEM_ARANGE:
        input_addr = cmd.OPD0_ADDR
        output_addr = cmd.RES0_ADDR
        table_addr = cmd.OPD1_ADDR
        table_entry_cnt = cmd.OPD1_N
        table_size = table_entry_cnt * 64
        N=1
        C=NPU_NUM
        H=1
        W=EU_NUM
        m = [table_addr, table_size, table_size, True, 0, desc+ ", index_table, entry_cnt={}".format(table_entry_cnt)]
        mem_list.append(m)
        m = __get_local_mem_size_info(input_addr, N, C, H, W,
            0, 0, 0, 0, BDFormat.FP32, True, False, 0, desc, 4)
        mem_list.append(m)
        m = __get_local_mem_size_info(output_addr, N, C, H, W,
            0, 0, 0, 0, BDFormat.FP32, True, False, 1, desc, 4)
        mem_list.append(m)
    return [MemRecord(*m) for m in mem_list]

class CMDWrapper():
    _cmd_id_ = 0
    def __init__(self):
        self.ref_id= self.__class__._cmd_id_
        self.__class__._cmd_id_ += 1
    def test(self, mark_condition):
        args = {"gdma": False, "bd": True}
        class MStruct:
            pass
        for key, value in self.__dict__.items():
            if key == "mem_records":
                v_list = []
                for m in value:
                    v=MStruct()
                    v.is_out =m.is_out
                    v.is_in = not m.is_out
                    v.gmem =m.is_global
                    v.lmem = not v.is_global
                    v.start_addr = m.addr
                    v.end_addr = m.addr + m.cover_size
                    v.start_offset = m.addr
                    v.end_offset = m.addr + m.cover_size
                    v.data_size = m.data_size
                    v.cover_size = m.cover_size
                    v.desc = m.desc
                    v_list.append(v)
                args["m"] = v_list
                continue
            if key == "DIRECTION":
                args['gdma'] = True
                args['bd'] = True
            args[key] = value
        return eval(mark_condition, args)

    def __str__(self):
        info=[]
        reg_info=[]
        info.append("Registers:")
        for key, value in self.__dict__.items():
            if key == "mem_records":
                continue
            reg_info.append("  {}={}".format(key,value))
        reg_info.sort()
        info += reg_info
        info.append("MemRecords:")
        for m in self.mem_records:
            mem_str = "  "
            if m.is_out:
                mem_str += "out, "
            else:
                mem_str += "in,  "
            if m.is_global:
                mem_str += "gmem, "
                mem_str += "start_addr=%d[0x%x], "%(m.addr, m.addr)
                mem_str += "end_addr=%d[0x%x], "%(m.addr+m.cover_size, m.addr+m.cover_size)
            else:
                mem_str += "lmem, "
                mem_str += "start_offset=%d[0x%x], "%(m.addr, m.addr)
                mem_str += "end_offset=%d[0x%x], "%(m.addr+m.cover_size, m.addr+m.cover_size)
            mem_str += "data_size=%d, "%(m.data_size)
            mem_str += "cover_size=%d, "%(m.cover_size)
            mem_str += m.desc
            info.append(mem_str)
        return "\n".join(info)

class BaseCommandParser:
    def __init__(self):
        self.__attr_names = [name for name, _, _ in self._desc_]
        last_desc = self._desc_[0]
        for d in self._desc_:
            if last_desc[1]<d[1]:
                last_desc = d
    def parse(self, buf, max_num):
        desc = self._desc_
        cmd_list = []
        for i in range(max_num):
            cmd = CMDWrapper()
            cmd.mlir_cmd = None
            for d in desc:
                name, bit_begin, bit_len = d
                byte_begin = bit_begin//8
                byte_end = (bit_begin + bit_len+7)//8
                bit_offset = bit_begin%8
                byte_slice=buf[byte_begin:byte_end]
                total_val = 0
                for v in byte_slice[::-1]:
                    total_val = (total_val<<8 | v)
                total_val = total_val>>bit_offset
                total_val = (total_val & ((1<<bit_len)-1))
                cmd.__dict__[name] = total_val
            cmd.type = self.__class__._type_func_(cmd)
            cmd.mem_records = self.__class__._mem_record_func_(cmd)
            cmd.dep_id = -1
            cmd_list.append(cmd)
            buf = buf[self.command_byte_len():]
            cmd.alg_ops = 0
            cmd.arch_ops = 0
        return cmd_list

    def attr_names(self):
        return self.__attr_names

    def command_byte_len(self):
        return self._byte_len_

class GDMACommandParser(BaseCommandParser):
    _mem_record_func_ = get_gdma_mem_record
    _byte_len_ = 128
    _type_func_ = lambda d: GDMAFuncType(d.__dict__['SPECIAL_FUNC']).name
    _desc_ = [
        ("PIO_GDMA_ENABLE", 128 - 128, 1),
        ("DES_TYPE", 129 - 128, 1),
        ("CHAIN_END", 130 - 128, 1),
        ("INTR_EN", 131 - 128, 1),
        ("BARRIER_ENABLE", 132 - 128, 1),
        ("STRIDE_ENABLE", 133 - 128, 1),
        ("DIRECTION", 134 - 128, 2),
        ("ACC_WRITE_ENABLE", 136 - 128, 1),
        ("COMMON_MODE", 137 - 128, 1),
        ("PREFETCH_DISABLE", 138 - 128, 1),
        ("HOLD_DES_VALUE", 139 - 128, 1),
        ("CMD_ID", 144 - 128, 16),
        ("SPECIAL_FUNC", 160 - 128, 3),
        ("DST_DATA_FORMAT", 163 - 128, 3),
        ("CHW_COPY", 166 - 128, 1),
        ("SYS_MEM_TYPE", 167 - 128, 1),
        ("SRC_DATA_FORMAT", 168 - 128, 3),
        ("LRN_SHIFT_NUM", 171 - 128, 4),
        ("LRN_SHIFT_DIR", 175 - 128, 1),
        ("ENG0_SYNC_ID", 176 - 128, 16),
        ("ENG1_SYNC_ID", 192 - 128, 16),
        ("ENG3_SYNC_ID", 208 - 128, 16),
        ("CONSTANT_VALUE", 224 - 128, 32),
        ("SRC_NSTRIDE", 256 - 128, 32),
        ("SRC_CSTRIDE", 288 - 128, 32),
        ("SRC_HSTRIDE", 320 - 128, 32),
        ("SRC_WSTRIDE", 352 - 128, 32),
        ("DST_NSTRIDE", 384 - 128, 32),
        ("DST_CSTRIDE", 416 - 128, 32),
        ("DST_HSTRIDE", 448 - 128, 32),
        ("DST_WSTRIDE", 480 - 128, 32),
        ("SRC_NSIZE", 512 - 128, 16),
        ("SRC_CSIZE", 528 - 128, 16),
        ("SRC_HSIZE", 544 - 128, 16),
        ("SRC_WSIZE", 560 - 128, 16),
        ("DST_NSIZE", 576 - 128, 16),
        ("DST_CSIZE", 592 - 128, 16),
        ("DST_HSIZE", 608 - 128, 16),
        ("DST_WSIZE", 624 - 128, 16),
        ("SRC_START_ADDR_L32", 640 - 128, 32),
        ("DST_START_ADDR_L32", 672 - 128, 32),
        ("SRC_START_ADDR_H8", 704 - 128, 8),
        ("DST_START_ADDR_H8", 712 - 128, 8),
        ("SRC_HSHIFT", 736 - 128, 8),
        ("SRC_WSHIFT", 744 - 128, 8),
        ("DST_HSHIFT", 752 - 128, 8),
        ("DST_WSHIFT", 760 - 128, 8),
        ("LOCALMEM_MASK_L32", 768 - 128, 32),
        ("LOCALMEM_MASK_H32", 800 - 128, 32),
        ("SINGLE_STEP", 832 - 128, 1),
        ("DEBUG_MODE", 833 - 128, 1),
    ]

class BDCommandParser(BaseCommandParser):
    _mem_record_func_ = get_bd_mem_record
    _byte_len_ = 128
    _type_func_ = lambda d: BDFuncType(d.__dict__['TSK_TYP']).name
    _desc_ = [
        ("CMD_EN", 0, 1),
        ("CMD_END", 1, 1),
        ("CMD_ID_EN", 2, 1),
        ("CMD_ID_TPU", 3, 16),
        ("CMD_ID_GDMA", 19, 16),
        ("CMD_KEEP", 35, 1),
        ("CMD_INTR_EN", 36, 1),
        ("TSK_TYP", 37, 4),
        ("TSK_EU_TYP", 41, 5),
        ("TSK_OPD_NUM", 46, 2),
        ("OPT_RIGHT_SHIFT", 48, 5),
        ("OPT_LEFT_SHIFT", 53, 5),
        ("OPT_SHIFT_TYP", 58, 1),
        ("OPT_RES_ADD", 59, 1),
        ("OPT_RELU", 60, 1),
        ("OPT_LEFT_TRAN", 61, 1),
        ("OPT_WINOGRAD", 62, 1),
        ("OPT_KERNEL_ROTATE", 63, 1),
        ("OPT_OPD0_SIGN", 64, 1),
        ("OPT_OPD1_SIGN", 65, 1),
        ("OPT_OPD2_SIGN", 66, 1),
        ("OPT_RES0_PREC", 67, 3),
        ("OPT_OPD0_PREC", 70, 3),
        ("OPT_OPD1_PREC", 73, 3),
        ("OPT_OPD2_PREC", 76, 3),
        ("OPT_OPD0_CONST", 79, 1),
        ("OPT_OPD1_CONST", 80, 1),
        ("OPT_OPD2_CONST", 81, 1),
        ("SHORT_RES0_STR", 82, 3),
        ("SHORT_OPD0_STR", 85, 3),
        ("SHORT_OPD1_STR", 88, 3),
        ("SHORT_OPD2_STR", 91, 3),
        ("OPD0_X_INS0", 94, 4),
        ("OPD0_Y_INS0", 98, 4),
        ("OPD1_X_INS0", 102, 4),
        ("OPD1_Y_INS0", 106, 4),
        ("OPD0_UP_PAD", 110, 4),
        ("OPD0_DN_PAD", 114, 4),
        ("OPD0_LF_PAD", 118, 4),
        ("OPD0_RT_PAD", 122, 4),
        ("RES_OP_X_STR", 126, 4),
        ("RES_OP_Y_STR", 130, 4),
        ("TSK_LANE_NUM_0", 134, 32),
        ("TSK_LANE_NUM_1", 166, 32),
        ("RES0_N", 198, 16),
        ("RES0_C", 214, 12),
        ("RES0_H", 226, 16),
        ("RES0_W", 242, 16),
        ("OPD0_N", 258, 16),
        ("OPD0_C", 274, 12),
        ("OPD0_H", 286, 16),
        ("OPD0_W", 302, 16),
        ("OPD1_N", 318, 12),
        ("OPD1_C", 330, 12),
        ("OPD1_H", 342, 16),
        ("OPD1_W", 358, 16),
        ("RES0_H_SHIFT", 374, 4),
        ("RES0_W_SHIFT", 378, 4),
        ("OPD0_H_SHIFT", 382, 4),
        ("OPD0_W_SHIFT", 386, 4),
        ("OPD1_H_SHIFT", 390, 4),
        ("OPD1_W_SHIFT", 394, 4),
        ("RES0_N_STR", 398, 19),
        ("RES0_C_STR", 417, 19),
        ("OPD0_N_STR", 436, 19),
        ("OPD0_C_STR", 455, 19),
        ("OPD1_N_STR", 474, 19),
        ("OPD1_C_STR", 493, 19),
        ("OPD2_N_STR", 512, 19),
        ("OPD2_C_STR", 531, 19),
        ("RES_ADD_SIGN", 550, 1),
        ("OPD1_NEQ1", 552, 1),
        ("OPT_OPD3_CONST", 553, 1),
        ("RES0_ADDR", 576, 32),
        ("OPD0_ADDR", 608, 32),
        ("OPD1_ADDR", 640, 32),
        ("OPD2_ADDR", 672, 32),
        ("RES0_H_STR", 704, 32),
        ("RES0_W_STR", 736, 32),
        ("OPD0_H_STR", 768, 32),
        ("OPD0_W_STR", 800, 32),
        ("OPD1_H_STR", 832, 32),
        ("OPD1_W_STR", 864, 32),
        ("OPD2_H_STR", 896, 32),
        ("OPD2_W_STR", 928, 32),
        ("RES1_ADDR", 960, 32),
        ("OPD3_ADDR", 992, 32),
    ]
    def parse(self, cmd_buf, max_num):
        new_buf = bytes()
        for i in range(max_num):
            buf = cmd_buf[i*self._byte_len_: (i+1)*self._byte_len_]
            buf_len = self._byte_len_
            reversed_buf = buf[buf_len-4: buf_len]
            word_len = buf_len//4
            for i in range(word_len-1):
                reversed_buf += buf[(word_len-2-i)*4: (word_len-1-i)*4]
            new_buf += reversed_buf
        return super().parse(new_buf, max_num)

def get_node_set_info(dyn_data):
    raw_id = dyn_data.id
    engine = EngineType((raw_id>>48)&0xF)
    gdma_cmd_id = (raw_id>>16) & 0x0FFFF
    bd_cmd_id = raw_id & 0x0FFFF
    return EngineType((raw_id>>48)&0xF), gdma_cmd_id, bd_cmd_id

def parse_raw_id(dyn_type, raw_id, begin_usec, end_usec, current_func,
                 bd_set_data, gdma_set_data, layer_id, layer_type):

    func_types = ["global", "local", "gdma"]
    func_type = ""
    height = -1
    info = []
    bd_func = "UNKNOWN"

    if dyn_type == DynRecordType.FUNC:
        group_id = (raw_id >> 48) & 0x0FFFF
        base_func_type = func_types[(raw_id >>32) & 0x0FFFF]
        info.append("group_id={}".format(group_id))
        info.append("func_type={}".format(base_func_type))
        type_value = raw_id & 0x0FFFF
        cast_type = FWLayerType if base_func_type != "gdma" else FWGDMAType
        func_type = "gdma" if base_func_type == "gdma" else base_func_type+":"+layer_type
        layer_type = enum_cast(type_value, cast_type).name
        height = 0.5
        if base_func_type != "gdma":
            info.append("layer_type={}".format(layer_type))
            height = 1
        layer_id = group_id # use group_id as layer_id
    elif dyn_type == DynRecordType.NODE_SET:
        engine = EngineType((raw_id>>48)&0xF)
        state = "parallel" if bool((raw_id>>32) & 0x1) else "serial"
        gdma_cmd_id = (raw_id>>16) & 0x0FFFF
        bd_cmd_id = raw_id & 0x0FFFF
        info.append("gdma_id={}".format(gdma_cmd_id))
        info.append("bd_id={}".format(bd_cmd_id))
        info.append(state)
        version = (raw_id >> 60)&0xF
        if current_func is not None and begin_usec>=current_func.begin_usec and end_usec <= current_func.end_usec:
            layer_id = current_func.layer_id
            layer_type = current_func.layer_type
        if engine == EngineType.BD:
            if version > 0:
                bd_func = enum_cast((raw_id>>36)&0xF, BDFuncType)
                bd_op = enum_cast((raw_id>>40)&0x1F, BDEUType)
                info.append(bd_func.name)
                info.append(bd_op.name)
            bd_set_data.append([bd_cmd_id, begin_usec, layer_id, layer_type, bd_func])
        if engine == EngineType.GDMA:
            gdma_set_data.append([gdma_cmd_id, begin_usec, layer_id, layer_type])
            if version > 0:
                gdma_dir = enum_cast(((raw_id>>36)&0x3), GDMADirection)
                gdma_func = enum_cast(((raw_id>>40)&0x7), GDMAFuncType)
                info.append(gdma_func.name)
                info.append(gdma_dir.name)
        func_type = "set_{}".format(engine.name)
        end_usec = begin_usec  # just mark time point
        begin_usec = begin_usec - 1
    elif dyn_type == DynRecordType.NODE_WAIT:
        gdma_cmd_id = (raw_id>>16) & 0x0FFFF
        bd_cmd_id = raw_id & 0x0FFFF
        info.append("gdma_id={}".format(gdma_cmd_id))
        info.append("bd_id={}".format(bd_cmd_id))
        func_type = "wait"
    elif dyn_type == DynRecordType.CDMA:
        cdma_type = (raw_id>>56)&0x1
        if cdma_type == 0:
            func_type = "CDMA:general_move"
            info.append("len={}".format(raw_id&0x0FFFFFFFF))
        elif cdma_type == 1:
            func_type = "CDMA:sort_move"
            info.append("len={}".format(raw_id&0x0FFFFFFFF))
            info.append("sort_cnt={}".format((raw_id>>32)&0x0FFFF))
            info.append("order={}".format((raw_id>>50)&0x1))
            info.append("auto_index={}".format((raw_id>>49)&0x1))
            info.append("index_valid={}".format((raw_id>>48)&0x1))
    elif dyn_type == DynRecordType.GDE:
        func_type = "GDE:gather"
        info.append("len={}".format(raw_id&0x0FFFFFFFF))
    elif dyn_type == DynRecordType.CUSTOM:
        height = 0.5
        func_type = "CUSTOM"
        info.append("data=%d[0x%x]"%(raw_id, raw_id))

    return func_type, height, layer_id, layer_type, info, bd_set_data, gdma_set_data

def show_arch_info():
    print("BM1684 bdc_freq={} gdma_freq={}".format(1/BDCyclePeriod, 1/GDMACyclePeriod))
