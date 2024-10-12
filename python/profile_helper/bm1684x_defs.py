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
from typing import NamedTuple
import numpy as np

from bmprofile_common import FWGDMAType, FWLayerType, dictStructure
from bmprofile_utils import enum_cast

arch_name = "BM1684X"
ID_WIDTH = 20
EU_NUM = 16
NPU_NUM = 64
LOCAL_MEM_SHIFT=18
LOCAL_MEM_SIZE=1<<LOCAL_MEM_SHIFT #256k
LOCAL_BANK_NUM=16
TOTAL_LOCAL_MEM_SIZE = LOCAL_MEM_SIZE * NPU_NUM
LOCAL_MEM_ADDR_MASK = TOTAL_LOCAL_MEM_SIZE -1
LOCAL_BANK_SIZE=LOCAL_MEM_SIZE//LOCAL_BANK_NUM

class EngineType(Enum):
    BD = 0
    GDMA = 1
    GDE = 2
    SORT = 3
    NMS = 4
    CDMA = 5
    UNKNOWN = -1

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
    UNKNOWN=-1
    INT8 = 0
    FLOAT16 = 1
    FLOAT32 = 2
    INT16 = 3
    INT32 = 4
    BFLOAT16 = 5

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


GDMACyclePeriod= 1.0/1000
BDCyclePeriod= 1.0/1000
PeriodFixed = False

class MemType(Enum):
    GLOBAL = 0
    STATIC = 1
    L2MEM = 2
    LOCAL = 3
    UNKNOWN = -1

class BDFormat(Enum):
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT16 = 3
    INT32 = 4
    BFP16 = 5
    UNKNOWN = -1

def local_mem_partition():
    partitions = []
    for i in range(LOCAL_BANK_NUM):
        partitions.append([i*LOCAL_BANK_SIZE, LOCAL_BANK_SIZE, "BANK[{}]".format(i)])
    return partitions

def get_node_set_info(dyn_data):
    raw_id = dyn_data.id
    engine = EngineType((raw_id >> 40) & 0x7)
    gdma_cmd_id = (raw_id >> 20) & 0xFFFFF
    bd_cmd_id = raw_id & 0xFFFFF
    return engine, gdma_cmd_id, bd_cmd_id

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
        engine = EngineType((raw_id >> 40) & 0x7)
        state = "parallel" if bool((raw_id >> 52) & 0x1) else "serial"
        gdma_cmd_id = (raw_id >> 20) & 0xFFFFF
        bd_cmd_id = raw_id & 0xFFFFF
        info.append("gdma_id={}".format(gdma_cmd_id))
        info.append("bd_id={}".format(bd_cmd_id))
        info.append(state)
        version = (raw_id >> 60) & 0xF
        if current_func is not None and begin_usec>=current_func.begin_usec and end_usec <= current_func.end_usec:
            layer_id = current_func.layer_id
            layer_type = current_func.layer_type
        if engine == EngineType.BD:
            #bd_set_data.append([bd_cmd_id, begin_usec, layer_id, layer_type])
            if version > 0:
                bd_func = enum_cast((raw_id >> 43) & 0xF, BDFuncType)
                #bd_op = enum_cast((raw_id >> 47) & 0x1F, bd_func.name)
                info.append(bd_func.name)
                #info.append(bd_op.name)
            bd_set_data.append([bd_cmd_id, begin_usec, layer_id, layer_type, bd_func])
        if engine == EngineType.GDMA:
            gdma_set_data.append([gdma_cmd_id, begin_usec, layer_id, layer_type])
            if version > 0:
                gdma_func = enum_cast(((raw_id >> 43) & 0xF), GDMAFuncType)
                info.append(gdma_func.name)
        func_type = "set_{}".format(engine.name)
        end_usec = begin_usec  # just mark time point
        begin_usec = begin_usec - 1
    elif dyn_type == DynRecordType.NODE_WAIT:
        gdma_cmd_id = (raw_id >> 20) & 0xFFFFF
        bd_cmd_id = raw_id & 0xFFFFF
        info.append("gdma_id={}".format(gdma_cmd_id))
        info.append("bd_id={}".format(bd_cmd_id))
        func_type = "wait"
    elif dyn_type == DynRecordType.GDE:
        func_type = "GDE"
        info.append("data=%d[0x%x]"%(raw_id, raw_id))
    elif dyn_type == DynRecordType.SORT:
        func_type = "SORT"
        info.append("data=%d[0x%x]"%(raw_id, raw_id))
    elif dyn_type == DynRecordType.NMS:
        func_type = "NMS"
        info.append("data=%d[0x%x]"%(raw_id, raw_id))
    elif dyn_type == DynRecordType.CUSTOM:
        height = 0.5
        func_type = "CUSTOM"
        info.append("data=%d[0x%x]"%(raw_id, raw_id))

    return func_type, height, layer_id, layer_type, info, bd_set_data, gdma_set_data

def show_arch_info():
    print("BM1684X")


import opdef_1684x, opparam_1684x
from op_support import Layout
MemRecord=namedtuple("MemRecord", "addr data_size cover_size is_global is_out desc")

class CMDWrapper():
    _cmd_id_ = 0
    def __init__(self):
        self.ref_id= self.__class__._cmd_id_
        self.__class__._cmd_id_ += 1
        self.__attr_names = []
    def set_attr_names(self, names):
        self.__attr_names = names

    def attr_names(self):
        return self.__attr_names
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
        info = []
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
    def __get_mem_records(self, cmd):
        import copy

        def _get_size(shape, stride):
            return sum(((x - 1) * y for x, y in zip(shape, stride))) + 1

        def get_size(memref):
            num = 1
            if memref.stride:
                num = _get_size(memref.shape, memref.stride)
            else:
                for dim in memref.shape:
                    num *= dim
            return memref.itemsize * num

        def get_address(memref):
            if memref.mtype == opparam_1684x.MType.R:
                return memref.mtype.r_addr%LOCAL_MEM_SIZE
            return memref.address

        def get_info(memref):
            address = get_address(memref)
            size = get_size(memref)
            is_global = memref.mtype != opparam_1684x.MType.R
            if is_global:
                cover_size = get_size(memref)
                shape = memref.shape
                stride = memref.stride
            else:
                shape, stride = memref.local_shape, memref.local_stride
                if len(shape)==2 and len(stride)==4:
                    shape = [1, shape[0], 1, shape[1]]
                cover_size = _get_size(shape, stride) * memref.itemsize
            return (address, size, cover_size, is_global, shape, stride)

        def is_memref(memref):
            return isinstance(memref, opparam_1684x.MemRef)
        def get_desc(shape, stride, cmd, memref):
            shape = memref.shape
            cmd_type = "gdma" if isinstance(cmd, opdef_1684x.dma_base) else "tiu"
            desc = f"{cmd_type}:id={cmd.cmd_id-1}, shape={'x'.join([str(s) for s in shape])},"
            desc += f"lstride={'x'.join([str(s) for s in stride] if stride is not None else 'None')},"
            desc += f"layout={'Tensor' if memref.layout is None else memref.layout.name}"
            return desc

        # 0:addr 1:size 2:cover_size 3:is_global 4:rw_type(1-w,0-r) 5:desc
        mem_info = []
        for opd in cmd.operands:
            if is_memref(opd):
                info = get_info(opd)
                mem_info.append((*info[0:4], 0, get_desc(*info[-2:], cmd, opd)))
        for ret in cmd.results:
            if is_memref(ret):
                info = get_info(ret)
                mem_info.append((*info[0:4], 1, get_desc(*info[-2:], cmd, ret)))
        return [MemRecord(*m) for m in mem_info]


    def __decode_single(self, cmd_buf, cmd_bits, cmd_set):
        cmd = None
        l, h = cmd_bits
        cmd_key = opdef_1684x.packbits(cmd_buf[l:h])

        for op in cmd_set.get(cmd_key,[]):
            if op.is_comp(cmd_buf):
                # check whether this command is recognized by the operation
                cmd = op.decode(cmd_buf)
                break
        assert cmd, "Can not decode cmd, with opcode: {}.".format(cmd_key)
        full_command = CMDWrapper()
        full_command.mlir_cmd = cmd
        full_command.type = cmd.op_name
        full_command.dep_id = cmd.cmd_id_dep
        full_command.mem_records = self.__get_mem_records(cmd)
        full_command.len = cmd.length
        fields = list(cmd.reg.keys())
        full_command.set_attr_names(fields)
        full_command.__dict__.update(cmd.reg)
        full_command.alg_ops = cmd.alg_ops()
        full_command.arch_ops = cmd.arch_ops()
        return full_command

    def decode(self, cmd_buf, cmd_bits, cmd_set, max_num):
        cmds= []
        cur = 0
        if cmd_buf is None:
            return cmds
        cmd_buf = np.frombuffer(cmd_buf, np.uint8)
        cmd_buf = np.unpackbits(cmd_buf, bitorder="little" )
        consume_len = 0
        while cmd_buf.size > 0 and len(cmds) < max_num:
            cmd = self.__decode_single(cmd_buf, cmd_bits, cmd_set)
            cmds.append(cmd)
            offset = cmd.len
            if cmd.type == 'dma.sys' or cmd.type == 'sys.end':
                # every command group is stored align to 128 bytes
                align_size = 128*8
                offset = (cmd.len + consume_len + align_size - 1)//align_size * align_size - consume_len
                #print(f"{cmd.type} offset = {offset}, consume_len={consume_len}, num_cmd={len(cmds)}")
            cmd_buf = cmd_buf[offset:]
            consume_len += offset

        assert len(cmds)<=max_num
        return cmds

    def parse(self, buf, max_num):
        assert False

    def command_byte_len(self):
        return self._byte_len_

class GDMACommandParser(BaseCommandParser):
    _byte_len_ = 256
    def __init__(self):
        super().__init__()
    def parse(self, cmd_buf, max_num):
        return self.decode(
            cmd_buf,
            opdef_1684x.dma_base.opcode_bits,
            opdef_1684x.dma_cmd, max_num)

class BDCommandParser(BaseCommandParser):
    _byte_len_ = 256
    def __init__(self):
        super().__init__()
    def parse(self, cmd_buf, max_num):
        return self.decode(
            cmd_buf,
            opdef_1684x.bdc_base.opcode_bits,
            opdef_1684x.bdc_cmd, max_num)
