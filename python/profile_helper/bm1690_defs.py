#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import ctypes as ct
from profile_helper.bmprofile_common import dictStructure
from enum import Enum
from debugger.target_common import get_target_context
from debugger.target_common import DType

ID_WIDTH = 20
GDMA_FREQ = 1000
BD_FREQ = 1000
BYTE_PER_BEAT = 128
GDMACyclePeriod = 1.0 / GDMA_FREQ
BDCyclePeriod = 1.0 / BD_FREQ
PeriodFixed = False
arch_name = "BM1690"
bd_sys_code = 15
dma_sys_code = 6
profile_sys_num = 2

class DynRecordType(Enum):
    FUNC = 0
    NODE_SET = 1
    NODE_WAIT = 2
    CDMA = 3
    GDE = 4
    SORT = 5
    NMS = 6
    VSDMA = 7
    SEND_MSG = 8
    WAIT_MSG = 9
    CUSTOM = 100
    UNKNOWN = -1

class EngineType(Enum):
    BD = 0
    GDMA = 1
    HAU = 2
    SDMA = 3
    CDMA = 4
    VSDMA = 5

class MEMTYPE(Enum):
    LMEM = 0
    DDR = 1

class TagType(Enum):
    TAG_USERS = 0
    TAG_WEIGHT = 1
    TAG_ACTIVATION = 2
    TAG_GLOBAL = 3
    TAG_L2M = 30
    TAG_LMEM = 31

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        return super().__eq__(other)

class DATATYPE(Enum):
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT16 = 3
    INT32 = 4
    BFP16 = 5
    INT4 = 6
    FP8 = 7
    FP20 = 8
    TF32 = 9

    def prec(self):
        if self.value == 0:
            return 1
        if self.value == 1 or self.value == 3 or self.value == 5:
            return 2
        if self.value == 2 or self.value == 4:
            return 4
        if self.value == 6:
            return 0.5


# CORE_OFFSET_BIT = 28
# CORE_OFFSET = 1 << CORE_OFFSET_BIT
# # local 256k * 64 = 16M
# LOCAL_MEM_ADDRWIDTH = 18
# LOCAL_MEM_SIZE = 1 << LOCAL_MEM_ADDRWIDTH
# NPU_SHIFT = 6
# NPU_NUM = 1 << NPU_SHIFT
# # static mem 64k
# STATIC_MEM_SIZE = 0x10000

# MAX_GMEM_SIZE = 1 << 40
# # global 4096M
# GLOBAL_MEM_START_ADDR = 0
# GLOBAL_MEM_SIZE = 0x100000000
# # l2 128MB
# L2_SRAM_START_ADDR = 0x6980000000
# L2_SRAM_SIZE = 0x8000000

# def is_lmem(addr, coreid):
#     lmem_start_addr = 0x6900000000 + coreid * CORE_OFFSET
#     return addr >= lmem_start_addr and addr < lmem_start_addr + NPU_NUM * LOCAL_MEM_SIZE

# def is_smem(addr, coreid):
#     smem_start_addr = 0x6904000000 + coreid * CORE_OFFSET
#     return addr >= smem_start_addr and addr < smem_start_addr + STATIC_MEM_SIZE

# def is_gmem(addr):
#     addr &= (MAX_GMEM_SIZE - 1)
#     return addr >= GLOBAL_MEM_START_ADDR and \
#         addr < (GLOBAL_MEM_START_ADDR + GLOBAL_MEM_SIZE)

# def is_l2mem(addr):
#     addr &= (MAX_GMEM_SIZE - 1)
#     return addr >= L2_SRAM_START_ADDR and \
#         addr < (L2_SRAM_START_ADDR  + L2_SRAM_SIZE)

# def mem_type(addr, core_id):
#     if is_lmem(addr, core_id):
#         assert()
#         return 'LMEM'
#     if is_smem(addr, core_id):
#         return 'SMEM'
#     if is_gmem(addr):
#         return 'DDR'
#     if is_l2mem(addr):
#         return 'L2M'
#     return 'None'
# static inline int is_gmem(u64 addr) {
#   addr &= (MAX_GMEM_SIZE - 1);
#   return addr >= GLOBAL_MEM_START_ADDR &&
#          addr < (GLOBAL_MEM_START_ADDR + CONFIG_GLOBAL_MEM_SIZE);
# }
# static inline int is_l2mem(u64 addr) {
#   addr &= (MAX_GMEM_SIZE - 1);
#   return (addr >= L2_SRAM_START_ADDR &&
#           addr < (L2_SRAM_START_ADDR  + L2_SRAM_SIZE));
# }

def get_node_set_info(dyn_data, extra_info=False):
    '''
    # type
    # firmware_profile.h
    # |0.....|8.......|11.............|15................|20.......|21....|28.......|32
    # | type | engine | bd_op/dma_op  | bd_func/dma_dir  | parellel| rsvd | version |
    # | 8    |  3     | 4             | 5                | 1       | 7    | 4       |

    # id
    # firmware_profile.c
    # |0...|16....|32....|48.....|64
    # | bd | gdma | sdma | vsdma |
    # | 16 | 16   | 16   | 16    |
    '''
    raw_type = dyn_data.type
    raw_id = dyn_data.id
    engine = EngineType((raw_type >> 8) & 0x7)
    if extra_info:
        dyn_data.des_tsk_typ = (raw_type >> 11) & 0xF
        dyn_data.des_tsk_eu_typ = (raw_type >> 15) & 0x1F
        dyn_data.parellel = (raw_type >> 20) & 0x1

    bd_cmd_id = raw_id & 0xFFFF  # notice just use 16bit/20bit
    gdma_cmd_id = (raw_id >> 16) & 0xFFFF
    sdma_cmd_id = (raw_id >> 32) & 0xFFFF
    vsdma_cmd_id = (raw_id >> 48) & 0xFFFF
    return engine, gdma_cmd_id, bd_cmd_id, sdma_cmd_id, vsdma_cmd_id

def dma_addr(H, L):
    addr = H * 2**32 + L
    tag = (addr >> 40) & 0x1f
    if tag == 0x0 :     # for workround
        addr =  addr | (0x1 << 40)
    return addr

class BDProfileFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint32),
        ("thread_id_and_bank_conflict", ct.c_uint32),
    ]


class GDMAProfileFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        # 32 bit align
        # H0
        ("inst_start_time", ct.c_uint32),      ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint32),              ("thread_id", ct.c_uint32, 1),
        ("ar_latency_cnt", ct.c_uint32, 19),   ("rip_valid_latency", ct.c_uint32, 12),
        # H1
        ("gif_wr_rd_stall_cntr", ct.c_uint32), ("axi_d0_w_cntr", ct.c_uint32),
        ("axi_d0_ar_cntr", ct.c_uint32),       ("axi_d0_aw_cntr", ct.c_uint32),
        # H2
        ("axi_d0_wr_stall_cntr", ct.c_uint32), ("axi_d0_rd_stall_cntr", ct.c_uint32),
        ("gif_mem_w_cntr", ct.c_uint32),       ("gif_mem_ar_cntr", ct.c_uint32),
        # H3
        ("axi_d0_wr_vaild_cntr", ct.c_uint32), ("axi_d0_rd_vaild_cntr", ct.c_uint32),
        ("gif_wr_valid_cntr", ct.c_uint32),    ("gif_rd_valid_cntr", ct.c_uint32),
    ]

class GDMACmdFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        # 32 bit align
        ("intr_en", ct.c_uint32, 1),           ("stride_enable", ct.c_uint32, 1),
        ("nchw_copy", ct.c_uint32, 1),         ("cmd_short", ct.c_uint32, 1),
        ("cache_en", ct.c_uint32, 1),          ("cache_flush", ct.c_uint32, 1),
        # 32bit
        ("reserved0", ct.c_uint32, 26),        ("cmd_type", ct.c_uint32, 5),
        ("cmd_special_function", ct.c_uint32, 3), ("fill_constant_en", ct.c_uint32, 1),
        ("src_data_format", ct.c_uint32, 4),   ("reserved1", ct.c_uint32, 3),
        # 64 bit
        ("reserved2", ct.c_uint32, 16),        ("cmd_id_dep", ct.c_uint32, 17),
        ("reserved3", ct.c_uint32, 6),         ("break_point", ct.c_uint32, 1),
        # 128 bit
        ("reserved4", ct.c_uint32, 8),         ("constant_value", ct.c_uint32),
        ("src_nstride", ct.c_uint32),        ("src_cstride", ct.c_uint32),
        ("src_hstride", ct.c_uint32),        ("src_wstride", ct.c_uint32),
        ("dst_nstride", ct.c_uint32),        ("dst_cstride", ct.c_uint32),
        ("dst_hstride", ct.c_uint32),        ("dst_wstride", ct.c_uint32),
        ("src_nsize", ct.c_uint32, 16),      ("src_csize", ct.c_uint32, 16),
        ("src_hsize", ct.c_uint32, 16),      ("src_wsize", ct.c_uint32, 16),
        ("dst_nsize", ct.c_uint32, 16),      ("dst_csize", ct.c_uint32, 16),
        ("dst_hsize", ct.c_uint32, 16),      ("dst_wsize", ct.c_uint32, 16),
        ("src_start_addr_l32", ct.c_uint32), ("src_start_addr_h13", ct.c_uint32, 13),
        ("reserved5", ct.c_uint32, 19),
        ("dst_start_addr_l32", ct.c_uint32), ("dst_start_addr_h13", ct.c_uint32, 13),
        ("reserved6", ct.c_uint32, 19),      ("all_reduce_code", ct.c_uint32, 16),
        ("reserved7", ct.c_uint32, 16),      ("reserved8", ct.c_uint32, 16),
        ("reserved9", ct.c_uint32, 16),
        ("localmem_mask_l32", ct.c_uint32),  ("localmem_mask_h32", ct.c_uint32),
    ]

class ProfileFormat(dictStructure):
    ''' |0.....|8.......|11.............|16................|21.......|22...................|32
        | type | engine | bd_op/dma_op  | bd_func/dma_dir  | parellel| rsvd/dtype/sys_info |
        | 8    |  3     | 5             | 5                | 1       | 10                 |
        rsvd/dtype/sys_info:
           bd: dtype (0xf, 4bits)
           dma: (direct << 4 | dtype) (0x7f, 7bit)
           sys: rsvd(0) or send_cnt/ wait_cnt (0x7f 7bits)
    '''
    _pack_ = 1
    _fields_ = [
        ("type", ct.c_uint32, 8),     ("engine", ct.c_uint32, 3),
        ("des_tsk_typ", ct.c_uint32, 5),   ("des_tsk_eu_typ", ct.c_uint32, 5),
        ("parellel", ct.c_uint32, 1),       ("extra_info", ct.c_uint32, 10),
        ("inst_id", ct.c_uint32), ("begin_cycle", ct.c_uint32)
    ]

class GDMACommandParser():
    _byte_len_ = 256
    def __init__(self) -> None:
        self.ctx = get_target_context("BM1690")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_dma_cmds(tmp)

class BDCommandParser():
    def __init__(self) -> None:
        self.ctx = get_target_context("BM1690")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_tiu_cmds(tmp)


DMA_ARCH = {
    "Chip Arch": "sg2260",
    "Platform": "ASIC",
    "Core Num": 8,
    "NPU Num": 64,
    "TPU Lmem Size(MiB)": 16777216,
    "Tpu Lmem Addr Width(bits)": 18,
    "Tpu Bank Addr Width(bits)": 14,
    "Execution Unit Number(int8)": 64,
    "Bus Max Burst": 2,
    "L2 Max Burst": 1,
    "Bus Bandwidth": 128,
    "DDR Frequency(GHz)": 8533,
    "DDR Max BW(GB/s/Core)": 68.264,
    "L2 Max BW(GB/s)": 128,
    "Cube IC Align(8bits)": 64,
    "Cube OHOW Align(8bits)": 4,
    "Cube OHOW Align(16bits)": 4,
    "Vector OHOW Align(8bits)": 64,
    "TIU Frequency(MHz)": 1000,
    "DMA Frequency(MHz)": 1000}

TIU_ARCH = DMA_ARCH

def get_src_dst_type(v):
    if v == 0:
        return "DDR", "LMEM"
    elif v == 1:
        return "LMEM", "DDR"
    elif v == 2:
        return "DDR", "DDR"
    elif v == 3:
        return "LMEM", "LMEM"

def mem_type(v):
    if v == TagType.TAG_LMEM:
        return "LMEM"
    if v == TagType.TAG_USERS \
            or v == TagType.TAG_WEIGHT \
            or v == TagType.TAG_ACTIVATION \
            or v == TagType.TAG_GLOBAL:
        return "DDR"
    if v == TagType.TAG_L2M:
        return "L2M"
    raise ValueError(f"Unknow dma mem_type: {v}")

def get_dma_info_dyn(monitor_info, reg_info, engine_id=1):
    if reg_info is None:
        reg_info = ProfileFormat()
        reg_info.des_tsk_typ = -1
    extra_info = reg_info.extra_info
    dma_info = dict()
    # step1 : get registor information from command
    dtype = extra_info&0xF
    data_type = DATATYPE(dtype)
    direction = extra_info >> 4
    src_type, dst_type = get_src_dst_type(direction)
    # step2: get custom information
    dma_info["Engine Id"] = engine_id
    dma_info["Direction"] = "{}->{}".format(src_type, dst_type)
    dma_info["from_addr"] = src_type
    dma_info["to_addr"] = dst_type
    dma_info["src_data_format"] = dtype
    dma_info["cmd_type"] = reg_info.des_tsk_typ
    dma_info["cmd_special_function"] = reg_info.des_tsk_eu_typ
    dma_info["Function Type"], dma_info["Function Name"] = getDmaFunctionName(
        reg_info.des_tsk_typ, reg_info.des_tsk_eu_typ, dma_info["Direction"])
    dma_info["Start Cycle"] = monitor_info.inst_start_time
    dma_info["End Cycle"] = monitor_info.inst_end_time
    dma_info["Cmd Id"] = monitor_info.inst_id + 1
    dma_info["Data Type"] = data_type.name
    dma_info["Asic Cycle"] = monitor_info.inst_end_time - \
        monitor_info.inst_start_time + 1
    dma_info["Stall Cycle"] = monitor_info.gif_wr_rd_stall_cntr
    dma_info["gmem_xfer_bytes(B)"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["gmem_bandwidth"] = round(dma_info["gmem_xfer_bytes(B)"] /
                                       dma_info["Asic Cycle"], 4)
    dma_info["gmem_dma_data_size(B)"] = dma_info["gmem_xfer_bytes(B)"]
    dma_info["lmem_xfer_bytes"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["lmem_bandwidth"] = round(dma_info["lmem_xfer_bytes"] / dma_info["Asic Cycle"], 4)

    dma_info["lmem_dma_data_size(B)"] = dma_info["lmem_xfer_bytes"]

    dma_info["DMA data size(B)"] = max(dma_info["gmem_dma_data_size(B)"], dma_info["lmem_dma_data_size(B)"])
    dma_info["DDR Bandwidth(GB/s)"] = max(dma_info["lmem_bandwidth"], dma_info["gmem_bandwidth"])
    # not implemented
    dma_info["gmem_bl_sum"] = 0
    dma_info["gmem_avg_burst_length"] = 0
    dma_info["lmem_bl_sum"] = 0
    dma_info["lmem_avg_burst_length"] = 0
    dma_info['L2M Bandwidth(GB/s)'] = 0
    # no need
    # dma_info["lmem_xact_cnt"] = monitor_info.axi_d0_ar_cntr + monitor_info.axi_d0_aw_cntr
    # dma_info["gmem_xact_cnt"] = dma_info["gmem_xfer_bytes(B)"] // BYTE_PER_BEAT
    dma_info["lmem_xact_cnt"] = monitor_info.gif_wr_valid_cntr + monitor_info.gif_rd_valid_cntr
    dma_info["gmem_xact_cnt"] = monitor_info.axi_d0_wr_vaild_cntr + monitor_info.axi_d0_rd_vaild_cntr
    dma_info["lmem_msk_wr_cnt"] = 0
    dma_info["gmem_msk_wr_cnt"] = 0
    dma_info["lmem_n32Ba_sa_cnt"] = 0
    dma_info["gmem_n32Ba_sa_cnt"] = 0
    dma_info["funcName"] = " "
    dma_info["layer_ID"] = ""
    dma_info["index"] = " "
    dma_info["Msg Id"] = " "
    dma_info["Sd\Wt Count"] = " "
    dma_info["mask_data_format"] = " "
    dma_info["index_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    dma_info["src_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    dma_info["dst_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    dma_info["dst_start_addr"] = 0
    dma_info["src_start_addr"] = 0
    return dma_info, None

def get_dma_info(monitor_info, reg_info, core_id, engine_id=1):
    is_sys = reg_info.name == 'sDMA_sys'
    _reg_info = reg_info
    reg_info = reg_info.reg
    dma_info = dict()
    # step1 : get registor information from command
    field_trans = {
        "dst_start_addr_h32": "dst_start_addr_l32",
        "dst_start_addr_l8": "dst_start_addr_h8",
        "src_start_addr_h32": "src_start_addr_l32",
        "src_start_addr_l8": "src_start_addr_h8"
    }

    for key, value in dict(reg_info).items():
        trans_key = field_trans.get(key, key)
        dma_info[trans_key] = value
    dma_info["mask_start_addr_h8"] = dma_info.get("mask_start_addr_h8", 0)
    dma_info["mask_start_addr_l32"] = dma_info.get("mask_start_addr_l32", 0)
    if is_sys:
        dma_info["dst_start_addr"] = 0
        dma_info["src_start_addr"] = 0
        dma_info["src_start_addr_h13"] = 0
        dma_info["dst_start_addr_h13"] = 0
    else:
        dma_info["dst_start_addr"] = (
            int(dma_info["dst_start_addr_h13"]) << 32) + int(dma_info["dst_start_addr_l32"])
        dma_info["src_start_addr"] = (
            int(dma_info["src_start_addr_h13"]) << 32) + int(dma_info["src_start_addr_l32"])

    # step2: get custom information
    src_type = mem_type(dma_info['src_start_addr_h13'] >> 8)
    dst_type = mem_type(dma_info['dst_start_addr_h13'] >> 8)
    # src_type = mem_type(dma_info['src_start_addr'], core_id)
    # dst_type = mem_type(dma_info['dst_start_addr'], core_id)
    data_type = DATATYPE(reg_info.src_data_format)

    dma_info["Engine Id"] = engine_id
    dma_info["Direction"] = "{}->{}".format(src_type, dst_type)
    dma_info["from_addr"] = src_type
    dma_info["to_addr"] = dst_type
    dma_info["Function Type"], dma_info["Function Name"] = getDmaFunctionName(
        reg_info.cmd_type, reg_info.cmd_special_function, dma_info["Direction"])
    dma_info["Start Cycle"] = monitor_info.inst_start_time
    dma_info["End Cycle"] = monitor_info.inst_end_time
    dma_info["Cmd Id"] = monitor_info.inst_id + 1
    dma_info["Data Type"] = data_type.name
    dma_info["Asic Cycle"] = monitor_info.inst_end_time - \
        monitor_info.inst_start_time + 1
    dma_info["Stall Cycle"] = monitor_info.gif_wr_rd_stall_cntr
    # print(monitor_info.axi_d0_w_cntr, monitor_info.axi_d0_ar_cntr, monitor_info.axi_d0_aw_cntr)
    dma_info["gmem_xfer_bytes(B)"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["gmem_bandwidth"] = round(dma_info["gmem_xfer_bytes(B)"] /
                                       dma_info["Asic Cycle"], 4)
    dma_info["gmem_dma_data_size(B)"] = dma_info["gmem_xfer_bytes(B)"]
    dma_info["lmem_xfer_bytes"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["lmem_bandwidth"] = round(dma_info["lmem_xfer_bytes"] / dma_info["Asic Cycle"], 4)

    dma_info["lmem_dma_data_size(B)"] = dma_info["lmem_xfer_bytes"]
    dma_info["DMA data size(B)"] = max(dma_info["gmem_dma_data_size(B)"], dma_info["lmem_dma_data_size(B)"])
    if "DDR" in dma_info["Direction"]:
        dma_info["DDR Bandwidth(GB/s)"] = max(dma_info["lmem_bandwidth"], dma_info["gmem_bandwidth"])
        dma_info['L2M Bandwidth(GB/s)'] = 0
    elif "L2M" in dma_info["Direction"]:
       dma_info["DDR Bandwidth(GB/s)"] = 0
       dma_info['L2M Bandwidth(GB/s)'] = max(dma_info["lmem_bandwidth"], dma_info["gmem_bandwidth"])
    else:
       dma_info["DDR Bandwidth(GB/s)"] = 0
       dma_info['L2M Bandwidth(GB/s)'] = 0
    # else:
    #     raise ValueError(f"Unknow direction type: {dma_info['Direction']}")

    # not implemented
    dma_info["gmem_bl_sum"] = 0
    dma_info["gmem_avg_burst_length"] = 0
    dma_info["lmem_bl_sum"] = 0
    dma_info["lmem_avg_burst_length"] = 0
    # no need
    # dma_info["lmem_xact_cnt"] = monitor_info.axi_d0_ar_cntr + monitor_info.axi_d0_aw_cntr
    # dma_info["gmem_xact_cnt"] = dma_info["gmem_xfer_bytes(B)"] // BYTE_PER_BEAT
    dma_info["lmem_xact_cnt"] = monitor_info.gif_wr_valid_cntr + monitor_info.gif_rd_valid_cntr
    dma_info["gmem_xact_cnt"] = monitor_info.axi_d0_wr_vaild_cntr + monitor_info.axi_d0_rd_vaild_cntr
    dma_info["lmem_msk_wr_cnt"] = 0
    dma_info["gmem_msk_wr_cnt"] = 0
    dma_info["lmem_n32Ba_sa_cnt"] = 0
    dma_info["gmem_n32Ba_sa_cnt"] = 0
    dma_info["funcName"] = " "
    dma_info["layer_ID"] = ""
    dma_info["index"] = " "
    dma_info["Msg Id"] = " "
    dma_info["Sd\Wt Count"] = " "
    dma_info["mask_data_format"] = " "
    dma_info["index_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    dma_info["src_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    dma_info["dst_shape"] = "(1, 128, 128 ,9)(FP32)(147456, 1152, 9, 1)"
    return dma_info, None

def get_tiu_info_dyn(monitor_info, reg_info):
    tiu_info0, tiu_info1 = dict(), dict()
    if reg_info is None:
        reg_info = ProfileFormat()
        reg_info.des_tsk_typ = -1
    for key in reg_info._fields_:
        if  isinstance(key, tuple):
            key = key[0]
        tiu_info1[key] = getattr(reg_info, key)
    tiu_info0["Function Type"], tiu_info0["Function Name"] = getTiuFunctionName(
        reg_info.des_tsk_typ, reg_info.des_tsk_eu_typ)
    if tiu_info0["Function Type"] != 'sys':
        tiu_info1["des_opt_res0_prec"] = reg_info.extra_info
    tiu_info0["Start Cycle"] = monitor_info.inst_start_time
    tiu_info0["End Cycle"] = monitor_info.inst_end_time
    tiu_info0["Cmd Id"] = monitor_info.inst_id + 1
    tiu_info0["Asic Cycle"] = monitor_info.inst_end_time - \
        monitor_info.inst_start_time + 1
    tiu_info0["Engine Id"] = 0
    tiu_info0["Alg Ops"] = 1
    tiu_info0["uArch Ops"] = 1
    tiu_info0["Alg Cycle"] = 0
    tiu_info0["uArch Rate"] = "100.0%"


    # not implemented
    tiu_info0["Initial Cycle Ratio"] = "0.0%"
    tiu_info0["Bank Conflict Ratio"] = "0.0%"
    tiu_info0['Sim Power(W)'] = 0
    tiu_info1["Msg Id"] = 0
    tiu_info1["Sd\Wt Count"] = 0
    return tiu_info0, tiu_info1

def get_tiu_info(monitor_info, reg_info):
    _reg_info = reg_info
    reg_info = reg_info.reg
    tiu_info0, tiu_info1 = dict(), dict()
    for key, value in dict(reg_info).items():
        tiu_info1[f"des_{key}"] = value
    tiu_info1["Msg Id"] = 0
    tiu_info1["Sd\Wt Count"] = 0
    if reg_info.OP_NAME == 'SYS' and reg_info.tsk_eu_typ // 2 == 4:
        tiu_info1["Msg Id"] = reg_info.imm & 0b1111111
        tiu_info1["Sd\Wt Count"] = (reg_info.imm >> 16) & 0b111

    tiu_info0["Function Name"] = _reg_info.op_name
    tiu_info0["Function Type"] = reg_info.OP_NAME
    tiu_info0["Start Cycle"] = monitor_info.inst_start_time
    tiu_info0["End Cycle"] = monitor_info.inst_end_time
    tiu_info0["Cmd Id"] = monitor_info.inst_id + 1
    tiu_info0["Asic Cycle"] = monitor_info.inst_end_time - \
        monitor_info.inst_start_time + 1
    tiu_info0["Engine Id"] = 0

    tiu_info0["Alg Ops"] = _reg_info.ops(False)
    tiu_info0["uArch Ops"] = _reg_info.ops(True)

    # tiu_info0["Alg Cycle"] = _reg_info.alg_cycle(tiu_info0["Alg Ops"])
    # tiu_info0["uArch Rate"] = "{:.1%}".format(
    #     tiu_info0["Alg Ops"]/tiu_info0["uArch Ops"])
    # not implemented
    tiu_info0["Alg Cycle"] = 1
    tiu_info0["uArch Rate"] = "0.0%"
    tiu_info0["Initial Cycle Ratio"] = "{:.1%}".format(1e-4)
    tiu_info0["Bank Conflict Ratio"] = "{:.1%}".format(1e-4)


    # not implemented
    tiu_info0['Sim Power(W)'] = 0
    return tiu_info0, tiu_info1

def getTiuFunctionName(cmd_type, cmd_special_function):
    tiuFunctionNameDict = {
        # conv
        0: "conv",
        (0, 0): 'conv', (0, 1): 'conv_normal', (0, 2): 'conv_tf32',
        # depthwise or pooling
        1: "dw/pool",
        (1, 0): 'depthwise', (1, 1): 'avg pooling', (1, 3): 'min pooling',(1, 4): 'max pooling',
        (1, 5): 'ROI depthwise',(1, 6): 'ROI avg pooling',(1, 7): 'ROI max pooling',(1, 8): 'ROI min pooling',
        # matrix multiply && matrix multiply2
        2: "mm/mm2",
        (2, 0): 'MM_NORMAL', (2, 4): 'MM2_NN', (2, 5): 'MM2_NT', (2, 6): 'MM2_TT',
        (2, 7): 'MM_NN_TF32', (2, 8): 'MM_NT_TF32',  (2, 9): 'MM_TT_TF32',
        # arithmetic && SEG
        3: "arith/SEG",
        (3, 0): 'mul', (3, 1): 'not', (3, 2): 'add', (3, 3): 'sub', (3, 4): 'max', (3, 5): 'min',
        (3, 6): 'logic_shift', (3, 7): 'and', (3, 8): 'or', (3, 9): 'xor', (3, 10): 'select_great',
        (3, 11): 'select_equal', (3, 12): 'div', (3, 13): 'select_less', (3, 14): 'data_convert',
        (3, 15): 'add_satu', (3, 16): 'sub_satu', (3, 18): 'mac', (3, 19): 'copy', (3, 20): 'mul_satu',
        (3, 21): 'arith shift', (3, 22): 'rotate shift', (3, 23): 'mulDHR [N]', (3, 26): 'ABS',
        (3, 27): 'FSUBABS', (3, 28): 'copy_mb [N]', (3, 29): 'get_first_one', (3, 30): 'get_first_zero',
        # RQ && DQ
        4: "RQ/DQ",
        (4, 0): 'RQ_0', (4, 1): 'RQ_1', (4, 3): 'DQ_0', (4, 4): 'DQ_1',
        # TRANS && BC
        5: "TRANS/BC",
        (5, 0): 'C_W_transpose', (5, 1): 'W_C_transpose', (5, 2): 'lane_copy', (5, 3): 'lane_broad',
        (5, 4): 'static_broad', (5, 5): 'static_distribute',
        # scatter_gather && scatter_gather_line
        6: "gather/scatter",
        (6, 0): 'PL_gather_d1coor', (6, 1): 'PL_gather_d2coor', (6, 2): 'PL_gather_rec [N]',
        (6, 3): 'PL_scatter_d1coor', (6, 4): 'PL_scatter_d2coor',
        (6, 5): 'PE_S_gather_d1coor', (6, 6): 'PE_S_scatter_d1coor', (6, 7): 'PE_M_gather_d1coor [N]',
        (6, 8): 'PE_S_mask_select', (6, 9): 'PE_S_nonzero', (6, 10): 'PE_S_scatter_pp_d1coor [N]',
        (6, 13): 'PE_S_gather_hzd', (6, 14): 'PE_S_scatter_hzd', (6, 15): 'PE_S_mask_selhzd',
        (6, 16): 'PE_S_nonzero_hzd', (6, 17): 'PE_S_gather_line', (6, 18): 'PE_S_scatter_line',
        # linear_arithmetic
        7: 'linear_arithmetic',
        (7, 0): 'linear_mul', (7, 1): 'linear_not', (7, 2): 'linear_add', (7, 3): 'linear_sub',
        (7, 4): 'linear_max', (7, 5): 'linear_min', (7, 6): 'linear_logic_shift', (7, 7): 'linear_and',
        (7, 8): 'linear_or', (7, 9): 'linear_xor', (7, 10): 'linear_select_great',
        (7, 11): 'linear_select_equal', (7, 12): 'linear_div', (7, 13): 'linear_select_less',
        (7, 14): 'linear_data_convert', (7, 15): 'linear_add_satu', (7, 16): 'linear_sub_satu',
        (7, 17): 'linear_clamp', (7, 18): 'linear_mac', (7, 19): 'linear_copy',(7, 20): 'linear_mul_satu',
        # rand_gen
        8: "rand_gen",
        (8, 0): 'prng ', (8, 1): 'prng with intial seed', (8, 2): ':prng with loaded states',
        # special_function
        9: 'md/sfu',
        (9, 12): 'tailor_4x', (9, 13): 'tailor', (9, 15): 'normalize', (9, 17): 'rsqrt',
        # fused_linear
        10: 'md_linear',
        (10, 1): 'mac', (10, 20): '(a+b)^2', (10, 21): '(a-b)^2',
        # SYS_TR_WR
        12: 'WR_IMM',
        (12, 0): 'WR_IMM',
        # fused_cmpare
        13: 'fused_cmpare',
        (13, 22): 'CMP_gt and CMP_sel_gt', (13, 23): 'CMP_sel_gt', (13, 24): 'CMP_sel_eq',
        (13, 25): 'CMP_lt and CMP_sel_lt', (13, 26): 'CMP_sel_lt', (13, 27): 'CMP_srch_bin',
        # vector correlation
        14: 'vector_correlation',
        (14, 0): 'vc_mul',  (14, 2): 'vc_add', (14, 3): 'vc_sub', (14, 4): 'vc_max',
        (14, 5): 'vc_min',  (14, 7): 'vc_and', (14, 8): 'vc_or', (14, 9): 'xor',
        (14, 10): 'select_great', (14, 11): 'select_equal', (14, 12): 'div', (14, 13): 'select_less',
        (14, 15): 'add_satu', (14, 16): 'sub_satu', (14, 20): 'mul_satu', (14, 23): 'mulDHR',
        # system
        15: 'sys',
        (15, 0): 'intr barrier[N]', (15, 1): 'spb', (15, 2): 'swr', (15, 3): 'swr_from_lmem',
        (15, 4): 'swr_collect_from_lmem', (15, 5): 'ata barrier[N]',
        (15, 8): 'send_msg', (15, 9): 'wait_msg', (15, 10): 'sys_fork', (15, 11): 'sys_join',
        (15, 12): 'sys_exit', (15, 13): 'rand_seed', (15, 30): 'nop', (15, 31): 'end',
        31: 'unknown',
        (31, 0): 'unknown'
    }
    functionType = tiuFunctionNameDict.get((cmd_type), f'cmd_type_{cmd_type}')
    functinName = tiuFunctionNameDict.get((cmd_type, cmd_special_function),  f"{functionType}_{cmd_special_function}")
    return functionType, functinName

def getDmaFunctionName(cmd_type, cmd_special_function, direction):
    dmaFunctionNameDict = {
        (0, 0): 'DMA_tensor', (0, 1): 'NC trans', (0, 2): 'collect', (0, 3): 'broadcast', (0, 4): 'distribute', (0, 5): 'lmem 4 bank copy', (0, 6): 'lmem 4 bank broadcast',
        (1, 0): 'DMA_matrix', (1, 1): 'matrix transpose',
        (2, 0): 'DMA_masked_select', (2, 1): 'ncw mode',
        (3, 0): 'DMA_general', (3, 1): 'broadcast',
        (4, 0): 'DMA_cw transpose',
        (5, 0): 'DMA_nonzero',
        (6, 0): 'DMA_sys', (6, 1): 'nop', (6, 2): 'sys_tr_wr', (6, 3): 'sys_send', (6, 4): 'sys_wait',
        (7, 0): 'DMA_gather',
        (8, 0): 'DMA_scatter',
        (9, 0): 'DMA_reverse', (9, 1): 'h reverse', (9, 2): 'c reverse', (9, 3): 'n reverse',
        (10, 0): 'DMA_compress',
        (11, 0): 'DMA_decompress',
        (12, 0): 'DMA_lossy_compress',
        (13, 0): 'DMA_lossy_decompress',
        (14, 0): 'DMA_randmask',
        (15, 0): 'DMA_transfer',
        (31, 0): "unknown",
    }
    functionType = dmaFunctionNameDict[(cmd_type, 0)]
    direction_dict = {
        "DDR->DDR": "Ld",
        "DDR->LMEM": "Ld",
        "LMEM->DDR": "St",
        "LMEM->LMEM": "Mv"
    }
    functinName = dmaFunctionNameDict[(cmd_type, cmd_special_function)]
    if functionType != "unknown" and cmd_special_function == 0 and cmd_type <= 1:
        functinName = "tensor{}".format(direction_dict.get(direction, ""))

    return functionType, functinName

def show_arch_info():
    print("BM1690")
