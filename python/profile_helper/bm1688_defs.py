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

GDMA_FREQ = 750
BD_FREQ = 900
arch_name = "BM1688"

class EngineType(Enum):
    BD = 0
    GDMA = 1


class MEMTYPE(Enum):
    LMEM = 0
    DDR = 1


class DATATYPE(Enum):
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT16 = 3
    INT32 = 4
    BFP16 = 5
    INT4 = 6

    def prec(self):
        if self.value == 0:
            return 1
        if self.value == 1 or self.value == 3 or self.value == 5:
            return 2
        if self.value == 2 or self.value == 4:
            return 4
        if self.value == 6:
            return 0.5


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
        ("inst_start_time", ct.c_uint32), ("inst_end_time", ct.c_uint32), ("inst_id",
                                                                           ct.c_uint32, 20), ("reserved", ct.c_uint32, 12), ("d0_aw_bytes", ct.c_uint32),
        ("d0_wr_bytes", ct.c_uint32), ("d0_ar_bytes",
                                       ct.c_uint32), ("d1_aw_bytes", ct.c_uint32), ("d1_wr_bytes", ct.c_uint32),
        ("d1_ar_bytes", ct.c_uint32), ("fmem_aw_bytes",
                                       ct.c_uint32), ("fmem_wr_bytes", ct.c_uint32), ("fmem_ar_bytes", ct.c_uint32),
        ("l2sram_aw_bytes", ct.c_uint32), ("l2sram_wr_bytes",
                                           ct.c_uint32), ("l2sram_ar_bytes", ct.c_uint32), ("reserved1", ct.c_uint32),
        ("d0_wr_valid_bytes", ct.c_uint32), ("d0_rd_valid_bytes", ct.c_uint32), (
            "d1_wr_valid_bytes", ct.c_uint32), ("d1_rd_valid_bytes", ct.c_uint32),
        ("fmem_wr_valid_bytes", ct.c_uint32), ("fmem_rd_valid_bytes", ct.c_uint32), (
            "l2sram_wr_valid_bytes", ct.c_uint32), ("l2sram_rd_valid_bytes", ct.c_uint32),
        # ("no_need40", ct.c_uint64), ("no_need48", ct.c_uint64),
        # ("no_need50", ct.c_uint64), ("no_need58", ct.c_uint64),
        ("no_need60", ct.c_uint64), ("no_need68", ct.c_uint64),
        ("gif_fmem_wr_stall_bytes", ct.c_uint32), ("gif_fmem_rd_stall_bytes", ct.c_uint32), (
            "gif_l2sram_wr_stall_bytes", ct.c_uint32), ("gif_12sram_rd_stall_bytes", ct.c_uint32),
        # ("no_need70", ct.c_uint64), ("no_need78", ct.c_uint64),
        ("no_need80", ct.c_uint64), ("no_need88", ct.c_uint64),
        ("no_need90", ct.c_uint64), ("no_need98", ct.c_uint64),
        ("no_needA0", ct.c_uint64), ("no_needA8", ct.c_uint64),
        ("no_needB0", ct.c_uint64), ("no_needB8", ct.c_uint64),
        ("no_needC0", ct.c_uint64), ("no_needC8", ct.c_uint64),
        ("no_needD0", ct.c_uint64), ("no_needD8", ct.c_uint64),
        ("no_needE0", ct.c_uint64), ("no_needE8", ct.c_uint64),
        ("no_needF0", ct.c_uint64), ("no_needF8", ct.c_uint64),
    ]


class GDMACommandParser():
    def __init__(self) -> None:
        self.ctx = get_target_context("BM1688")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_dma_cmds(tmp)


class BDCommandParser():
    def __init__(self) -> None:
        self.ctx = get_target_context("BM1688")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_tiu_cmds(tmp)


DMA_ARCH = {
    "Chip Arch": "A2",
    "Platform": "ASIC",
    "Core Num": 2,
    "NPU Num": 32,
    "TPU Lmem Size(MiB)": 4194304,
    "Tpu Lmem Addr Width(bits)": 17,
    "Tpu Bank Addr Width(bits)": 13,
    "Execution Unit Number(int8)": 16,
    "Bus Max Burst": 16,
    "L2 Max Burst": 0,
    "Bus Bandwidth": 64,
    "DDR Frequency(GHz)": 4266,
    "DDR Max BW(GB/s/Core)": 32,
    "L2 Max BW(GB/s)": 0,
    "Cube IC Align(8bits)": 32,
    "Cube OHOW Align(8bits)": 4,
    "Cube OHOW Align(16bits)": 4,
    "Vector OHOW Align(8bits)": 16,
    "TIU Frequency(MHz)": 900,
    "DMA Frequency(MHz)": 750}

TIU_ARCH = DMA_ARCH


def get_dma_info(monitor_info, reg_info):
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
        dma_info["src_start_addr_h8"] = 0
        dma_info["dst_start_addr_h8"] = 0
    else:
        dma_info["dst_start_addr"] = (
            int(dma_info["dst_start_addr_h8"]) << 32) + int(dma_info["dst_start_addr_l32"])
        dma_info["src_start_addr"] = (
            int(dma_info["src_start_addr_h8"]) << 32) + int(dma_info["src_start_addr_l32"])

    # step2: get custom information
    src_type = MEMTYPE(dma_info['src_start_addr_h8'] >> 7).name
    dst_type = MEMTYPE(dma_info['dst_start_addr_h8'] >> 7).name
    data_type = DATATYPE(reg_info.src_data_format)

    dma_info["Engine Id"] = 1
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
    dma_info["Stall Cycle"] = monitor_info.gif_fmem_wr_stall_bytes + \
        monitor_info.gif_fmem_rd_stall_bytes

    dma_info["gmem_xfer_bytes(B)"] = monitor_info.d0_aw_bytes + \
        monitor_info.d1_aw_bytes + monitor_info.d0_ar_bytes + monitor_info.d1_ar_bytes
    dma_info["gmem_bandwidth"] = round(dma_info["gmem_xfer_bytes(B)"] /
                                       dma_info["Asic Cycle"] * 0.75, 4)
    # dma_info["gmem_dma_data_size(B)"] = max((getattr(reg_info, "src_nsize", 0) or 1) * (getattr(reg_info, "src_csize", 0) or 1) * getattr(reg_info,"src_hsize", 0) * getattr(reg_info,"src_wsize", 0),
    #                                         (getattr(reg_info, "dst_nsize", 0) or 1) * (getattr(reg_info, "dst_csize", 0) or 1) * getattr(reg_info,"dst_hsize", 0) * getattr(reg_info,"src_wsize", 0)) * data_type.prec()
    # dma_info["lmem_dma_data_size(B)"] = max((getattr(reg_info, "src_nsize", 0) or 1) * (getattr(reg_info, "src_csize", 0) or 1) * getattr(reg_info,"src_hsize", 0) * getattr(reg_info,"src_wsize", 0),
    #                                         (getattr(reg_info, "dst_nsize", 0) or 1) * (getattr(reg_info, "dst_csize", 0) or 1) * getattr(reg_info,"dst_hsize", 0) * getattr(reg_info,"src_wsize", 0)) * data_type.prec()
    dma_info["gmem_dma_data_size(B)"] = dma_info["gmem_xfer_bytes(B)"]
    dma_info["gmem_xact_cnt"] = (monitor_info.d0_ar_bytes + monitor_info.d0_aw_bytes +
                                 monitor_info.d1_ar_bytes + monitor_info.d1_aw_bytes) // DMA_ARCH["Vector OHOW Align(8bits)"]
    dma_info["lmem_xfer_bytes"] = monitor_info.fmem_aw_bytes + \
        monitor_info.fmem_ar_bytes
    dma_info["lmem_bandwidth"] = round(dma_info["lmem_xfer_bytes"] /
                                       dma_info["Asic Cycle"] * 0.75, 4)
    dma_info["lmem_dma_data_size(B)"] = dma_info["lmem_xfer_bytes"]
    dma_info["lmem_xact_cnt"] = (
        monitor_info.fmem_ar_bytes + monitor_info.fmem_aw_bytes) // DMA_ARCH["Vector OHOW Align(8bits)"]
    dma_info["DMA data size(B)"] = max(
        dma_info["gmem_dma_data_size(B)"], dma_info["lmem_dma_data_size(B)"])
    dma_info["DDR Bandwidth(GB/s)"] = max(
        dma_info["lmem_bandwidth"], dma_info["gmem_bandwidth"])

    # not implemented
    dma_info["gmem_bl_sum"] = 0
    dma_info["gmem_avg_burst_length"] = 0
    dma_info["lmem_bl_sum"] = 0
    dma_info["lmem_avg_burst_length"] = 0
    dma_info['L2M Bandwidth(GB/s)'] = 0
    # no need
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
    return dma_info


def get_tiu_info(monitor_info, reg_info):
    # cmd_params = reg_info.ops(reg_info.reg)
    # print(cmd_params)
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
    tiu_info0["Alg Cycle"] = _reg_info.alg_cycle(tiu_info0["Alg Ops"])
    tiu_info0["uArch Rate"] = "{:.1%}".format(
        tiu_info0["Alg Ops"]/tiu_info0["uArch Ops"])
    tiu_info0["Initial Cycle Ratio"] = "{:.1%}".format(
        _reg_info.initial_cycle() / (tiu_info0["Asic Cycle"] + 1e-4))
    tiu_info0["Bank Conflict Ratio"] = "{:.1%}".format(
        _reg_info.bank_conflict_cycle() / (tiu_info0["Asic Cycle"] + 1e-4))

    # not implemented
    tiu_info0['Sim Power(W)'] = 0
    return tiu_info0, tiu_info1


def getDmaFunctionName(cmd_type, cmd_special_function, direction):
    dmaFunctionNameDict = {
        (0, 0): 'DMA_tensor', (0, 1): 'NC trans', (0, 2): 'collect', (0, 3): 'broadcast', (0, 4): 'distribute', (0, 5): 'lmem 4 bank copy', (0, 6): 'lmem 4 bank broadcast',
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
    functionType = dmaFunctionNameDict[(cmd_type, 0)]
    direction_dict = {
        "DDR->DDR": "Ld",
        "DDR->LMEM": "Ld",
        "LMEM->DDR": "St",
        "LMEM->LMEM": "Mv"
    }
    functinName = dmaFunctionNameDict[(cmd_type, cmd_special_function)]
    if cmd_special_function == 0 and cmd_type <= 1:
        functinName = "tensor{}".format(direction_dict.get(direction, ""))

    return functionType, functinName
