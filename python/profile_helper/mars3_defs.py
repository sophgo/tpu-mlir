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

GDMA_FREQ = 650
BD_FREQ = 750
# TODO: mars3 has more than three kinds of work freqs, need auto parse form pmu header
arch_name = "MARS3"


class EngineType(Enum):
    BD = 0
    GDMA = 1


class MEMTYPE(Enum):
    LMEM = 0
    DDR = 1


class DATATYPE(Enum):
    INT8 = 0
    FP32 = 2
    INT16 = 3
    INT32 = 4
    BFP16 = 5


class BDProfileFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        ("inst_id", ct.c_uint32, 20),
        ("reserved", ct.c_uint32, 12),
        ("thread_id", ct.c_uint32, 1),
        ("bank_conflict", ct.c_uint32, 31),
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
    ]


class GDMAProfileFormat(dictStructure):
    _pack_ = 1
    _fields_ = [
        # H0
        ("inst_start_time", ct.c_uint32),
        ("inst_end_time", ct.c_uint32),
        ("inst_id", ct.c_uint32),
        ("thread_id", ct.c_uint32, 1),
        ("ar_latency_cnt", ct.c_uint32, 19),
        ("rip_valid_latency", ct.c_uint32, 12),
        # H1
        ("gif_wr_rd_stall_cntr", ct.c_uint32),
        ("axi_d0_w_cntr", ct.c_uint32),
        ("axi_d0_ar_cntr", ct.c_uint32),
        ("axi_d0_aw_cntr", ct.c_uint32),
        # H2
        ("axi_d0_wr_stall_cntr", ct.c_uint32),
        ("axi_d0_rd_stall_cntr", ct.c_uint32),
        ("gif_mem_w_cntr", ct.c_uint32),
        ("gif_mem_ar_cntr", ct.c_uint32),
        # H3
        ("axi_d0_wr_vaild_cntr", ct.c_uint32),
        ("axi_d0_rd_vaild_cntr", ct.c_uint32),
        ("gif_wr_valid_cntr", ct.c_uint32),
        ("gif_rd_valid_cntr", ct.c_uint32),
    ]


class GDMACommandParser():

    def __init__(self) -> None:
        self.ctx = get_target_context("MARS3")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_dma_cmds(tmp)


class BDCommandParser():

    def __init__(self) -> None:
        self.ctx = get_target_context("MARS3")

    def parse(self, raw_data):
        tmp = bytearray(raw_data)
        return self.ctx.decoder.decode_tiu_cmds(tmp)


DMA_ARCH = {
    "Chip Arch": "MARS3",
    "Platform": "ASIC",
    "Core Num": 1,
    "NPU Num": 8,
    "TPU Lmem Size(Bytes)": 64 * 1024 * 8,
    "Tpu Lmem Addr Width(bits)": 16,
    "Tpu Bank Addr Width(bits)": 12,
    "Execution Unit Number(int8)": 8,
    "Bus Max Burst": 32,
    "L2 Max Burst": 0,
    "Bus Bandwidth": 64,
    "DDR Frequency(MHz)": 2133,
    "DDR Max BW(GB/s/Core)": 4,
    "L2 Max BW(GB/s)": 0,
    "Cube IC Align(8bits)": 16,
    "Cube OHOW Align(8bits)": 4,
    "Cube OHOW Align(16bits)": 2,
    "Vector OHOW Align(8bits)": 8,
    "TIU Frequency(MHz)": BD_FREQ,
    "DMA Frequency(MHz)": GDMA_FREQ
}

TIU_ARCH = DMA_ARCH


def get_dma_info(monitor_info, reg_info):
    is_sys = reg_info.name == 'sDMA_sys'
    _reg_info = reg_info
    reg_info = reg_info.reg
    dma_info = dict()
    # step1 : get registor information from command
    field_trans = {
        "dst_start_addr_h32": "dst_start_addr_l32",
        "dst_start_addr_l8": "dst_start_addr_h13",
        "src_start_addr_h32": "src_start_addr_l32",
        "src_start_addr_l8": "src_start_addr_h13"
    }
    for key, value in dict(reg_info).items():
        trans_key = field_trans.get(key, key)
        dma_info[trans_key] = value
    dma_info["mask_start_addr_h8"] = dma_info.get("mask_start_addr_h8", 0)
    dma_info["mask_start_addr_l32"] = dma_info.get("mask_start_addr_l32", 0)
    if is_sys:
        dma_info["dst_start_addr_l32"] = 0
        dma_info["src_start_addr_l32"] = 0
        dma_info["src_start_addr_h13"] = 0
        dma_info["dst_start_addr_h13"] = 0
    else:
        dma_info["dst_start_addr"] = (int(dma_info["dst_start_addr_h13"]) << 32) + int(
            dma_info["dst_start_addr_l32"])
        dma_info["src_start_addr"] = (int(dma_info["src_start_addr_h13"]) << 32) + int(
            dma_info["src_start_addr_l32"])

    # step2: get custom information
    src_type = MEMTYPE(dma_info['src_start_addr_l32'] >> 31).name
    dst_type = MEMTYPE(dma_info['dst_start_addr_l32'] >> 31).name
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
    dma_info["Stall Cycle"] = monitor_info.gif_wr_rd_stall_cntr
    dma_info["gmem_xfer_bytes(B)"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["gmem_bandwidth"] = round(
        dma_info["gmem_xfer_bytes(B)"] / dma_info["Asic Cycle"] * (GDMA_FREQ / 1000), 4)
    # dma_info["gmem_dma_data_size(B)"] = max((getattr(reg_info, "src_nsize", 0) or 1) * (getattr(reg_info, "src_csize", 0) or 1) * getattr(reg_info,"src_hsize", 0) * getattr(reg_info,"src_wsize", 0),
    #                                         (getattr(reg_info, "dst_nsize", 0) or 1) * (getattr(reg_info, "dst_csize", 0) or 1) * getattr(reg_info,"dst_hsize", 0) * getattr(reg_info,"src_wsize", 0)) * data_type.prec()
    # dma_info["lmem_dma_data_size(B)"] = max((getattr(reg_info, "src_nsize", 0) or 1) * (getattr(reg_info, "src_csize", 0) or 1) * getattr(reg_info,"src_hsize", 0) * getattr(reg_info,"src_wsize", 0),
    #                                         (getattr(reg_info, "dst_nsize", 0) or 1) * (getattr(reg_info, "dst_csize", 0) or 1) * getattr(reg_info,"dst_hsize", 0) * getattr(reg_info,"src_wsize", 0)) * data_type.prec()
    dma_info["gmem_dma_data_size(B)"] = dma_info["gmem_xfer_bytes(B)"]
    dma_info[
        "gmem_xact_cnt"] = monitor_info.axi_d0_wr_vaild_cntr + monitor_info.axi_d0_rd_vaild_cntr
    dma_info["lmem_xfer_bytes"] = monitor_info.gif_mem_w_cntr + monitor_info.axi_d0_w_cntr
    dma_info["lmem_bandwidth"] = round(
        dma_info["lmem_xfer_bytes"] / dma_info["Asic Cycle"] * (GDMA_FREQ / 1000), 4)
    dma_info["lmem_dma_data_size(B)"] = dma_info["lmem_xfer_bytes"]
    dma_info["lmem_xact_cnt"] = monitor_info.gif_wr_valid_cntr + monitor_info.gif_rd_valid_cntr
    dma_info["DMA data size(B)"] = max(dma_info["gmem_dma_data_size(B)"],
                                       dma_info["lmem_dma_data_size(B)"])
    dma_info["DDR Bandwidth(GB/s)"] = max(dma_info["lmem_bandwidth"], dma_info["gmem_bandwidth"])

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
    tiu_info0["uArch Rate"] = "{:.1%}".format(tiu_info0["Alg Ops"] / tiu_info0["uArch Ops"])
    tiu_info0["Initial Cycle Ratio"] = "{:.1%}".format(_reg_info.initial_cycle() /
                                                       (tiu_info0["Asic Cycle"] + 1e-4))
    tiu_info0["Bank Conflict Ratio"] = "{:.1%}".format(_reg_info.bank_conflict_cycle() /
                                                       (tiu_info0["Asic Cycle"] + 1e-4))

    # not implemented
    tiu_info0['Sim Power(W)'] = 0
    return tiu_info0, tiu_info1


def getDmaFunctionName(cmd_type, cmd_special_function, direction):
    dmaFunctionNameDict = {
        (0, 0): 'DMA_tensor',
        (0, 1): 'NC trans',
        (0, 2): 'collect',
        (0, 3): 'broadcast',
        (0, 4): 'distribute',
        (0, 5): 'lmem 4 bank copy',
        (0, 6): 'lmem 4 bank broadcast',
        (1, 0): 'DMA_matrix',
        (1, 1): 'matrix transpose',
        (2, 0): 'DMA_masked_select',
        (2, 1): 'ncw mode',
        (3, 0): 'DMA_general',
        (3, 1): 'broadcast',
        (4, 0): 'cw transpose',
        (5, 0): 'DMA_nonzero',
        (6, 0): 'chain end',
        (6, 1): 'nop',
        (6, 2): 'sys_tr_wr',
        (6, 3): 'sys_send',
        (6, 4): 'sys_wait',
        (7, 0): 'DMA_gather',
        (8, 0): 'DMA_scatter',
        (9, 0): 'w reverse',
        (9, 1): 'h reverse',
        (9, 2): 'c reverse',
        (9, 3): 'n reverse',
        (10, 0): 'non-random-access',
        (10, 1): 'random-access',
        (11, 0): 'non-random-access',
        (11, 1): 'random-access'
    }
    functionType = dmaFunctionNameDict[(cmd_type, 0)]
    direction_dict = {"DDR->DDR": "Ld", "DDR->LMEM": "Ld", "LMEM->DDR": "St", "LMEM->LMEM": "Mv"}
    functinName = dmaFunctionNameDict[(cmd_type, cmd_special_function)]
    if cmd_special_function == 0 and cmd_type <= 1:
        functinName = "tensor{}".format(direction_dict.get(direction, ""))

    return functionType, functinName
