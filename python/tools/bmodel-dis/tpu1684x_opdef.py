# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

from abc import abstractmethod
from collections import namedtuple
import pandas as pd
import numpy as np
from typing import Tuple

bm1684x_tiu_reg = "./BM1686_TPU_TIU_Reg6.5.xlsx"
bm1684x_dma_reg = "./GDMA_1686_DES_REG_v7.7.xlsx"

reg_bdc = pd.read_excel(bm1684x_tiu_reg, sheet_name=None)
reg_dma = pd.read_excel(bm1684x_dma_reg, sheet_name=None)

bdc_cmd = dict()
dma_cmd = dict()

bank_size = 2**14

# cache for convert binary to unsigned integer.
table = 2 ** np.arange(64, dtype=np.uint64)


def packbits(arr):
    return int(arr.dot(table[: arr.size]))


def decode_cmd(buffer, reg_bdc):
    name = reg_bdc.iloc[:, 0]
    bits_width = np.cumsum(reg_bdc.iloc[:, 1])
    bits_sec = np.split(buffer, bits_width[:-1])
    value = (packbits(x) for x in bits_sec)
    return dict(zip(name, value))


def regist_bdc(sheet_name):
    def decorate(cls):
        pd = reg_bdc[sheet_name]
        setattr(cls, "reg_def", pd)
        setattr(cls, "len", pd.iloc[:, 1].sum())
        bdc_cmd.setdefault(cls.opcode, set()).add(cls)
        return cls

    return decorate


def regist_dma(sheet_name):
    def decorate(cls):
        pd = reg_dma[sheet_name]
        # filter out invalid recorder
        valid = ~pd.iloc[:, 1].isnull()
        pd = pd[valid].copy()
        pd.iloc[:, 1] = pd.iloc[:, 1].astype(int)
        setattr(cls, "reg_def", pd)
        setattr(cls, "len", pd.iloc[:, 1].sum())
        dma_cmd.setdefault(cls.opcode, set()).add(cls)
        return cls

    return decorate


def format_lmem(addr):
    i = addr // bank_size
    s = addr % bank_size
    if s == 0:
        return f"{i}"
    else:
        return f"{i}.{s}"


opopd = namedtuple("opopd", ["addr", "shape", "dtype"])


def fmt_opd(opd: opopd):
    s = [str(s) for s in opd.shape]
    return f"<{'x'.join(s)}x{opd.dtype}>"


# BDC definition
class bdc_base:
    len = 0
    description = "This is a base op."
    reg_def = None
    opcode = None
    eu_type = ()
    cmd_bits = (41, 45)
    eu_bits = (45, 50)
    short_cmd = False  # long_code by default

    def __init__(self, cmd_reg):
        cmd = cmd_reg[: self.len]
        self.cmd = cmd
        attr = decode_cmd(cmd, self.reg_def)
        self.attribute = attr
        self.cmd_id = attr["des_cmd_id"]
        self.cmd_id_dep = attr["des_cmd_id_dep"]

    @classmethod
    def is_comp_base(cls, cmd_reg):
        if cmd_reg.size < cls.len:
            return False
        if cls.short_cmd is not None and bool(cmd_reg[0]) != cls.short_cmd:
            return False
        if packbits(cmd_reg[cls.cmd_bits[0] : cls.cmd_bits[1]]) != cls.opcode:
            return False
        if packbits(cmd_reg[cls.eu_bits[0] : cls.eu_bits[1]]) not in cls.eu_type:
            return False
        return True

    @classmethod
    def is_comp(cls, cmd_reg):
        return cls.is_comp_base(cmd_reg)

    @abstractmethod
    def get_elt(self) -> Tuple[opopd, opopd, dict]:
        # (results, operands, attribute)
        pass

    def __repr__(self):
        if self.get_elt() is None:
            return self.description
        results, operands, attribute = self.get_elt()  # type: ignore
        res_str = [f"R{format_lmem(x.addr)}" for x in results]
        opd_str = [f"R{format_lmem(x.addr)}" for x in operands]
        res_sig = [fmt_opd(x) for x in results]
        opd_sig = [fmt_opd(x) for x in operands]
        op_name = self.eu_type[self.attribute["des_tsk_eu_typ"]]
        return (
            f"{', '.join(res_str)} = {op_name}.{self.cmd_id}"
            + f" ({', '.join(opd_str)}, {{{self.cmd_id_dep}}}) "
            + f"{{{attribute}}} : ({', '.join(opd_sig)}) -> ({', '.join(res_sig)})"
        )


@regist_bdc("CONV")
class conv_op(bdc_base):
    opcode = 0
    eu_type = {0: "conv.normal", 1: "conv.wrq", 2: "conv.wrqrelu"}
    description = "convolution"

    def get_elt(self):
        attr = self.attribute  # type: ignore
        des_shape_attr = ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w")
        opd0_shape_attr = ("des_opd0_c", "des_opd0_h", "des_opd0_w")
        res_shape = [attr[x] for x in des_shape_attr]
        opd0_shape = [attr[x] for x in opd0_shape_attr]
        opd1_shape = [attr[x] for x in ("des_opd1_h", "des_opd1_w")]
        res_addr = attr["des_res0_addr"]
        opd0_addr = attr["des_opd0_addr"]
        opd1_addr = attr["des_opd1_addr"]
        opd2_addr = attr["des_opd2_addr"]
        res = opopd(res_addr, res_shape, "f32")
        opd0 = opopd(opd0_addr, opd0_shape, "f32")
        opd1 = opopd(opd1_addr, opd1_shape, "f32")
        opd2 = opopd(opd2_addr, [], "f32")
        return ([res], [opd0, opd1, opd2], None)


@regist_bdc("sCONV")
class sconv_op(conv_op):
    short_cmd = True
    description = "short convolution"


@regist_bdc("MM")
class mm_op(bdc_base):
    opcode = 2
    eu_type = {1: "mm.normal", 2: "mm.wrq", 3: "mm.wrqrelu"}
    description = "matrix multiply"


@regist_bdc("sMM")
class smm_op(mm_op):
    short_cmd = True
    description = "short matrix multiply"

    def get_elt(self):
        attr = self.attribute  # type: ignore
        res_shape = [attr[x] for x in ("des_res0_c", "des_res0_w")]
        opd0_shape = [attr[x] for x in ("des_opd0_n", "des_opd0_c", "des_opd0_w")]
        opd1_shape = [attr["des_opd1_w"]]
        res_addr = attr["des_res0_addr"]
        opd0_addr = attr["des_opd0_addr"]
        opd1_addr = attr["des_opd1_addr"]
        res = opopd(res_addr, res_shape, "f32")
        opd0 = opopd(opd0_addr, opd0_shape, "f32")
        opd1 = opopd(opd1_addr, opd1_shape, "f32")
        return ([res], [opd0, opd1], None)


@regist_bdc("MM2")
class mm2_op(bdc_base):
    opcode = 2
    eu_type = (4, 5, 6)
    description = "matrix multiply2"


@regist_bdc("sMM2")
class smm2_op(mm2_op):
    short_cmd = True
    description = "short matrix multiply2"


@regist_bdc("CMP")
class cmp_op(bdc_base):
    opcode = 13
    eu_type = (22, 23, 24, 25, 26)
    description = "fused_cmpare"


@regist_bdc("sCMP")
class scmp_op(cmp_op):
    short_cmd = True
    description = "short fused_cmpare"


@regist_bdc("SFU")
class sfu_op(bdc_base):
    opcode = 9
    eu_type = (12, 13, 15, 17)
    description = "special_function"


@regist_bdc("sSFU")
class ssfu_op(sfu_op):
    short_cmd = True
    description = "short special_function"


@regist_bdc("VC")
class vc_op(bdc_base):
    opcode = 14
    eu_type = (0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 23)
    description = "vector correlation"


@regist_bdc("sVC")
class svc_op(vc_op):
    short_cmd = True
    description = "short vector correlation"


@regist_bdc("LIN")
class lin_op(bdc_base):
    opcode = 10
    eu_type = (1, 20, 21)
    description = "fused_linear"


@regist_bdc("sLIN")
class slin_op(lin_op):
    short_cmd = True
    description = "short fused_linear"


@regist_bdc("AR")
class ar_op(bdc_base):
    opcode = 3
    eu_type = {
        0: "arith.mul",
        1: "arith.not",
        2: "arith.add",
        3: "arith.sub",
        4: "arith.max",
        5: "arith.min",
        6: "arith.logicShift",
        7: "arith.and",
        8: "arith.or",
        9: "arith.xor",
        10: "arith.selectGreat",
        11: "arith.selectEqual",
        12: "arith.div",
        13: "arith.selectLess",
        14: "arith.dataConvert",
        15: "arith.addSatu",
        16: "arith.subSatu",
        18: "arith.mac",
        19: "arith.copy",
        20: "arith.mulSatu",
        21: "arith.arithShift",
        22: "arith.rotateShift",
        23: "arith.mulDHR",
        26: "arith.abs",
        27: "arith.fsubabs",
        28: "arith.copyMb",
        29: "arith.getFirstOne",
        30: "arith.getFirstZero",
    }
    description = "arithmetic"

    def get_elt(self):
        attr = self.attribute  # type: ignore
        des_shape_attr = ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w")
        res_shape = [attr[x] for x in des_shape_attr]
        res_addr = attr["des_res0_addr"]
        opd0_addr = attr["des_opd0_addr"]
        res = opopd(res_addr, res_shape, "f32")
        opd0 = opopd(opd0_addr, res_shape, "f32")
        return ([res], [opd0], None)


@regist_bdc("sAR")
class sar_op(ar_op):
    short_cmd = True
    description = "short arithmetic"


# @regist_bdc("SEG")
# class seg_op(bdc_base):
#     opcode = 3
#     eu_type = [17, 24, 25]
#     description = "arithmetic"


# @regist_bdc("sSEG")
# class sseg_op(seg_op):
#     short_cmd = True
#     description = "short arithmetic"


@regist_bdc("PorD")
class pord_op(bdc_base):
    opcode = 1
    eu_type = {
        0: "pord.depthwise",
        1: "pord.avgpooling",
        2: "pord.depthwiserelu",
        3: "pord.maxpooling",
        4: "pord.roiDepthwise",
        5: "pord.roiavgpooling",
        6: "pord.roimaxpooling",
    }
    description = "depthwise or pooling"

    def get_elt(self):
        attr = self.attribute  # type: ignore
        des_shape_attr = ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w")
        opd0_shape_attr = ("des_opd0_h", "des_opd0_w")
        res_shape = [attr[x] for x in des_shape_attr]
        opd0_shape = [attr[x] for x in opd0_shape_attr]
        opd1_shape = [attr[x] for x in ("des_opd1_h", "des_opd1_w")]
        res_addr = attr["des_res0_addr"]
        opd0_addr = attr["des_opd0_addr"]
        opd1_addr = attr["des_opd1_addr"]
        opd2_addr = attr["des_opd2_addr"]
        res = opopd(res_addr, res_shape, "f32")
        opd0 = opopd(opd0_addr, opd0_shape, "f32")
        opd1 = opopd(opd1_addr, opd1_shape, "f32")
        opd2 = opopd(opd2_addr, [], "f32")
        return ([res], [opd0, opd1, opd2], None)


@regist_bdc("sPorD")
class spord_op(pord_op):
    short_cmd = True
    description = "short depthwise or pooling"


@regist_bdc("RQ&DQ")
class rqdq_op(bdc_base):
    opcode = 4
    eu_type = range(5)
    description = "RQ && DQ"


@regist_bdc("sRQ&sDQ")
class srqdq_op(rqdq_op):
    short_cmd = True
    description = "short RQ && DQ"


@regist_bdc("SG")
class sg_op(bdc_base):
    opcode = 6
    eu_type = set(range(17)) - set([11, 12])
    description = "scatter_gather"


@regist_bdc("sSG")
class ssg_op(sg_op):
    short_cmd = True
    description = "short scatter_gather"


@regist_bdc("SGL")
class sgl_op(bdc_base):
    opcode = 6
    eu_type = (17, 18)
    description = "scatter_gather_line"


@regist_bdc("sSGL")
class ssgl_op(sgl_op):
    short_cmd = True
    description = "short scatter_gather_line"


@regist_bdc("TRANS&BC")
class transbc_op(bdc_base):
    opcode = 5
    eu_type = range(6)
    description = "TRANS && BC"


@regist_bdc("sTRANS&sBC")
class stransbc_op(transbc_op):
    short_cmd = True
    description = "short TRANS && BC"


@regist_bdc("LAR")
class lar_op(bdc_base):
    opcode = 7
    short_cmd = None
    eu_type = range(31)
    description = "linear_arithmetic"


# @regist_bdc("SYS")
# class sys_op(bdc_base):
#     opcode = 15
#     short_cmd = None
#     eu_type = (0, 1, 2, 3, 4, 5, 30, 31)
#     description = "system"


@regist_bdc("SYSID")
class sysid_op(bdc_base):
    opcode = 15
    short_cmd = None
    eu_type = (0, 1, 2, 3, 4, 5, 30, 31)
    description = "system"

    def __repr__(self):
        return "syncID"


# TPU1686/common/include/memmap.h
memmap = {
    "R": (int("0x8000000", 16), int("0x9000000", 16)),  # lmen_base 16M
    "S": (int("0x9000000", 16), int("0x9004000", 16)),  # static memory 16KB
    "L": (int("0x10000000", 16), int("0x9004000", 16)),  # L2 SRAM  2M
    "G": (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}


def format_addr(addr):
    for k, v in memmap.items():
        if addr >= v[0] and addr < v[1]:
            if k == "R":
                return f"R{format_lmem(addr - v[0])}"
            return f"{k}.{addr - v[0]}"


# GDMA definition
class dma_base:
    len = 0
    description = "This is a base op."
    reg_def = None
    opcode = None
    cmd_bits = (32, 36)
    short_cmd = False  # long_code by default

    def __init__(self, cmd_reg):
        cmd = cmd_reg[: self.len]
        self.cmd = cmd
        attr = decode_cmd(cmd, self.reg_def)
        self.attribute = attr
        self.cmd_id = attr["cmd_id"]
        self.cmd_id_dep = attr["cmd_id_dep"]

    @classmethod
    def is_comp_base(cls, cmd_reg):
        if cmd_reg.size < cls.len:
            return False
        if cls.short_cmd is not None and bool(cmd_reg[3]) != cls.short_cmd:
            return False
        if packbits(cmd_reg[cls.cmd_bits[0] : cls.cmd_bits[1]]) != cls.opcode:
            return False
        return True

    @classmethod
    def is_comp(cls, cmd_reg):
        return cls.is_comp_base(cmd_reg)

    def __repr__(self):
        return self.description


@regist_dma("DMA_tensor（0x000）")
class dma_tensor(dma_base):
    opcode = 0
    description = "DMA tensor"

    def __repr__(self):
        attr = self.attribute  # type: ignore
        src_shape = [
            attr[x] for x in ("src_nsize", "src_csize", "src_hsize", "src_wsize")
        ]
        dst_addr = attr["dst_start_addr_h8"] * 2**32 + attr["dst_start_addr_l32"]
        src_addr = attr["src_start_addr_h8"] * 2**32 + attr["src_start_addr_l32"]
        dst_addr = format_addr(dst_addr)
        src_addr = format_addr(src_addr)
        op_name = "dma.tensor"
        return (
            f"{dst_addr} = {op_name}.{self.cmd_id} "
            + f"({src_addr} {src_shape}, {{{self.cmd_id_dep}}})"
        )


@regist_dma("DMA_matrix")
class dma_matrix(dma_base):
    opcode = 1
    description = "DMA matrix"

    def __repr__(self):
        attr = self.attribute  # type: ignore
        src_shape = [
            attr[x] for x in ("src_nsize", "src_csize", "src_hsize", "src_wsize")
        ]
        dst_addr = attr["dst_start_addr_l8"] * 2**32 + attr["dst_start_addr_h32"]
        src_addr = attr["src_start_addr_l8"] * 2**32 + attr["src_start_addr_h32"]
        dst_addr = format_addr(dst_addr)
        src_addr = format_addr(src_addr)
        op_name = "dma.matrix"
        return (
            f"{dst_addr} = {op_name}.{self.cmd_id} "
            + f"({src_addr} {src_shape}, {{{self.cmd_id_dep}}})"
        )


@regist_dma("sDMA_matrix")
class sdma_matrix(dma_matrix):
    opcode = 1
    short_cmd = True
    description = "short DMA matrix"


@regist_dma("DMA_masked_select")
class dma_masked_select(dma_base):
    opcode = 2
    description = "DMA masked select"


@regist_dma("sDMA_masked_select ")
class sdma_masked_select(dma_masked_select):
    short_cmd = True
    description = "short DMA masked select"


@regist_dma("DMA_general")
class dma_general(dma_base):
    opcode = 3
    description = "DMA general"


@regist_dma("sDMA_general")
class sdma_general(dma_general):
    short_cmd = True
    description = "short DMA general"


@regist_dma("DMA_cw_transpose")
class dma_cw_transpose(dma_base):
    opcode = 4
    description = "DMA CW Transpose"


@regist_dma("DMA_nonzero")
class dma_nonzero(dma_base):
    opcode = 5
    description = "DMA nonzero"


@regist_dma("sDMA_nonzero")
class sdma_nonzero(dma_nonzero):
    short_cmd = True
    description = "short DMA nonzero"


@regist_dma("sDMA_sys")
class sdma_sys(dma_base):
    opcode = 6
    short_cmd = True
    description = "short DMA sys"


@regist_dma("DMA_gather")
class dma_gather(dma_base):
    opcode = 7
    description = "DMA gather"


@regist_dma("DMA_scatter")
class dma_scatter(dma_base):
    opcode = 8
    description = "DMA scatter"
