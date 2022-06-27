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
import numpy as np
from enum import Enum
from typing import Tuple
import regdef_1684x

# global data and type
# ------------------------------------------------------------
bdc_cmd = dict()
dma_cmd = dict()

bank_size = 2**14

# cache for convert binary to unsigned integer.
table = 2 ** np.arange(64, dtype=np.uint64)

# TPU1686/common/include/memmap.h
memmap = {
    "R": (int("0x8000000", 16), int("0x9000000", 16)),  # lmen_base 16M
    "S": (int("0x9000000", 16), int("0x9004000", 16)),  # static memory 16KB
    "L": (int("0x10000000", 16), int("0x9004000", 16)),  # L2 SRAM  2M
    "G": (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}

# for DType. Only the bits width is correct.
class DType(Enum):
    i8 = 0
    f16 = 1
    f32 = 2
    i16 = 3
    i32 = 4
    b16 = 5
    i64 = 6
    # offset 8
    u8 = i8 + 8  # type: ignore
    u16 = i16 + 8  # type: ignore
    u32 = i32 + 8  # type: ignore


def get_dtype(prec, sign=1):  # unsigned -> 0; sign -> 1
    if prec in (DType.f32.value, DType.b16.value, DType.f16.value):
        return DType(prec)
    return DType(prec + (sign == 0) * 8)


class MemRef:
    def __init__(self, addr, shape, dtype: DType, relative_addr=False):
        self.addr = addr
        self.shape = shape
        self.dtype = dtype
        self.relative_addr = relative_addr

    @property
    def addr_str(self):
        if self.relative_addr:
            return f"R{self.fmt_lmem(self.addr)}"
        for k, v in memmap.items():
            if self.addr >= v[0] and self.addr < v[1]:
                if k == "R":
                    return f"R{self.fmt_lmem(self.addr - v[0])}"
                return f"{k}.{self.addr - v[0]:,}".replace(",", "`")

    @property
    def shape_str(self):
        s = [str(s) for s in self.shape]
        return f"<{'x'.join(s)}x{self.dtype.name}>"

    def fmt_lmem(self, addr):
        i = addr // bank_size
        s = addr % bank_size
        if s == 0:
            return f"{i}"
        else:
            return f"{i}.{s}"

    def __repr__(self):
        return f"{self.addr_str} {self.shape_str}"


def attribute_builder(des_attr, reg_field):
    for key, value in reg_field.items():  # type: ignore
        if isinstance(value, str):
            yield key, des_attr[value]
        else:
            yield key, [des_attr[x] for x in value]


# ------------------------------------------------------------

# utility function
# ------------------------------------------------------------
def packbits(arr):
    return int(arr.dot(table[: arr.size]))


# ------------------------------------------------------------
# registry function
# ------------------------------------------------------------
def registry_base(cmd_type, sheet_name, cls):
    attr = regdef_1684x.reg_def[sheet_name]
    fields = [x[0] for x in attr]
    bits = [x[1] for x in attr]
    setattr(cls, "des_reg", {"fields": fields, "bits": bits})
    setattr(cls, "len", bits[-1])
    cmd_type.setdefault(cls.opcode, set()).add(cls)
    return cls


def registry_bdc(sheet_name):
    def decorate(cls):
        return registry_base(bdc_cmd, sheet_name, cls)

    return decorate


def registry_dma(sheet_name):
    def decorate(cls):
        return registry_base(dma_cmd, sheet_name, cls)

    return decorate


def decode_reg(buffer, des_reg):
    bits_sec = np.split(buffer, des_reg["bits"][:-1])
    value = (packbits(x) for x in bits_sec)
    return dict(zip(des_reg["fields"], value))


# ------------------------------------------------------------
# BDC definition
# ------------------------------------------------------------
class bdc_base:
    len = 0
    description = "This is a base op."
    des_reg = None
    opcode = None
    eu_type = ()
    cmd_bits = (41, 45)
    eu_bits = (45, 50)
    short_cmd = False  # long_code by default
    __slots__ = (
        "results",
        "operands",
        "attribute",
        "cmd",
        "des_attr",
        "cmd_id",
        "cmd_id_dep",
    )

    @classmethod
    def decode(cls, cmd_reg):
        cls = cls()
        cls.cmd = cmd_reg[: cls.len]
        cls.des_attr = decode_reg(cls.cmd, cls.des_reg)
        cls.cmd_id = cls.des_attr["des_cmd_id"]
        cls.cmd_id_dep = cls.des_attr["des_cmd_id_dep"]
        cls.set_elt()
        return cls

    def __is_comp_base(self, cmd_reg):
        if cmd_reg.size < self.len:
            return False
        if self.short_cmd is not None and bool(cmd_reg[0]) != self.short_cmd:
            return False
        if packbits(cmd_reg[self.cmd_bits[0] : self.cmd_bits[1]]) != self.opcode:
            return False
        if packbits(cmd_reg[self.eu_bits[0] : self.eu_bits[1]]) not in self.eu_type:
            return False
        return True

    @classmethod
    def is_comp(cls, cmd_reg):
        return cls.__is_comp_base(cls, cmd_reg)

    def set_elt(self):
        self.results = []
        self.operands = []
        self.attribute = {}

    def __memref(self, reg_field):
        for addr, shape, dtype in zip(*(reg_field[i::3] for i in range(3))):  # type: ignore
            addr = self.des_attr[addr]
            shape = [self.des_attr[x] for x in shape]
            if isinstance(dtype, str):
                dtype = get_dtype(self.des_attr[dtype])
            elif isinstance(dtype, tuple):
                _type, _sign = dtype
                dtype = get_dtype(self.des_attr[_type], self.des_attr[_sign])
            yield MemRef(addr, shape, dtype, True)

    def memref(self, reg_field):
        return list(self.__memref(reg_field))

    def set_attibute(self, reg_field):
        return dict(attribute_builder(self.des_attr, reg_field))

    def __repr__(self):
        if self.operands == []:
            return self.description
        res_str, res_shape_t = zip(*((x.addr_str, x.shape_str) for x in self.results))
        opd_str, opd_shape_t = zip(*((x.addr_str, x.shape_str) for x in self.operands))
        op_name = self.eu_type[self.des_attr["des_tsk_eu_typ"]]
        return (
            f"{', '.join(res_str)} = {op_name}.{self.cmd_id}"
            + f" ({', '.join(opd_str)}, {{{self.cmd_id_dep}}}) "
            + f"{self.attribute}".replace(":", " =").replace("'", "")
            + " : "
            + f"({', '.join(opd_shape_t)}) -> ({', '.join(res_shape_t)})"
        )


@registry_bdc("CONV")
class conv_op(bdc_base):
    opcode = 0
    eu_type = {0: "conv.normal", 1: "conv.wrq", 2: "conv.wrqrelu"}
    description = "convolution"


@registry_bdc("sCONV")
class sconv_op(conv_op):
    short_cmd = True
    description = "short convolution"

    def set_elt(self):
        results = (
            "des_res0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            "des_opt_res0_prec",
        )
        operands = (
            "des_opd0_addr",
            ("des_opd0_c", "des_opd0_h", "des_opd0_w"),
            ("des_opt_opd0_prec", "des_opt_opd0_sign"),
            "des_opd1_addr",
            ("des_opd1_h", "des_opd1_w"),
            ("des_opt_opd0_prec", "des_opt_opd1_sign"),
            "des_opd2_addr",
            ("des_res0_c",),
            ("des_opt_opd0_prec", "des_opt_opd2_sign"),
        )
        attribute = {
            "padding": (
                "des_opd0_up_pad",
                "des_opd0_dn_pad",
                "des_opd0_lf_pad",
                "des_opd0_rt_pad",
            ),
            "padding_mode": "des_pad_mode",
            "result_add": "des_opt_res_add",
            "ins0": ("des_opd0_x_ins0", "des_opd0_y_ins0"),
            "dilation": ("des_opd1_x_ins0", "des_opd1_y_ins0"),
        }
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = self.set_attibute(attribute)


@registry_bdc("MM")
class mm_op(bdc_base):
    opcode = 2
    eu_type = {1: "mm.normal", 2: "mm.wrq", 3: "mm.wrqrelu"}
    description = "matrix multiply"


@registry_bdc("sMM")
class smm_op(mm_op):
    short_cmd = True
    description = "short matrix multiply"

    def set_elt(self):
        results = ("des_res0_addr", ("des_res0_c", "des_res0_w"), "des_opt_res0_prec")
        operands = (
            "des_opd0_addr",
            ("des_opd0_n", "des_opd0_c", "des_opd0_w"),
            ("des_opt_opd0_prec", "des_opt_opd0_sign"),
            "des_opd1_addr",
            ("des_opd1_w",),
            ("des_opt_opd0_prec", "des_opt_opd1_sign"),
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


@registry_bdc("MM2")
class mm2_op(bdc_base):
    opcode = 2
    eu_type = (4, 5, 6)
    description = "matrix multiply2"


@registry_bdc("sMM2")
class smm2_op(mm2_op):
    short_cmd = True
    description = "short matrix multiply2"


@registry_bdc("CMP")
class cmp_op(bdc_base):
    opcode = 13
    eu_type = (22, 23, 24, 25, 26)
    description = "fused_cmpare"


@registry_bdc("sCMP")
class scmp_op(cmp_op):
    short_cmd = True
    description = "short fused_cmpare"


@registry_bdc("SFU")
class sfu_op(bdc_base):
    opcode = 9
    eu_type = (12, 13, 15, 17)
    description = "special_function"


@registry_bdc("sSFU")
class ssfu_op(sfu_op):
    short_cmd = True
    description = "short special_function"


@registry_bdc("VC")
class vc_op(bdc_base):
    opcode = 14
    eu_type = (0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 23)
    description = "vector correlation"


@registry_bdc("sVC")
class svc_op(vc_op):
    short_cmd = True
    description = "short vector correlation"


@registry_bdc("LIN")
class lin_op(bdc_base):
    opcode = 10
    eu_type = (1, 20, 21)
    description = "fused_linear"


@registry_bdc("sLIN")
class slin_op(lin_op):
    short_cmd = True
    description = "short fused_linear"


@registry_bdc("AR")
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
        14: "arith.cast",
        15: "arith.adds",
        16: "arith.subs",
        18: "arith.mac",
        19: "arith.copy",
        20: "arith.muls",
        21: "arith.ashift",  # Arithmetic shift
        22: "arith.cshift",  # Circular shift
        23: "arith.mulDHR",
        26: "arith.abs",
        27: "arith.fsubabs",
        28: "arith.copyMb",
        29: "arith.getFirstOne",
        30: "arith.getFirstZero",
    }
    description = "arithmetic"


@registry_bdc("sAR")
class sar_op(ar_op):
    short_cmd = True
    description = "short arithmetic"

    def set_elt(self):
        results = (
            "des_res0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            "des_opt_res0_prec",
        )
        operands = (
            "des_opd0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            ("des_opt_opd0_prec", "des_opt_opd0_sign"),
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


# @registry_bdc("SEG")
# class seg_op(bdc_base):
#     opcode = 3
#     eu_type = [17, 24, 25]
#     description = "arithmetic"


# @registry_bdc("sSEG")
# class sseg_op(seg_op):
#     short_cmd = True
#     description = "short arithmetic"


@registry_bdc("PorD")
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


@registry_bdc("sPorD")
class spord_op(pord_op):
    short_cmd = True
    description = "short depthwise or pooling"

    def set_elt(self):
        results = (
            "des_res0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            "des_opt_res0_prec",
        )
        operands = (
            "des_opd0_addr",
            ("des_opd0_h", "des_opd0_w"),
            ("des_opt_opd0_prec", "des_opt_opd0_sign"),
            "des_opd1_addr",
            ("des_opd1_h", "des_opd1_w"),
            ("des_opt_opd0_prec", "des_opt_opd1_sign"),
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


@registry_bdc("RQ&DQ")
class rqdq_op(bdc_base):
    opcode = 4
    eu_type = {
        0: "quant.rq0",
        1: "quant.rq1",
        2: "quant.rq2",
        3: "quant.dq0",
        4: "quant.dq1",
        5: "quant.dq2",
    }
    description = "RQ && DQ"


@registry_bdc("sRQ&sDQ")
class srqdq_op(rqdq_op):
    short_cmd = True
    description = "short RQ && DQ"

    def set_elt(self):
        results = (
            "des_res0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            ("des_opt_res0_prec", "des_opt_opd2_sign"),
        )
        operands = (
            "des_opd0_addr",
            ("des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w"),
            ("des_opt_opd0_prec", "des_opt_opd0_sign"),
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


@registry_bdc("SG")
class sg_op(bdc_base):
    opcode = 6
    eu_type = set(range(17)) - set([11, 12])
    description = "scatter_gather"


@registry_bdc("sSG")
class ssg_op(sg_op):
    short_cmd = True
    description = "short scatter_gather"


@registry_bdc("SGL")
class sgl_op(bdc_base):
    opcode = 6
    eu_type = (17, 18)
    description = "scatter_gather_line"


@registry_bdc("sSGL")
class ssgl_op(sgl_op):
    short_cmd = True
    description = "short scatter_gather_line"


@registry_bdc("TRANS&BC")
class transbc_op(bdc_base):
    opcode = 5
    eu_type = range(6)
    description = "TRANS && BC"


@registry_bdc("sTRANS&sBC")
class stransbc_op(transbc_op):
    short_cmd = True
    description = "short TRANS && BC"


@registry_bdc("LAR")
class lar_op(bdc_base):
    opcode = 7
    short_cmd = None
    eu_type = range(31)
    description = "linear_arithmetic"


# @registry_bdc("SYS")
# class sys_op(bdc_base):
#     opcode = 15
#     short_cmd = None
#     eu_type = (0, 1, 2, 3, 4, 5, 30, 31)
#     description = "system"


@registry_bdc("SYSID")
class sysid_op(bdc_base):
    opcode = 15
    short_cmd = None
    eu_type = (0, 1, 2, 3, 4, 5, 30, 31)
    description = "system"

    def __repr__(self):
        return "syncID"


# ------------------------------------------------------------
# GDMA definition
# ------------------------------------------------------------
class dma_base:
    len = 0
    description = "This is a base op."
    des_reg = None
    opcode = None
    cmd_bits = (32, 36)
    op_name = "GDMA"
    short_cmd = False  # long_code by default
    __slots__ = (
        "results",
        "operands",
        "attribute",
        "cmd",
        "des_attr",
        "cmd_id",
        "cmd_id_dep",
    )

    @classmethod
    def decode(cls, cmd_reg):
        cls = cls()
        cmd = cmd_reg[: cls.len]
        cls.cmd = cmd
        attr = decode_reg(cmd, cls.des_reg)
        cls.des_attr = attr
        cls.cmd_id = attr["cmd_id"]
        cls.cmd_id_dep = attr["cmd_id_dep"]
        cls.set_elt()
        return cls

    def __is_comp_base(self, cmd_reg):
        if cmd_reg.size < self.len:
            return False
        if self.short_cmd is not None and bool(cmd_reg[3]) != self.short_cmd:
            return False
        if packbits(cmd_reg[self.cmd_bits[0] : self.cmd_bits[1]]) != self.opcode:
            return False
        return True

    @classmethod
    def is_comp(cls, cmd_reg):
        return cls.__is_comp_base(cls, cmd_reg)

    def set_elt(self):
        self.results = []
        self.operands = []
        self.attribute = {}

    def __memref(self, reg_field):
        for addr, shape, dtype in zip(*(reg_field[i::3] for i in range(3))):  # type: ignore
            h8, l32 = addr
            addr = self.des_attr[h8] * 2**32 + self.des_attr[l32]
            shape = [self.des_attr[x] for x in shape]
            dtype = get_dtype(self.des_attr[dtype])
            yield MemRef(addr, shape, dtype, False)

    def memref(self, reg_field):
        return list(self.__memref(reg_field))

    def __repr__(self):
        if self.operands == []:
            return self.description
        res_str = [x.addr_str for x in self.results]
        opd_str, opd_shape_t = zip(*((x.addr_str, x.shape_str) for x in self.operands))
        return (
            f"{', '.join(res_str)} = {self.op_name}.{self.cmd_id}"
            + f" ({', '.join(opd_str)}, {{{self.cmd_id_dep}}}) "
            + f"{{{self.attribute}}} : {opd_shape_t[0]}"
        )


@registry_dma("DMA_tensor（0x000）")
class dma_tensor(dma_base):
    opcode = 0
    op_name = "dma.tensor"
    description = "DMA tensor"

    def set_elt(self):
        results = (
            ("dst_start_addr_h8", "dst_start_addr_l32"),
            [],
            "src_data_format",
        )
        operands = (
            ("src_start_addr_h8", "src_start_addr_l32"),
            ("src_nsize", "src_csize", "src_hsize", "src_wsize"),
            "src_data_format",
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


@registry_dma("DMA_matrix")
class dma_matrix(dma_base):
    opcode = 1
    op_name = "dma.matrix"
    description = "DMA matrix"

    def set_elt(self):
        results = (
            ("dst_start_addr_l8", "dst_start_addr_h32"),
            [],
            "src_data_format",
        )
        operands = (
            ("src_start_addr_l8", "src_start_addr_h32"),
            ("src_nsize", "src_csize", "src_hsize", "src_wsize"),
            "src_data_format",
        )
        self.results = self.memref(results)
        self.operands = self.memref(operands)
        self.attribute = None


@registry_dma("sDMA_matrix")
class sdma_matrix(dma_matrix):
    opcode = 1
    short_cmd = True
    description = "short DMA matrix"


@registry_dma("DMA_masked_select")
class dma_masked_select(dma_base):
    opcode = 2
    description = "DMA masked select"


@registry_dma("sDMA_masked_select ")
class sdma_masked_select(dma_masked_select):
    short_cmd = True
    description = "short DMA masked select"


@registry_dma("DMA_general")
class dma_general(dma_base):
    opcode = 3
    description = "DMA general"


@registry_dma("sDMA_general")
class sdma_general(dma_general):
    short_cmd = True
    description = "short DMA general"


@registry_dma("DMA_cw_transpose")
class dma_cw_transpose(dma_base):
    opcode = 4
    description = "DMA CW Transpose"


@registry_dma("DMA_nonzero")
class dma_nonzero(dma_base):
    opcode = 5
    description = "DMA nonzero"


@registry_dma("sDMA_nonzero")
class sdma_nonzero(dma_nonzero):
    short_cmd = True
    description = "short DMA nonzero"


@registry_dma("sDMA_sys")
class sdma_sys(dma_base):
    opcode = 6
    short_cmd = True
    description = "short DMA sys"


@registry_dma("DMA_gather")
class dma_gather(dma_base):
    opcode = 7
    description = "DMA gather"


@registry_dma("DMA_scatter")
class dma_scatter(dma_base):
    opcode = 8
    description = "DMA scatter"
