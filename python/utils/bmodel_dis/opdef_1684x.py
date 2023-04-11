# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# keep This file clean and neat.

import numpy as np
from . import regdef_1684x
from .opparam_1684x import NamedDict, opparam_converter

# global data and type
# ------------------------------------------------------------
bdc_cmd = dict()
dma_cmd = dict()

bank_size = 2**14

# cache for convert binary to unsigned integer.
table = 2 ** np.arange(64, dtype=np.uint64)


def attribute_builder(attr, reg_field):
    for key, value in reg_field.items():  # type: ignore
        if isinstance(value, str):
            yield key, attr[value]
        else:
            yield key, [attr[x] for x in value]


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
    if sheet_name in opparam_converter:
        setattr(cls, "set_elt", staticmethod(opparam_converter[sheet_name]))
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
        "attr",
        "cmd_id",
        "cmd_id_dep",
    )

    def __eq__(self, other):
        if len(self.cmd) != len(other.cmd):
            return False
        return (self.cmd == other.cmd).all()

    def __hash__(self):
        return hash(str(self.cmd))

    @classmethod
    def decode(cls, cmd_reg):
        cls = cls()
        cls.cmd = cmd_reg[: cls.len]
        attr = NamedDict(decode_reg(cls.cmd, cls.des_reg), ("des_", "short_", "opt_"))
        cls.cmd_id = attr.cmd_id
        cls.cmd_id_dep = attr.cmd_id_dep
        cls.attr = attr
        cls.results, cls.attribute, cls.operands = cls.set_elt(attr)
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

    def set_elt(self, _):
        return ([],) * 3

    def __repr__(self):
        if self.operands == []:
            return self.description
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        op_name = self.eu_type[self.attr.tsk_eu_typ]
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %B{self.cmd_id} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %D{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
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


@registry_bdc("MM")
class mm_op(bdc_base):
    opcode = 2
    eu_type = {1: "mm.normal", 2: "mm.wrq", 3: "mm.wrqrelu"}
    description = "matrix multiply"


@registry_bdc("sMM")
class smm_op(mm_op):
    short_cmd = True
    description = "short matrix multiply"


@registry_bdc("MM2")
class mm2_op(bdc_base):
    opcode = 2
    eu_type = {4: "mm2.nn", 5: "mm2.nt", 6: "mm2.tt"}
    description = "matrix multiply2"


@registry_bdc("sMM2")
class smm2_op(mm2_op):
    short_cmd = True
    description = "short matrix multiply2"


@registry_bdc("CMP")
class cmp_op(bdc_base):
    opcode = 13
    eu_type = {
        22: "cmp.gt_and_sel",
        23: "cmp.sel_gt",
        24: "cmp.sel_eq",
        25: "cmp.lt_and_sel",
        26: "cmp.sel_lt",
    }
    description = "fused_cmpare"


@registry_bdc("sCMP")
class scmp_op(cmp_op):
    short_cmd = True
    description = "short fused_cmpare"


@registry_bdc("SFU")
class sfu_op(bdc_base):
    opcode = 9
    eu_type = {
        12: "sfu.tailor_4x",
        13: "sfu.tailor",
        15: "sfu.normalize",
        17: "sfu.rsqrt",
    }
    description = "special_function"


@registry_bdc("sSFU")
class ssfu_op(sfu_op):
    short_cmd = True
    description = "short special_function"


@registry_bdc("VC")
class vc_op(bdc_base):
    opcode = 14
    eu_type = {
        0: "vc.mul",
        2: "vc.add",
        3: "vc.sub",
        4: "vc.max",
        5: "vc.min",
        7: "vc.and",
        8: "vc.or",
        9: "vc.xor",
        10: "vc.select_gt",
        11: "vc.select_eq",
        12: "vc.div",
        13: "vc.select_lt",
        15: "vc.add_satu",
        16: "vc.sub_satu",
        20: "vc.mul_satu",
        23: "vc.mulDHR",
    }
    description = "vector correlation"


@registry_bdc("sVC")
class svc_op(vc_op):
    short_cmd = True
    description = "short vector correlation"


@registry_bdc("LIN")
class lin_op(bdc_base):
    opcode = 10
    eu_type = {1: "lin.mac", 20: "lin.square_sum", 21: "lin.square_diff"}
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
        24: "arith.euIdxGen",  # not in HW.spec
        25: "arith.npuIdxGen",  # not in HW.spec
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
        4: "pord.maxpooling",
        5: "pord.roiDepthwise",
        6: "pord.roiavgpooling",
        7: "pord.roimaxpooling",
    }
    description = "depthwise or pooling"


@registry_bdc("sPorD")
class spord_op(pord_op):
    short_cmd = True
    description = "short depthwise or pooling"


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


@registry_bdc("SG")
class sg_op(bdc_base):
    opcode = 6
    eu_type = {
        0: "sg.pl_gather_d1coor",
        1: "sg.pl_gather_d2coor",
        2: "sg.pl_gather_rec",
        3: "sg.pl_scatter_d1coor",
        4: "sg.pl_scatter_d2coor",
        5: "sg.pe_s_gather_d1coor",
        6: "sg.pe_s_scatter_d1coor",
        7: "sg.pe_m_gather_d1coor",
        8: "sg.pe_s_mask_select",
        9: "sg.pe_s_nonzero",
        10: "sg.pe_s_scatter_pp_d1coor",
        11: "sg.pl_gather_perw",
        12: "sg.pl_scatter_perw",
        13: "sg.pe_s_gather_hzd",
        14: "sg.pe_s_scatter_hzd",
        15: "sg.pe_s_mask_selhzd",
        16: "sg.pe_s_nonzero_hzd",
    }
    description = "scatter_gather"


@registry_bdc("sSG")
class ssg_op(sg_op):
    short_cmd = True
    description = "short scatter_gather"


@registry_bdc("SGL")
class sgl_op(bdc_base):
    opcode = 6
    eu_type = {17: "sgl.pe_s_gather_line", 18: "sgl.pe_s_scatter_line"}
    description = "scatter_gather_line"


@registry_bdc("sSGL")
class ssgl_op(sgl_op):
    short_cmd = True
    description = "short scatter_gather_line"


@registry_bdc("TRANS&BC")
class transbc_op(bdc_base):
    opcode = 5
    eu_type = {
        0: "tsbc.cw_ts",
        1: "tsbc.wc_ts",
        2: "tsbc.l_copy",
        3: "tsbc.l_bc",
        4: "tsbc.s_bc",
        5: "tsbc.s_distribute",
    }
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
    fun_bits = (36, 39)
    sp_fun = ()
    op_name = "GDMA"
    short_cmd = False  # long_code by default
    __slots__ = (
        "results",
        "operands",
        "attribute",
        "cmd",
        "attr",
        "cmd_id",
        "cmd_id_dep",
    )

    def __eq__(self, other):
        if len(self.cmd) != len(other.cmd):
            return False
        return (self.cmd == other.cmd).all()

    def __hash__(self):
        return hash(str(self.cmd))

    @classmethod
    def decode(cls, cmd_reg):
        cls = cls()
        cmd = cmd_reg[: cls.len]
        cls.cmd = cmd
        attr = NamedDict(decode_reg(cmd, cls.des_reg))
        cls.cmd_id = attr.cmd_id
        cls.cmd_id_dep = attr.cmd_id_dep
        cls.attr = attr
        cls.results, cls.attribute, cls.operands = cls.set_elt(attr)
        return cls

    def __is_comp_base(self, cmd_reg):
        if cmd_reg.size < self.len:
            return False
        if self.short_cmd is not None and bool(cmd_reg[3]) != self.short_cmd:
            return False
        if packbits(cmd_reg[self.cmd_bits[0] : self.cmd_bits[1]]) != self.opcode:
            return False
        sp_fun_id = packbits(cmd_reg[self.fun_bits[0] : self.fun_bits[1]])
        if self.sp_fun and (sp_fun_id not in self.sp_fun):
            return False
        return True

    @classmethod
    def is_comp(cls, cmd_reg):
        return cls.__is_comp_base(cls, cmd_reg)

    def set_elt(self, _):
        return ([],) * 3

    def __repr__(self):
        if self.operands == []:
            return self.description
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        if self.sp_fun:
            op_name = self.sp_fun[self.attr.cmd_special_function]
        else:
            op_name = self.op_name
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %D{self.cmd_id} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %B{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )


@registry_dma("DMA_tensor（0x000）")
class dma_tensor(dma_base):
    opcode = 0
    op_name = "dma.tensor"
    sp_fun = {
        0: "dma.tensor",
        1: "dma.tensor.transpose",
        2: "dma.tensor.collect",
        3: "dma.tensor.broadcast",
        4: "dma.tensor.distribute",
        5: "dma.tensor.4bank_copy",
        6: "dma.tensor.4bank_broadcast",
    }
    description = "DMA tensor"


@registry_dma("DMA_matrix")
class dma_matrix(dma_base):
    opcode = 1
    op_name = "dma.matrix"
    sp_fun = {
        0: "dma.matrix",
        1: "dma.matrix.transpose",
    }
    description = "DMA matrix"


@registry_dma("sDMA_matrix")
class sdma_matrix(dma_matrix):
    opcode = 1
    short_cmd = True
    description = "short DMA matrix"


@registry_dma("DMA_masked_select")
class dma_masked_select(dma_base):
    opcode = 2
    op_name = "dma.masked_select"
    description = "DMA masked select"


@registry_dma("sDMA_masked_select ")
class sdma_masked_select(dma_masked_select):
    short_cmd = True
    description = "short DMA masked select"


@registry_dma("DMA_general")
class dma_general(dma_base):
    opcode = 3
    op_name = "dma.general"
    sp_fun = {
        0: "dma.general",
        1: "dma.general.broadcast",
    }
    description = "DMA general"


@registry_dma("sDMA_general")
class sdma_general(dma_general):
    short_cmd = True
    description = "short DMA general"


@registry_dma("DMA_cw_transpose")
class dma_cw_transpose(dma_base):
    opcode = 4
    op_name = "dma.cw_transpose"
    description = "DMA CW Transpose"


@registry_dma("DMA_nonzero")
class dma_nonzero(dma_base):
    opcode = 5
    op_name = "dma.nonzero"
    description = "DMA nonzero"


@registry_dma("sDMA_nonzero")
class sdma_nonzero(dma_nonzero):
    short_cmd = True
    description = "short DMA nonzero"


@registry_dma("sDMA_sys")
class sdma_sys(dma_base):
    opcode = 6
    short_cmd = True
    op_name = "dma.sys"
    sp_fun = {
        0: "dma.sys",
        1: "dma.sys.nop",
    }
    description = "short DMA sys"


@registry_dma("DMA_gather")
class dma_gather(dma_base):
    opcode = 7
    op_name = "gdma.gather"
    description = "DMA gather"


@registry_dma("DMA_scatter")
class dma_scatter(dma_base):
    opcode = 8
    op_name = "gdma.scatter"
    description = "DMA scatter"
