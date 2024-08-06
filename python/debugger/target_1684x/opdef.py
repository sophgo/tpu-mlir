# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Dict, Tuple

from ..target_common import OpInfo, atomic_reg, BaseTpuCmd, Tiu, Dma, ALIGN, RegIndex
from .memmap import *
from .regdef import *
from .opparam import opparam_converter, default_converter

tiu_cls = dict()
dma_cls = dict()
tiu_index = RegIndex()
dma_index = RegIndex()


class TiuCmd(BaseTpuCmd, Tiu):
    opparam_converter = opparam_converter
    default_converter = default_converter

    name: str = None
    description = "TIU Operation."
    # extension
    eu_type = {}
    short_cmd = False  # long_code by default

    def __init__(self, reg: atomic_reg, *, buf: memoryview, subnet_id=0) -> None:
        super().__init__(reg, buf=buf, subnet_id=subnet_id)
        self.eu_name = tiu_cls[reg.OP_NAME]["tsk_eu_typ"][reg.tsk_eu_typ]

    def __init_subclass__(cls) -> None:
        if len(cls.eu_type) == 0:
            cls.eu_type = {0: "none"}
        if cls.short_cmd is None:
            tiu_index[(0, cls.opcode, cls.eu_type)] = cls
            tiu_index[(1, cls.opcode, cls.eu_type)] = cls
        else:
            tiu_index[(cls.short_cmd, cls.opcode, cls.eu_type)] = cls

        tiu_cls[cls.name] = {
            "description": cls.description,
            "tsk_eu_typ": cls.eu_type,
            "tsk_typ": cls.opcode,
            "short_cmd": cls.short_cmd,
        }
        return cls

    def __repr__(self) -> str:
        if self.operands == []:
            return tiu_cls[self.reg.OP_NAME]["description"]

        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))

        attribute_dic = {}
        if self.attribute:
            attribute_dic.update(self.attribute)

        op_name = self.op_name

        attribute = f"{attribute_dic}" if len(attribute_dic) > 0 else ""
        attribute = f" {attribute}".replace(":", " =").replace("'", "")
        return (
            f"{', '.join(res_name)}, %B{self.cmd_id} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %D{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
        )

    @property
    def op_name(self):
        op_info = tiu_cls[self.reg.OP_NAME]
        eu_type_id = self.reg["tsk_eu_typ"]
        op_name = self.reg.OP_NAME
        if len(op_info["tsk_eu_typ"]) != 0:
            op_name = op_info["tsk_eu_typ"][eu_type_id]
        return op_name

    def ops(self, reg, is_arch):
        raise NotImplementedError()

    @property
    def cmd_id_dep(self):
        return self.reg.cmd_id_dep

    @property
    def cmd_id(self):
        return self.reg.cmd_id


class DmaCmd(BaseTpuCmd, Dma):
    opparam_converter = opparam_converter
    default_converter = default_converter

    name: str = None
    description = "GDMA Operation."
    sp_fun = {}
    short_cmd = False  # long_code by default

    def __init__(self, reg: atomic_reg, *, buf: memoryview, subnet_id=0) -> None:
        super().__init__(reg, buf=buf, subnet_id=subnet_id)

    def __init_subclass__(cls) -> None:
        if len(cls.sp_fun) == 0:
            cls.sp_fun = {0: "none"}
        dma_index[(cls.short_cmd, cls.opcode, cls.sp_fun)] = cls
        dma_cls[cls.name] = {
            "description": cls.description,
            "tsk_typ": cls.opcode,
            "sp_fun": cls.sp_fun,
            "short_cmd": cls.short_cmd,
        }
        return cls

    def __repr__(self):
        if self.operands == []:
            return dma_cls[self.reg.OP_NAME]["description"]

        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))

        attribute_dic = {}
        if self.attribute:
            attribute_dic.update(self.attribute)

        op_name = self.op_name
        attribute = f"{attribute_dic}" if len(attribute_dic) > 0 else ""
        attribute = f" {attribute}".replace(":", " =").replace("'", "")
        return (
            f"{', '.join(res_name)}, %D{self.cmd_id} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %B{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )

    @property
    def op_name(self):
        op_info = dma_cls[self.reg.OP_NAME]
        sp_func_id = self.reg["cmd_special_function"]
        op_name = self.reg.OP_NAME
        if len(op_info["sp_fun"]) != 0:
            op_name = op_info["sp_fun"][sp_func_id]
        return op_name

    def ops(self, reg, is_arch):
        return 0

    @property
    def cmd_id_dep(self):
        return self.reg.cmd_id_dep

    @property
    def cmd_id(self):
        return self.reg.cmd_id


class conv_op(TiuCmd):
    name = "CONV"
    opcode = 0
    eu_type = {0: "conv.normal", 1: "conv.wrq", 2: "conv.wrqrelu"}
    description = "convolution"

    def ops(self, reg, is_arch=False):
        n, ic, ih, iw = self.operands[0].shape
        n, oc, oh, ow = self.results[0].shape
        # remains_hw = 0

        has_bias = len(self.operands) > 2
        if is_arch:
            dtype = self.operands[0].dtype
            ic = ALIGN(ic, info.NPU_NUM)
            ow = ALIGN(oh * ow, info.EU_NUM(dtype))
            oh = 1
            oc = ALIGN(oc, info.NPU_NUM)
            # iw = ALIGN(ih * iw, info.EU_NUM(dtype))
            # ih = 1
            # kw = ALIGN(kw, 64)
            # remain_hw = ALIGN(ih*iw, info.EU_NUM) - ih*iw
        out_size = n * oc * oh * ow
        kh, kw = self.attribute["kernel"]
        return out_size * (2 * ic * kh * kw - 1 + has_bias)


class sconv_op(conv_op):
    name = "sCONV"
    short_cmd = True
    description = "short convolution"


class mm_op(TiuCmd):
    name = "MM"
    opcode = 2
    eu_type = {1: "mm.normal", 2: "mm.wrq", 3: "mm.wrqrelu"}
    description = "matrix multiply"

    def ops(self, reg: MM_reg, is_arch=False):
        m, lk = self.operands[0].shape
        rk, n = self.operands[1].shape
        if self.attribute["l_trans"]:
            lk, m = m, lk
        assert lk == rk
        k = lk
        res_add = reg.res_add
        has_bias = len(self.operands) > 2
        if is_arch:
            dtype = self.operands[0].dtype
            # align the column of B
            n = ALIGN(reg.res0_c, info.NPU_NUM) * ALIGN(reg.res0_w, info.EU_NUM(dtype))
        return m * n * (2 * k - 1 + has_bias + res_add)


class smm_op(mm_op):
    name = "sMM"
    short_cmd = True
    description = "short matrix multiply"


class mm2_op(TiuCmd):
    name = "MM2"
    opcode = 2
    eu_type = {4: "mm2.nn", 5: "mm2.nt", 6: "mm2.tt"}
    description = "matrix multiply2"

    def ops(self, reg, is_arch=False):
        m, lk = self.operands[0].shape
        rk, n = self.operands[1].shape
        if self.eu_name == "mm2.nt":
            rk, n = n, rk
        elif self.eu_name == "mm2.tt":
            rk, n = n, rk
            lk, m = m, lk

        assert lk == rk
        k = lk
        has_bias = len(self.operands) > 2

        if is_arch:
            dtype = self.operands[0].dtype
            k = ALIGN(k, info.EU_NUM(dtype))
            m = ALIGN(m, info.NPU_NUM)

        return m * n * (2 * k - 1 + has_bias)


class smm2_op(mm2_op):
    name = "sMM2"
    short_cmd = True
    description = "short matrix multiply2"


class cmp_op(TiuCmd):
    name = "CMP"
    opcode = 13
    eu_type = {
        22: "cmp.gt_and_sel",
        23: "cmp.sel_gt",
        24: "cmp.sel_eq",
        25: "cmp.lt_and_sel",
        26: "cmp.sel_lt",
    }
    description = "fused_cmpare"

    def ops(self, reg, is_arch=False):
        n, c, h, w = self.results[0].shape
        # res_num = len(self.results)

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(h * w, info.EU_NUM(dtype))
        return n * c * hw * 2


class scmp_op(cmp_op):
    name = "sCMP"
    short_cmd = True
    description = "short fused_cmpare"


class sfu_op(TiuCmd):
    name = "SFU"
    opcode = 9
    eu_type = {
        12: "sfu.taylor_4x",
        13: "sfu.taylor",
        15: "sfu.normalize",
        17: "sfu.rsqrt",
    }
    description = "special_function"

    def ops(self, reg: SFU_reg, is_arch=False):
        n, c, h, w = self.operands[0].shape
        factor = 1
        res_num = len(self.results)
        if self.eu_name == "sfu.taylor_4x" or self.eu_name == "sfu.taylor":
            factor = 2 * reg.opd1_n - 1  # 2* table_len -1
        elif self.eu_name == "sfu.normalize":
            factor = 1
        elif self.eu_name == "sfu.rsqrt":
            factor = reg.opd2_n_str + 1  # iteration times

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(w * h, info.EU_NUM(dtype))

        return res_num * n * c * hw * factor


class ssfu_op(sfu_op):
    name = "sSFU"
    short_cmd = True
    description = "short special_function"


class vc_op(TiuCmd):
    name = "VC"
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

    def ops(self, reg, is_arch):
        n, c, h, w = self.results[0].shape
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w


class svc_op(vc_op):
    name = "sVC"
    short_cmd = True
    description = "short vector correlation"


class lin_op(TiuCmd):
    name = "LIN"
    opcode = 10
    eu_type = {1: "lin.mac", 20: "lin.square_sum", 21: "lin.square_diff"}
    description = "fused_linear"

    def ops(self, reg, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 2
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor


class slin_op(lin_op):
    name = "sLIN"
    short_cmd = True
    description = "short fused_linear"


class ar_op(TiuCmd):
    name = "AR"
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

    def ops(self, reg, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        hw = h * w
        if self.eu_name == "arith.div":
            factor = 5  # TODO: fix the factor
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(w * h, info.EU_NUM(dtype))
        return n * c * hw * factor


class sar_op(ar_op):
    name = "sAR"
    short_cmd = True
    description = "short arithmetic"


class pord_op(TiuCmd):
    name = "PorD"
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

    def ops(self, reg: PorD_reg, is_arch):
        n, c, h, w = self.results[0].shape
        kh, kw = reg.opd1_h, reg.opd1_w
        factor = 1
        if self.eu_name == "pord.avgpooling" or self.eu_name == "pord.maxpooling":
            factor = len(self.results)  # TODO: fix the factor
        elif self.eu_name == "pord.depthwise":
            factor = 2
        elif self.eu_name == "pord.depthwiserelu":
            factor = 3
        else:
            # roi_pooling
            kh = 1
            kw = 1
            factor = 2 * 4 - 1  # bilinar, ignore coords generate
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(h * w, info.EU_NUM(dtype))
            h = 1
        return n * c * h * w * (factor * kh * kw - 1)


class spord_op(pord_op):
    name = "sPorD"
    short_cmd = True
    description = "short depthwise or pooling"


class rqdq_op(TiuCmd):
    name = "RQ&DQ"
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

    def ops(self, _, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 3  # mul, add, shift
        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(h * w, info.EU_NUM(dtype))
        return n * c * hw * factor


class srqdq_op(rqdq_op):
    name = "sRQ&sDQ"
    short_cmd = True
    description = "short RQ && DQ"


class sg_op(TiuCmd):
    name = "SG"
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

    def ops(self, _, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor


class ssg_op(sg_op):
    name = "sSG"
    short_cmd = True
    description = "short scatter_gather"


class sgl_op(TiuCmd):
    name = "SGL"
    opcode = 6
    eu_type = {17: "sgl.pe_s_gather_line", 18: "sgl.pe_s_scatter_line"}
    description = "scatter_gather_line"

    def ops(self, reg: SGL_reg, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor


class ssgl_op(sgl_op):
    name = "sSGL"
    short_cmd = True
    description = "short scatter_gather_line"


class transbc_op(TiuCmd):
    name = "TRANS&BC"
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

    def ops(self, reg: TRANS_BC_reg, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            if self.eu_name in (
                "tsbc.l_copy",
                "tsbc.l_bc",
                "tsbc.s_bc",
                "tsbc.s_distribute",
            ):
                hw = ALIGN(h * w, info.EU_NUM(dtype))
            else:
                hw = h * ALIGN(w, info.EU_NUM(dtype))
        return n * c * hw * factor


class stransbc_op(transbc_op):
    name = "sTRANS&sBC"
    short_cmd = True
    description = "short TRANS && BC"


class lar_op(TiuCmd):
    name = "LAR"
    opcode = 7
    short_cmd = None
    eu_type = range(31)
    description = "linear_arithmetic"

    def ops(self, reg: LAR_reg, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor


class tiu_sys(TiuCmd):
    name = "SYSID"
    opcode = 15
    short_cmd = None
    eu_type = {
        0: "sys.intr_barrier",
        1: "sys.spb",
        2: "sys.swr",
        3: "sys.swr_from_lmem",
        4: "sys.swr_collect_from_lmem",
        5: "sys.data_barrier",
        30: "sys.nop",
        31: "sys.end",
    }
    description = "system"

    def ops(self, reg, is_arch):
        return 1

    def __repr__(self):
        return self.eu_name


# ------------------------------------------------------------
# GDMA definition
# ------------------------------------------------------------


class dma_tensor(DmaCmd):
    name = "DMA_tensor（0x000）"
    opcode = 0
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


class dma_matrix(DmaCmd):
    name = "DMA_matrix"
    opcode = 1
    sp_fun = {
        0: "dma.matrix",
        1: "dma.matrix.transpose",
    }
    description = "DMA matrix"


class sdma_matrix(dma_matrix):
    name = "sDMA_matrix"
    opcode = 1
    short_cmd = True
    description = "short DMA matrix"


class dma_masked_select(DmaCmd):
    name = "DMA_masked_select"
    opcode = 2
    op_name = "dma.masked_select"
    description = "DMA masked select"


class sdma_masked_select(dma_masked_select):
    name = "sDMA_masked_select "
    short_cmd = True
    description = "short DMA masked select"


class dma_general(DmaCmd):
    name = "DMA_general"
    opcode = 3
    sp_fun = {
        0: "dma.general",
        1: "dma.general.broadcast",
    }
    description = "DMA general"


class sdma_general(dma_general):
    name = "sDMA_general"
    short_cmd = True
    description = "short DMA general"


class dma_cw_transpose(DmaCmd):
    name = "DMA_cw_transpose"
    opcode = 4
    op_name = "dma.cw_transpose"
    description = "DMA CW Transpose"


class dma_nonzero(DmaCmd):
    name = "DMA_nonzero"
    opcode = 5
    op_name = "dma.nonzero"
    description = "DMA nonzero"


class sdma_nonzero(dma_nonzero):
    name = "sDMA_nonzero"
    short_cmd = True
    description = "short DMA nonzero"


class dma_sys(DmaCmd):
    name = "sDMA_sys"
    opcode = 6
    short_cmd = True
    sp_fun = {
        0: "dma.sys",
        1: "dma.sys.nop",
    }
    description = "short DMA sys"

    def _set_op(self, reg):
        return ([],) * 3

    def __repr__(self):
        return self.op_name


class dma_gather(DmaCmd):
    name = "DMA_gather"
    opcode = 7
    op_name = "gdma.gather"
    description = "DMA gather"


class dma_scatter(DmaCmd):
    name = "DMA_scatter"
    opcode = 8
    op_name = "gdma.scatter"
    description = "DMA scatter"
