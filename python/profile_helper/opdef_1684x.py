# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# keep This file clean and neat.

from collections import OrderedDict
import numpy as np

try:
    from . import regdef_1684x
    from .opparam_1684x import opparam_converter, NPU_NUM, EU_NUM
    from .op_support import packbits, TIUBase, DMABase, NamedDict, ALIGN, DIV_UP
except:
    import regdef_1684x
    from opparam_1684x import opparam_converter, NPU_NUM, EU_NUM
    from op_support import packbits, TIUBase, DMABase, NamedDict, ALIGN, DIV_UP

# global data and type
# ------------------------------------------------------------
bdc_cmd = dict()
dma_cmd = dict()

bank_size = 2**14

# ------------------------------------------------------------
# registry function
# ------------------------------------------------------------
def base_registry(cmd_type, sheet_name, cls):
    attr = regdef_1684x.reg_def[sheet_name]
    fields, bits = zip(*attr)
    # high bit is the upper bit (open interval).
    setattr(cls, "des_reg", {"fields": fields, "high_bit": bits})
    setattr(cls, "length", bits[-1])
    cmd_type.setdefault(cls.opcode, set()).add(cls)
    if sheet_name in opparam_converter:
        setattr(cls, "_set_op", staticmethod(opparam_converter[sheet_name]))
    return cls


def bdc_registry(sheet_name):
    def decorate(cls):
        return base_registry(bdc_cmd, sheet_name, cls)

    return decorate


def dma_registry(sheet_name):
    def decorate(cls):
        return base_registry(dma_cmd, sheet_name, cls)

    return decorate


def decode_reg(buffer, des_reg):
    bits_sec = np.split(buffer, des_reg["high_bit"][:-1])
    value = (packbits(x) for x in bits_sec)
    return OrderedDict(zip(des_reg["fields"], value))


# ------------------------------------------------------------
# BDC definition
# ------------------------------------------------------------
class bdc_base(TIUBase):
    opcode_bits = (41, 45)
    # extension
    eu_type = ()
    eu_bits = (45, 50)
    short_cmd = False  # long_code by default

    def _decode(self):
        self.reg = NamedDict(decode_reg(self.cmd, self.des_reg))
        self.cmd_id = self.reg.cmd_id
        self.cmd_id_dep = self.reg.cmd_id_dep
        self.op_name = self.eu_type[self.reg.tsk_eu_typ]

    def _is_comp(self, cmd_reg):
        if cmd_reg.size < self.length:
            return False
        if self.short_cmd is not None and bool(cmd_reg[0]) != self.short_cmd:
            return False
        if packbits(cmd_reg[self.opcode_bits[0] : self.opcode_bits[1]]) != self.opcode:
            return False
        if packbits(cmd_reg[self.eu_bits[0] : self.eu_bits[1]]) not in self.eu_type:
            return False
        return True

    def __repr__(self):
        if self.operands == []:
            return self.description
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %B{self.cmd_id} = \"{self.op_name}\""
            + f"({', '.join(opd_name)}, %D{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
        )
    def ops(self, is_arch):
        assert False, "Not implemented!"

    def arch_ops(self):
        return self.ops(True)

    def alg_ops(self):
        return self.ops(False)


@bdc_registry("CONV")
class conv_op(bdc_base):
    opcode = 0
    eu_type = {0: "conv.normal", 1: "conv.wrq", 2: "conv.wrqrelu"}
    description = "convolution"

    def ops(self, is_arch=False):
        n, ic, ih, iw = self.operands[0].shape
        n, oc, oh, ow = self.results[0].shape
        # remains_hw = 0

        has_bias = len(self.operands) > 2
        if is_arch:
            dtype = self.operands[0].dtype
            ic = ALIGN(ic, NPU_NUM)
            ow = ALIGN(oh*ow, EU_NUM(dtype))
            oh = 1
            oc = ALIGN(oc, NPU_NUM)
            # iw = ALIGN(ih * iw, EU_NUM(dtype))
            # ih = 1
            # kw = ALIGN(kw, 64)
            # remain_hw = ALIGN(ih*iw, EU_NUM) - ih*iw
        out_size = n * oc * oh * ow
        kh, kw = self.attribute["kernel"]
        return out_size * (2 * ic * kh * kw - 1 + has_bias)


@bdc_registry("sCONV")
class sconv_op(conv_op):
    short_cmd = True
    description = "short convolution"


@bdc_registry("MM")
class mm_op(bdc_base):
    opcode = 2
    eu_type = {1: "mm.normal", 2: "mm.wrq", 3: "mm.wrqrelu"}
    description = "matrix multiply"

    def ops(self, is_arch=False):
        m, lk = self.operands[0].shape
        rk, n = self.operands[1].shape
        if self.attribute["l_trans"]:
            lk, m = m, lk
        assert lk == rk
        k = lk
        res_add = self.reg.res_add
        has_bias = len(self.operands) > 2
        if is_arch:
            dtype = self.operands[0].dtype
            # align the column of B
            n = ALIGN(self.reg.res0_c, NPU_NUM) * ALIGN(self.reg.res0_w, EU_NUM(dtype))
        return m * n * (2 * k - 1 + has_bias + res_add)


@bdc_registry("sMM")
class smm_op(mm_op):
    short_cmd = True
    description = "short matrix multiply"

@bdc_registry("MM2")
class mm2_op(bdc_base):
    opcode = 2
    eu_type = {4: "mm2.nn", 5: "mm2.nt", 6: "mm2.tt"}
    description = "matrix multiply2"

    def ops(self, is_arch=False):
        m, lk = self.operands[0].shape
        rk, n = self.operands[1].shape
        if self.op_name == "mm2.nt":
            rk, n = n, rk
        elif self.op_name == "mm2.tt":
            rk, n = n, rk
            lk, m = m, lk

        assert lk == rk
        k = lk
        has_bias = len(self.operands) > 2

        if is_arch:
            dtype = self.operands[0].dtype
            k = ALIGN(k, EU_NUM(dtype))
            m = ALIGN(m, NPU_NUM)

        return m*n*(2*k-1 + has_bias)


@bdc_registry("sMM2")
class smm2_op(mm2_op):
    short_cmd = True
    description = "short matrix multiply2"


@bdc_registry("CMP")
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

    def ops(self, is_arch=False):
        n, c, h, w = self.results[0].shape
        # res_num = len(self.results)

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            hw = ALIGN(h * w, EU_NUM(dtype))
        return n * c * hw * 2


@bdc_registry("sCMP")
class scmp_op(cmp_op):
    short_cmd = True
    description = "short fused_cmpare"


@bdc_registry("SFU")
class sfu_op(bdc_base):
    opcode = 9
    eu_type = {
        12: "sfu.taylor_4x",
        13: "sfu.taylor",
        15: "sfu.normalize",
        17: "sfu.rsqrt",
    }
    description = "special_function"

    def ops(self, is_arch=False):
        n, c, h, w = self.operands[0].shape
        factor = 1
        res_num = len(self.results)
        if self.op_name == "sfu.taylor_4x" or self.op_name == "sfu.taylor":
            factor = 2*self.reg.opd1_n - 1  # 2* table_len -1
        elif self.op_name == "sfu.normalize":
            factor = 1
        elif self.op_name == "sfu.rsqrt":
            factor = self.reg.opd2_n_str + 1  # iteration times

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            hw = ALIGN(w * h, EU_NUM(dtype))

        return res_num * n * c * hw * factor


@bdc_registry("sSFU")
class ssfu_op(sfu_op):
    short_cmd = True
    description = "short special_function"


@bdc_registry("VC")
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

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(w, EU_NUM(dtype))
        return n*c*h*w


@bdc_registry("sVC")
class svc_op(vc_op):
    short_cmd = True
    description = "short vector correlation"
    def ops(self, is_arch):
        n, c, h, w = self.operands[0].shape


@bdc_registry("LIN")
class lin_op(bdc_base):
    opcode = 10
    eu_type = {1: "lin.mac", 20: "lin.square_sum", 21: "lin.square_diff"}
    description = "fused_linear"

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 2
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(w, EU_NUM(dtype))
        return n*c*h*w*factor


@bdc_registry("sLIN")
class slin_op(lin_op):
    short_cmd = True
    description = "short fused_linear"


@bdc_registry("AR")
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
    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        hw = h * w
        if self.op_name == "arith.div":
            factor = 5 # TODO: fix the factor
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            hw = ALIGN(w * h, EU_NUM(dtype))
        return n*c*hw*factor


@bdc_registry("sAR")
class sar_op(ar_op):
    short_cmd = True
    description = "short arithmetic"


# @bdc_registry("SEG")
# class seg_op(bdc_base):
#     opcode = 3
#     eu_type = [17, 24, 25]
#     description = "arithmetic"


# @bdc_registry("sSEG")
# class sseg_op(seg_op):
#     short_cmd = True
#     description = "short arithmetic"


@bdc_registry("PorD")
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
    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        kh, kw = self.reg.opd1_h, self.reg.opd1_w
        factor = 1
        if self.op_name == "pord.avgpooling" or self.op_name == "pord.maxpooling":
            factor = len(self.results) # TODO: fix the factor
        elif self.op_name == "pord.depthwise":
            factor = 2
        elif self.op_name == "pord.depthwiserelu":
            factor = 3
        else:
            # roi_pooling
            kh = 1
            kw = 1
            factor = 2*4 -1 # bilinar, ignore coords generate
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(h*w, EU_NUM(dtype))
            h = 1
        return n*c*h*w*(factor*kh*kw-1)


@bdc_registry("sPorD")
class spord_op(pord_op):
    short_cmd = True
    description = "short depthwise or pooling"


@bdc_registry("RQ&DQ")
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
    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 3 # mul, add, shift
        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            hw = ALIGN(h * w, EU_NUM(dtype))
        return n * c * hw * factor


@bdc_registry("sRQ&sDQ")
class srqdq_op(rqdq_op):
    short_cmd = True
    description = "short RQ && DQ"


@bdc_registry("SG")
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
    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(w, EU_NUM(dtype))
        return n*c*h*w*factor


@bdc_registry("sSG")
class ssg_op(sg_op):
    short_cmd = True
    description = "short scatter_gather"


@bdc_registry("SGL")
class sgl_op(bdc_base):
    opcode = 6
    eu_type = {17: "sgl.pe_s_gather_line", 18: "sgl.pe_s_scatter_line"}
    description = "scatter_gather_line"
    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(w, EU_NUM(dtype))
        return n*c*h*w*factor


@bdc_registry("sSGL")
class ssgl_op(sgl_op):
    short_cmd = True
    description = "short scatter_gather_line"


@bdc_registry("TRANS&BC")
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

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            if self.op_name in (
                "tsbc.l_copy",
                "tsbc.l_bc",
                "tsbc.s_bc",
                "tsbc.s_distribute",
            ):
                hw = ALIGN(h * w, EU_NUM(dtype))
            else:
                hw = h * ALIGN(w, EU_NUM(dtype))
        return n * c * hw * factor


@bdc_registry("sTRANS&sBC")
class stransbc_op(transbc_op):
    short_cmd = True
    description = "short TRANS && BC"


@bdc_registry("LAR")
class lar_op(bdc_base):
    opcode = 7
    short_cmd = None
    eu_type = range(31)
    description = "linear_arithmetic"

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, NPU_NUM)
            w = ALIGN(w, EU_NUM(dtype))
        return n*c*h*w*factor


# @bdc_registry("SYS")
# class sys_op(bdc_base):
#     opcode = 15
#     short_cmd = None
#     eu_type = (0, 1, 2, 3, 4, 5, 30, 31)
#     description = "system"


@bdc_registry("SYSID")
class sysid_op(bdc_base):
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
    def ops(self, is_arch):
        return 1

    def _set_op(self, reg):
        return ([],) * 3

    def __repr__(self):
        return self.op_name


# ------------------------------------------------------------
# GDMA definition
# ------------------------------------------------------------
class dma_base(DMABase):
    description = "GDMA Operation."
    opcode_bits = (32, 36)
    fun_bits = (36, 39)
    sp_fun = ()
    short_cmd = False  # long_code by default

    def _decode(self):
        self.reg = NamedDict(decode_reg(self.cmd, self.des_reg))
        self.cmd_id = self.reg.cmd_id
        self.cmd_id_dep = self.reg.cmd_id_dep
        if self.sp_fun:
            self.op_name = self.sp_fun[self.reg.cmd_special_function]

    def _is_comp(self, cmd_reg):
        if cmd_reg.size < self.length:
            return False
        if self.short_cmd is not None and bool(cmd_reg[3]) != self.short_cmd:
            return False
        if packbits(cmd_reg[self.opcode_bits[0] : self.opcode_bits[1]]) != self.opcode:
            return False
        sp_fun_id = packbits(cmd_reg[self.fun_bits[0] : self.fun_bits[1]])
        if self.sp_fun and (sp_fun_id not in self.sp_fun):
            return False
        return True

    def __repr__(self):
        if self.operands == []:
            return self.description
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %D{self.cmd_id} = \"{self.op_name}\""
            + f"({', '.join(opd_name)}, %B{self.cmd_id_dep})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )
    def alg_ops(self):
        return 0
    def arch_ops(self):
        return 0

@dma_registry("DMA_tensor（0x000）")
class dma_tensor(dma_base):
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


@dma_registry("DMA_matrix")
class dma_matrix(dma_base):
    opcode = 1
    sp_fun = {
        0: "dma.matrix",
        1: "dma.matrix.transpose",
    }
    description = "DMA matrix"


@dma_registry("sDMA_matrix")
class sdma_matrix(dma_matrix):
    opcode = 1
    short_cmd = True
    description = "short DMA matrix"


@dma_registry("DMA_masked_select")
class dma_masked_select(dma_base):
    opcode = 2
    op_name = "dma.masked_select"
    description = "DMA masked select"


@dma_registry("sDMA_masked_select ")
class sdma_masked_select(dma_masked_select):
    short_cmd = True
    description = "short DMA masked select"


@dma_registry("DMA_general")
class dma_general(dma_base):
    opcode = 3
    sp_fun = {
        0: "dma.general",
        1: "dma.general.broadcast",
    }
    description = "DMA general"


@dma_registry("sDMA_general")
class sdma_general(dma_general):
    short_cmd = True
    description = "short DMA general"


@dma_registry("DMA_cw_transpose")
class dma_cw_transpose(dma_base):
    opcode = 4
    op_name = "dma.cw_transpose"
    description = "DMA CW Transpose"


@dma_registry("DMA_nonzero")
class dma_nonzero(dma_base):
    opcode = 5
    op_name = "dma.nonzero"
    description = "DMA nonzero"


@dma_registry("sDMA_nonzero")
class sdma_nonzero(dma_nonzero):
    short_cmd = True
    description = "short DMA nonzero"


@dma_registry("sDMA_sys")
class sdma_sys(dma_base):
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


@dma_registry("DMA_gather")
class dma_gather(dma_base):
    opcode = 7
    op_name = "gdma.gather"
    description = "DMA gather"


@dma_registry("DMA_scatter")
class dma_scatter(dma_base):
    opcode = 8
    op_name = "gdma.scatter"
    description = "DMA scatter"
