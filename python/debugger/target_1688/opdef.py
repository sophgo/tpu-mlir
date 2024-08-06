# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Dict, Tuple
from ..target_common import BaseTpuCmd, atomic_reg, OpInfo, Tiu, Dma, RegIndex, ALIGN, DType, DIV_UP
from .regdef import SYS_TR_ACC_reg
from .memmap import info, CubeOutputHeightWidthAlignNum, ExecutionUnitNumber
from abc import abstractmethod
import math
# global data and type
# ------------------------------------------------------------


tiu_cls = dict()
dma_cls = dict()
tiu_index = RegIndex()
dma_index = RegIndex()


class TiuCmd(BaseTpuCmd, Tiu):
    opparam_converter = None  # assigned by BM1686Context instance
    description = "TIU Operation."
    # extension
    eu_type = ()
    short_cmd = False  # long_code by default

    def __init__(
        self,
        reg: atomic_reg,
        *,
        buf: memoryview,
        cmd_id,
        param_fn,
        subnet_id=0,
        core_id=0,
    ) -> None:
        assert param_fn is not None
        super().__init__(
            reg, buf=buf, subnet_id=subnet_id, core_id=core_id, param_fn=param_fn
        )
        self.cmd_id = cmd_id
        # cmd_id_dep of SYS_TR_ACC_reg will be assigned in merge_instruction assigned in merge_instruction
        self.cmd_id_dep = getattr(reg, "cmd_id_dep", None)
        self.eu_name = tiu_cls[reg.OP_NAME]["tsk_eu_typ"][reg.tsk_eu_typ]

    @abstractmethod
    def ops(self, is_arch: bool) -> int:
        return 0

    @abstractmethod
    def initial_cycle(self) -> int:
        return 0

    @abstractmethod
    def bank_conflict_cycle(self) -> int:
        return 0

    @abstractmethod
    def alg_cycle(self, alg_ops: int) -> int:
        return 0

    @property
    def op_name(self):
        op_name = self.reg.OP_NAME
        op_info = tiu_cls[self.reg.OP_NAME]
        eu_type_id = self.reg["tsk_eu_typ"]

        if len(op_info["tsk_eu_typ"]) != 0:
            op_name = op_info["tsk_eu_typ"][eu_type_id]
        return op_name

    def __init_subclass__(cls) -> None:
        tiu_index[(cls.short_cmd, cls.opcode, cls.eu_type)] = cls
        tiu_cls[cls.name] = {
            "description": cls.description,
            "tsk_eu_typ": cls.eu_type,
            "tsk_typ": cls.opcode,
            "short_cmd": cls.short_cmd,
        }
        return cls

    def __repr__(self) -> str:
        ci = self.core_id
        if self.operands == []:
            if self.attribute:
                tmp_attr = self.attribute.copy()
                attribute = f" {tmp_attr}".replace(":", " =").replace("'", "")
                if "msg_id" in tmp_attr:
                    msg_id = tmp_attr.pop("msg_id")
                    return (
                        f'%B{self.cmd_id}C{ci} = "{self.op_name}"'
                        + f"(%D{self.cmd_id_dep}C{ci}, %msg{msg_id})"
                        + attribute
                    )
                return (
                    f'%B{self.cmd_id}C{ci} = "{self.op_name}"'
                    + f"(%D{self.cmd_id_dep}C{ci})"
                    + attribute
                )
            else:
                return self.description
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))

        attribute_dic = {}
        if self.attribute:
            attribute_dic.update(self.attribute)

        op_name = self.op_name
        attribute = f"{attribute_dic}" if len(attribute_dic) > 0 else ""
        attribute = f" {attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %B{self.cmd_id}C{ci} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %D{self.cmd_id_dep}C{ci})"
            + attribute
            + f": ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
        )


class DmaCmd(BaseTpuCmd, Dma):
    opparam_converter = None  # assigned by BM1686Context instance

    description = "GDMA Operation."
    sp_fun = ()
    short_cmd = False  # long_code by default

    def __init__(
        self,
        reg: atomic_reg,
        *,
        buf: memoryview,
        cmd_id,
        param_fn,
        subnet_id=0,
        core_id=0,
    ) -> None:
        assert param_fn is not None
        super().__init__(
            reg, buf=buf, subnet_id=subnet_id, core_id=core_id, param_fn=param_fn
        )
        self.cmd_id = cmd_id
        # dma cmd do not need this in 1688
        self.cmd_id_dep = reg.cmd_id_dep

    def __init_subclass__(cls) -> None:
        dma_index[(cls.short_cmd, cls.opcode, cls.sp_fun)] = cls
        dma_cls[cls.name] = {
            "description": cls.description,
            "tsk_typ": cls.opcode,
            "sp_fun": cls.sp_fun,
            "short_cmd": cls.short_cmd,
        }
        return cls

    def __repr__(self):
        ci = self.core_id
        if self.operands == []:
            if self.attribute:
                tmp_attr = self.attribute.copy()
                msg_id = tmp_attr.pop("msg_id")
                attribute = f" {self.attribute}".replace(":", " =").replace("'", "")
                return (
                    f'%D{self.cmd_id}C{ci} = "{self.op_name}"'
                    + f"(%B{self.cmd_id_dep}C{ci}, %msg{msg_id})"
                    + attribute
                )
            else:
                return self.description
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))

        attribute_dic = {}
        if self.attribute:
            attribute_dic.update(self.attribute)

        op_name = self.op_name

        attribute = f"{attribute_dic}" if len(attribute_dic) > 0 else ""
        attribute = f" {attribute}".replace(":", " =").replace("'", "")

        return (
            f"{', '.join(res_name)}, %D{self.cmd_id}C{ci} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %B{self.cmd_id_dep}C{ci})"
            + attribute
            + f": ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )

    @property
    def op_name(self):
        op_name = self.reg.OP_NAME
        op_info = dma_cls[self.reg.OP_NAME]
        sp_func_id = self.reg["cmd_special_function"]
        if len(op_info["sp_fun"]) != 0:
            # attribute_dic["tsk_typ"] = f'"{op_name}"'
            op_name = op_info["sp_fun"][sp_func_id]

        return op_name

class conv_op(TiuCmd):
    name = "CONV"
    opcode = 0
    eu_type = {0: "conv.normal"}
    description = "convolution"

    def ops(self, is_arch=False):
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        n, ic, ih, iw = self.operands[0].shape
        n, oc, oh, ow = self.results[0].shape

        # has_bias = len(self.operands) > 2
        biasLat = 1 if self.reg.opt_opd2_const else 0
        dtype = self.operands[0].dtype
        if is_arch:
            channelNumPerCyc = info.CUBE_NUM(dtype)
            if dtype == DType.f32:
                activatedEuNumber = info.EU_NUM(dtype)
            else:
                ic = ALIGN(ic, channelNumPerCyc)
                activatedEuNumber = CubeOutputHeightWidthAlignNum
            ow = ALIGN(oh * ow, activatedEuNumber)
            oh = 1
            oc = ALIGN(oc, info.NPU_NUM)
        out_size = n * oc * oh * ow
        kh, kw = self.reg.opd1_h, self.reg.opd1_w
        return out_size * (2 * ic * kh * kw - 1 + biasLat)

    def initial_cycle(self):
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        return 23

    def bank_conflict_cycle(self):
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        return 0

    def alg_cycle(self, alg_ops):
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        dtype = self.operands[0].dtype
        channelNumPerCyc = info.CUBE_NUM(dtype)
        if dtype == DType.f32:
            channelNumPerCyc = 1
            activatedEuNumber = info.EU_NUM(dtype)
        else:
            activatedEuNumber = CubeOutputHeightWidthAlignNum # CubeOutputHeightWidthAlignNum
        return DIV_UP(alg_ops, info.NPU_NUM * channelNumPerCyc * activatedEuNumber * 2)

class sconv_op(conv_op):
    name = "sCONV"
    short_cmd = True
    description = "short convolution"


class mm_op(TiuCmd):
    name = "MM"
    opcode = 2
    eu_type = {1: "mm.normal"}
    description = "matrix multiply"

    def ops(self, is_arch=False):
        m = self.reg.opd0_n
        k = self.reg.opd0_w * (self.reg.opd0_c - 1) + self.reg.opd0_w
        n = self.reg.res0_w * (self.reg.res0_c - 1) + self.reg.res0_w
        if self.attribute["l_trans"]:
            k, m = m, k
        has_bias = len(self.operands) > 2
        if is_arch:
            dtype = self.operands[0].dtype
            # align the column of B
            n = ALIGN(self.reg.res0_c, info.NPU_NUM) * ALIGN(self.reg.res0_w, info.EU_NUM(dtype))
        return m * n * (2 * k - 1 + has_bias)

    def initial_cycle(self) -> int:
        return 32

    def bank_conflict_cycle(self) -> int:
        return 0

    def alg_cycle(self, alg_ops: int) -> int:
        dtype = self.operands[0].dtype
        return DIV_UP(alg_ops, info.EU_NUM(dtype) * info.NPU_NUM * 2)

class smm_op(mm_op):
    name = "sMM"
    short_cmd = True
    description = "short matrix multiply"


class mm2_op(TiuCmd):
    name = "MM2"
    opcode = 2
    eu_type = {4: "mm2.nn", 5: "mm2.nt", 6: "mm2.tt"}
    description = "matrix multiply2"

    def ops(self, is_arch=False):
        # m, lk = self.operands[0].shape
        # rk, n = self.operands[1].shape
        m = self.reg.res0_c
        k = self.reg.opd1_w
        n = self.reg.opd1_c
        if self.eu_name == "mm2.nt":
            k, n = n, k
        elif self.eu_name == "mm2.tt":
            m, k, n = k, m, k
        dtype = self.results[0].dtype
        channelNumPerCyc = info.CUBE_NUM(dtype)
        activatedEuNumber = CubeOutputHeightWidthAlignNum
        if is_arch:
            k = ALIGN(k, activatedEuNumber)
            m = ALIGN(m, info.NPU_NUM)
            n = ALIGN(n, channelNumPerCyc)

        return m * n * k * 2

    def initial_cycle(self) -> int:
        init_cycle_dict = {
            4: 37+44,
            5: 37+19,
            6: 37
        }
        return init_cycle_dict.get(self.reg['tsk_eu_typ'], 0)

    def alg_cycle(self, alg_ops: int) -> int:
        dtype = self.operands[0].dtype
        channelNumPerCyc = info.CUBE_NUM(dtype)
        activatedEuNumber = CubeOutputHeightWidthAlignNum
        return DIV_UP(alg_ops, channelNumPerCyc * activatedEuNumber * info.NPU_NUM * 2)

class smm2_op(mm2_op):
    name = "sMM2"
    short_cmd = True
    description = "short matrix multiply2"


class cmp_op(TiuCmd):
    name = "CMP"
    opcode = 13
    eu_type = {
        22: "cmp.gt_and_sel_gt",
        23: "cmp.sel_gt",
        24: "cmp.sel_eq",
        25: "cmp.lt_and_sel_lt",
        26: "cmp.sel_lt",
    }
    description = "fused_cmpare"

    def ops(self, is_arch=False):
        n, c, h, w = self.results[0].shape
        # res_num = len(self.results)

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(h * w, info.EU_NUM(dtype))
        return n * c * hw * 2

    def alg_cycle(self, alg_ops: int) -> int:
        perLaneEuNumber = ExecutionUnitNumber // info.NPU_NUM
        activatedEuNumber = perLaneEuNumber // math.ceil(
            self.results[0].dtype.itemsize)
        return DIV_UP(alg_ops, activatedEuNumber * info.NPU_NUM * 2)

    def initial_cycle(self) -> int:
        return 9
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

    def ops(self, is_arch=False):
        n, c, h, w = self.operands[0].shape
        factor = 1
        res_num = len(self.results)
        if self.eu_name == "sfu.taylor_4x" or self.eu_name == "sfu.taylor":
            factor = 2 * self.reg.opd1_n - 1  # 2* table_len -1
        elif self.eu_name == "sfu.normalize":
            factor = 1
        elif self.eu_name == "sfu.rsqrt":
            factor = 30

        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(w * h, info.EU_NUM(dtype))

        return res_num * n * c * hw * factor

    def alg_cycle(self, alg_ops: int) -> int:
        dtype = self.operands[0].dtype
        factor = 1
        if self.eu_name == "sfu.taylor_4x" or self.eu_name == "sfu.taylor":
            factor = 2
        return DIV_UP(alg_ops, info.EU_NUM(dtype) * info.NPU_NUM * factor)

    def initial_cycle(self) -> int:
        if self.eu_name == "sfu.rsqrt":
            return 14
        return 0

    def bank_conflict_cycle(self) -> int:
        return 0
class ssfu_op(sfu_op):
    name = "sSFU"
    short_cmd = True
    description = "short special_function"


class lin_op(TiuCmd):
    name = "LIN"
    opcode = 10
    eu_type = {1: "lin.mac", 20: "lin.square_sum", 21: "lin.square_diff"}
    description = "fused_linear"

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 2
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor

    def alg_cycle(self, alg_ops: int) -> int:
        factor = 2
        perLaneEuNumber = ExecutionUnitNumber // info.NPU_NUM
        activatedEuNumber = perLaneEuNumber // math.ceil(
            self.results[0].dtype.itemsize)
        return DIV_UP(alg_ops, activatedEuNumber * info.NPU_NUM * factor)

    def initial_cycle(self) -> int:
        return 12

    def bank_conflict_cycle(self) -> int:
        return 0
class slin_op(lin_op):
    name = "sLIN"
    short_cmd = True
    description = "short fused_linear"


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

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        if is_arch:
            dtype = self.operands[0].dtype
            perLaneEuNumber = ExecutionUnitNumber // info.NPU_NUM
            opd0Byte = math.ceil(self.operands[0].dtype.itemsize)
            opd1Byte = math.ceil(self.operands[1].dtype.itemsize)
            res0Byte = math.ceil(self.results[1].dtype.itemsize)
            maxByte = max(opd0Byte, opd1Byte, res0Byte)
            activatedEuNumber = perLaneEuNumber // maxByte
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, activatedEuNumber)
        return n * c * h * w

    def alg_cycle(self, alg_ops: int) -> int:
        perLaneEuNumber = ExecutionUnitNumber // info.NPU_NUM
        opd0Byte = math.ceil(self.operands[0].dtype.itemsize)
        opd1Byte = math.ceil(self.operands[1].dtype.itemsize)
        res0Byte = math.ceil(self.results[1].dtype.itemsize)
        maxByte = max(opd0Byte, opd1Byte, res0Byte)
        activatedEuNumber = perLaneEuNumber // maxByte
        return DIV_UP(alg_ops, activatedEuNumber * info.NPU_NUM)

    def initial_cycle(self) -> int:
        return 0

    def bank_conflict_cycle(self) -> int:
        return 0

class svc_op(vc_op):
    name = "sVC"
    short_cmd = True
    description = "short vector correlation"


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
        6: "arith.logic_shift",
        7: "arith.and",
        8: "arith.or",
        9: "arith.xor",
        10: "arith.select_great",
        11: "arith.select_equal",
        12: "arith.div",
        13: "arith.select_less",
        14: "arith.cast",
        15: "arith.add_satu",
        16: "arith.sub_satu",
        18: "arith.mac",
        19: "arith.copy",
        20: "arith.mul_satu",
        21: "arith.arith_shift",
        22: "arith.rotate_shift",
        26: "arith.abs",
        27: "arith.fsub_abs",
        29: "arith.get_first_one",
        30: "arith.get_first_zero",
    }
    description = "arithmetic"

    def ops(self, is_arch):
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

    def initial_cycle(self) -> int:
        # if self.reg['tsk_eu_typ'] in [14, 19, 11, 15, 4, 5, 26]:
        #     misc = -1
        # elif self.reg['tsk_eu_typ'] == 12:
        #     misc = -4
        # else:
        #     misc = 0
        # return 10 + misc
        return 11

    def alg_cycle(self, alg_ops: int) -> int:
        factor = 1
        if self.reg['tsk_eu_typ'] == 12:
            factor = 5
        dtype = self.operands[0].dtype
        return DIV_UP(alg_ops, info.EU_NUM(dtype) * factor * info.NPU_NUM)


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
        3: "pord.minpooling",
        4: "pord.maxpooling",
        5: "pord.roi_depthwise",
        6: "pord.roi_avgpooling",
        7: "pord.roi_maxpooling",
        8: "pord.roi_minpooling",
    }
    description = "depthwise or pooling"

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        kh, kw = self.reg.opd1_h, self.reg.opd1_w
        factor = 1
        if self.eu_name == "pord.avgpooling" or self.eu_name == "pord.maxpooling":
            factor = 2  # TODO: fix the factor
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

    def initial_cycle(self) -> int:
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        return 10

    def bank_conflict_cycle(self) -> int:
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        return 0

    def alg_cycle(self, alg_ops: int) -> int:
        # borrowed from TPUPerf/c_model/src/tpu/tiuImpl.cc
        switcher = {
            0: 2,
            1: 1,
            2: 3,
            4: 1
            }
        factor = switcher.get(self.reg["tsk_eu_typ"], 7)
        return DIV_UP(alg_ops, factor * info.EU_NUM(self.operands[0].dtype) * info.NPU_NUM)

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
        3: "quant.dq0",
        4: "quant.dq1",
    }
    description = "RQ && DQ"

    def _set_op(self, reg):
        return ([],) * 3

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 3  # mul, add, shift
        hw = h * w
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            hw = ALIGN(h * w, info.EU_NUM(dtype))
        return n * c * hw * factor

    def alg_cycle(self, alg_ops: int) -> int:
        factor = 3
        dtype = self.operands[0].dtype
        activatedEuNumber = info.EU_NUM(dtype)
        return DIV_UP(alg_ops, activatedEuNumber * info.NPU_NUM * factor)

    def initial_cycle(self) -> int:
        return 11

    def bank_conflict_cycle(self) -> int:
        if (self.reg.res0_addr * 2) == self.reg.opd0_addr:
            bankConflictRatio = 2
        else:
            bankConflictRatio = 1
        return bankConflictRatio

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
        8: "sg.pe_s_mask_select",
        9: "sg.pe_s_nonzero",
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
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor

    def alg_cycle(self, alg_ops: int) -> int:
        return DIV_UP(alg_ops, info.NPU_NUM)

    def initial_cycle(self) -> int:
        return 11

    def bank_conflict_cycle(self) -> int:
        return 0

class ssg_op(sg_op):
    name = "sSG"
    short_cmd = True
    description = "short scatter_gather"


class sgl_op(TiuCmd):
    name = "SGL"
    opcode = 6
    eu_type = {17: "sgl.pe_s_gather_line", 18: "sgl.pe_s_scatter_line"}
    description = "scatter_gather_line"

    def ops(self, is_arch):
        n, c, h, w = self.results[0].shape
        factor = 1
        if is_arch:
            dtype = self.operands[0].dtype
            c = ALIGN(c, info.NPU_NUM)
            w = ALIGN(w, info.EU_NUM(dtype))
        return n * c * h * w * factor

    def alg_cycle(self, alg_ops: int) -> int:
        return DIV_UP(alg_ops, info.NPU_NUM * info.NPU_NUM)

    def initial_cycle(self) -> int:
        return 24

    def bank_conflict_cycle(self) -> int:
        return 0

class ssgl_op(sgl_op):
    name = "sSGL"
    short_cmd = True
    description = "short scatter_gather_line"


class transbc_op(TiuCmd):
    name = "CW&BC"
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

    def alg_cycle(self, alg_ops: int) -> int:
        perLaneEuNumber = ExecutionUnitNumber // info.NPU_NUM
        channelNumPerCyc = info.NPU_NUM
        resByte = math.ceil(self.results[0].dtype.itemsize)
        activatedEuNumber = perLaneEuNumber // resByte
        if self.eu_name in ("tsbc.cw_ts", "tsbc.wc_ts", "tsbc.l_copy", "tsbc.s_distribute"):
            throughPut = activatedEuNumber
        else:
            throughPut = activatedEuNumber * channelNumPerCyc
        return DIV_UP(alg_ops, throughPut)

    def initial_cycle(self) -> int:
        init_cycle_dict={
            0: 30,
            2: 27,
            3: 27,
            4: 3,
            5: 3
        }
        return init_cycle_dict.get(self.reg['tsk_eu_typ'], 0)

class stransbc_op(transbc_op):
    name = "sCW&sBC"
    short_cmd = True
    description = "short TRANS && BC"

class tiu_sys_tr_acc(TiuCmd):
    name = "SYS_TR_ACC"
    opcode = 12
    short_cmd = None
    eu_type = {
        0: "system_tr_wr.wr_imm",
    }
    description = "system tr wr"

    def ops(self, is_arch):
        return 1


class tiu_sys(TiuCmd):
    name = "SYS"
    opcode = 15
    eu_type = {
        1: "system.spb",
        2: "system.swr",
        3: "system.swr_from_lmm",
        4: "system.swr_collect_from_lmm",
        8: "system.send_msg",
        9: "system.wait_msg",
        30: "system.nop",
        31: "system.end",
    }
    description = "system"

    def ops(self, is_arch):
        return 1

    def _set_op(self, reg):
        return ([],) * 3


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


class dma_masked_select(DmaCmd):
    name = "DMA_masked_select"
    opcode = 2
    sp_fun = {
        0: "dma.masked_select",
        1: "dma.masked_select.ncw",
    }
    description = "DMA masked select"


class dma_general(DmaCmd):
    name = "DMA_general"
    opcode = 3
    sp_fun = {
        0: "dma.general",
        1: "dma.general.broadcast",
    }
    description = "DMA general"


class dma_cw_transpose(DmaCmd):
    name = "DMA_cw_transpose"
    opcode = 4
    sp_fun = {0: "dma.cw_transpose"}
    description = "DMA CW Transpose"


class dma_nonzero(DmaCmd):
    name = "DMA_nonzero"
    opcode = 5
    sp_fun = {0: "dma.nonzero"}
    description = "DMA nonzero"


class dma_sys(DmaCmd):
    name = "sDMA_sys"
    opcode = 6
    short_cmd = True
    sp_fun = {
        0: "dma.sys.chain_end",
        1: "dma.sys.nop",
        2: "dma.sys.sys_tr_wr",
        3: "dma.sys.sys_send",
        4: "dma.sys.sys_wait",
    }
    description = "short DMA sys"


class dma_gather(DmaCmd):
    name = "DMA_gather"
    opcode = 7
    sp_fun = {0: "gdma.gather"}
    description = "DMA gather"


class dma_scatter(DmaCmd):
    name = "DMA_scatter"
    opcode = 8
    sp_fun = {0: "gdma.scatter"}
    description = "DMA scatter"


class dma_reverse(DmaCmd):
    name = "DMA_reverse"
    opcode = 9
    sp_fun = {
        0: "dma.reverse.w",
        1: "dma.reverse.h",
        2: "dma.reverse.c",
        3: "dma.reverse.n",
    }
    description = "DMA reverse"


class dma_compress(DmaCmd):
    name = "DMA_compress"
    opcode = 10
    sp_fun = {
        0: "dma.compress.non_random_access",
        1: "dma.compress.random_access",
    }
    description = "DMA compress"


class dma_decompress(DmaCmd):
    name = "DMA_decompress "
    opcode = 11
    sp_fun = {
        0: "dma.decompress.non_random_access",
        1: "dma.decompress.random_access",
    }
    description = "DMA decompress"
