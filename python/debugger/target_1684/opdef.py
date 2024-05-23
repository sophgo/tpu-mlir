# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Dict, Tuple

from ..target_common import OpInfo, BaseTpuCmd, atomic_reg, Tiu, Dma
from .opparam import opparam_converter, default_converter

tiu_cls = dict()


class DmaCmd(BaseTpuCmd, Dma):
    opparam_converter = opparam_converter
    default_converter = default_converter

    length = 1024
    description = "DMA tensor"
    sp_fun = ()

    def __init__(self, reg: atomic_reg, *, buf: memoryview, subnet_id=0) -> None:
        super().__init__(reg, buf=buf, subnet_id=subnet_id)
        self.name = "dma"
        self._cmd_id = None
        self._cmd_id_dep = None

    def __repr__(self):
        cmd_id_desc = self.cmd_id
        cmd_id_dep_desc = self.cmd_id_dep
        if self._cmd_id is not None:
            cmd_id_desc = f"{cmd_id_desc}({self.reg.cmd_id})"
        if self._cmd_id_dep is not None:
            cmd_id_dep_desc = f"{cmd_id_dep_desc}({self.reg.cmd_id_dep})"

        if self.operands == []:
            op_name = dma_tensor.description
            return f"%D{cmd_id_desc} = {op_name}(%B{cmd_id_dep_desc})"

        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        op_name = self.sp_fun[self.reg["cmd_special_function"]]

        return (
            f"{', '.join(res_name)}, %D{cmd_id_desc} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %B{cmd_id_dep_desc})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )

    @property
    def cmd_id_dep(self):
        if self._cmd_id_dep is not None:
            return self._cmd_id_dep
        return self.reg.cmd_id_dep

    @cmd_id_dep.setter
    def cmd_id_dep(self, v):
        self._cmd_id_dep = v

    @property
    def cmd_id(self):
        if self._cmd_id is not None:
            return self._cmd_id
        return self.reg.cmd_id

    @cmd_id.setter
    def cmd_id(self, v):
        self._cmd_id = v


class TiuCmd(BaseTpuCmd, Tiu):
    opparam_converter = opparam_converter
    default_converter = default_converter

    description = "TIU Operation."
    length = 1024
    # extension
    eu_type = {}

    def __init__(self, reg: atomic_reg, *, buf: memoryview, subnet_id=0) -> None:
        super().__init__(reg, buf=buf, subnet_id=subnet_id)
        self.name = reg.OP_NAME
        self._cmd_id = None
        self._cmd_id_dep = None

    def __init_subclass__(cls) -> None:
        tiu_cls[cls.__name__] = {
            "description": cls.description,
            "tsk_eu_typ": cls.eu_type,
            "tsk_typ": cls.opcode,
        }
        return cls

    def __repr__(self) -> str:
        cmd_id_desc = self.cmd_id
        cmd_id_dep_desc = self.cmd_id_dep
        if self._cmd_id is not None:
            cmd_id_desc = f"{cmd_id_desc}({self.reg.cmd_id})"
        if self._cmd_id_dep is not None:
            cmd_id_dep_desc = f"{cmd_id_dep_desc}({self.reg.cmd_id_dep})"

        if self.operands == []:
            op_name = tiu_cls[self.reg.OP_NAME]["description"]
            return f"%B{cmd_id_desc} = {op_name}(%D{cmd_id_dep_desc})"

        res_name, res_type_t = zip(*((x.name, x.type_str) for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str) for x in self.operands))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(":", " =").replace("'", "")

        op_info = tiu_cls[self.reg.OP_NAME]
        eu_type_id = self.reg["tsk_eu_typ"]
        if len(op_info["tsk_eu_typ"]) == 0:
            op_name = self.reg.OP_NAME
        else:
            op_name = op_info["tsk_eu_typ"][eu_type_id]

        return (
            f"{', '.join(res_name)}, %B{cmd_id_desc} = \"{op_name}\""
            + f"({', '.join(opd_name)}, %D{cmd_id_dep_desc})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
        )

    @property
    def cmd_id_dep(self):
        if self._cmd_id_dep is not None:
            return self._cmd_id_dep
        return self.reg.cmd_id_dep

    @cmd_id_dep.setter
    def cmd_id_dep(self, v):
        self._cmd_id_dep = v

    @property
    def cmd_id(self):
        if self._cmd_id is not None:
            return self._cmd_id
        return self.reg.cmd_id

    @cmd_id.setter
    def cmd_id(self, v):
        self._cmd_id = v


class conv_op(TiuCmd):
    description = "convolution neuron"
    opcode = 0
    eu_type = {0: "conv", 1: "conv"}


class pord_op(TiuCmd):
    description = "depthwise or pooling"
    opcode = 1
    eu_type = {1: "pord", 4: "pord.maxpooling"}

    def _decode(self):
        super()._decode()
        if self.reg.tsk_eu_typ == 1:
            if self.reg.opt_opd1_const == 0:
                self.op_name = "pord.depthwise"
            else:
                self.op_name = "pord.avgpooling"


class mm_op(TiuCmd):
    description = "matrix multiply"
    opcode = 2
    eu_type = {0: "mm.mul", 1: "mm.mac"}


class ar_op(TiuCmd):
    description = "tensor arithmetic"
    opcode = 3
    eu_type = {
        0: "arith.mul",
        1: "arith.mac",
        2: "arith.add",
        3: "arith.sub",
        4: "arith.max",
        5: "arith.min",
        6: "arith.shift",
        7: "arith.and",
        8: "arith.or",
        9: "arith.xor",
        10: "arith.select_gt",
        11: "arith.select_eq",
        12: "arith.divide",
        13: "arith.taylor",
        14: "arith.fp32_to_int",
        15: "arith.int_normalize",
        16: "arith.fp32_normalize",
        17: "arith.rsqrt",
        18: "arith.add_tree",
        19: "arith.copy",
        20: "arith.square_sum",
        21: "arith.square_diff",
        22: "arith.cmp",
        23: "arith.select",
        24: "arith.cmp_select",
    }


class mm2_op(TiuCmd):
    description = "matrix multiply2"
    opcode = 4
    eu_type = {18: "mm2"}


class cc_op(TiuCmd):
    opcode = 5
    eu_type = {
        0: "mul",
        1: "mac",
        2: "add",
        3: "sub",
        4: "max",
        5: "min",
        6: "shift",
        7: "and",
        8: "or",
        9: "xor",
        10: "select_gt",
        11: "select_eq",
        12: "divide",
        13: "taylor",
        14: "fp32_to_int",
        15: "int_normalize",
        16: "fp32_normalize",
        17: "rsqrt",
        18: "add_tree",
        19: "copy",
        20: "arith.square_sum",
        21: "arith.square_diff",
        22: "arith.cmp",
        23: "arith.select",
        24: "arith.cmp_select",
    }
    description = "convolution correlation"


class lut_op(TiuCmd):
    description = "table lookup"
    opcode = 6
    eu_type = {0: "lut", 19: "lut"}


class md_sum_op(TiuCmd):
    description = "md sum"
    opcode = 7
    eu_type = {18: "mdsum"}


class md_scalar_op(TiuCmd):
    description = "md scalar"
    opcode = 8
    eu_type = {
        0: "mdscalar.mul",
        1: "mdscalar.mac",
        2: "mdscalar.add",
        3: "mdscalar.sub",
        4: "mdscalar.max",
        5: "mdscalar.min",
        6: "mdscalar.shift",
        7: "mdscalar.and",
        8: "mdscalar.or",
        9: "mdscalar.xor",
        10: "mdscalar.select_gt",
        11: "mdscalar.select_eq",
        12: "mdscalar.divide",
        13: "mdscalar.taylor",
        14: "mdscalar.fp32_to_int",
        15: "mdscalar.int_normalize",
        16: "mdscalar.fp32_normalize",
        17: "mdscalar.rsqrt",
        18: "mdscalar.add_tree",
        19: "mdscalar.copy",
        20: "mdscalar.square_sum",
    }


class md_sfu_op(TiuCmd):
    description = "md sfu"
    opcode = 9
    eu_type = {
        0: "mdsfu.mul",
        1: "mdsfu.mac",
        2: "mdsfu.add",
        3: "mdsfu.sub",
        4: "mdsfu.max",
        5: "mdsfu.min",
        6: "mdsfu.shift",
        7: "mdsfu.and",
        8: "mdsfu.or",
        9: "mdsfu.xor",
        10: "mdsfu.select_gt",
        11: "mdsfu.select_eq",
        12: "mdsfu.divide",
        13: "mdsfu.taylor",
        14: "mdsfu.fp32_to_int",
        15: "mdsfu.int_normalize",
        16: "mdsfu.fp32_normalize",
        17: "mdsfu.rsqrt",
        18: "mdsfu.add_tree",
        19: "mdsfu.copy",
        20: "mdsfu.square_sum",
        21: "mdsfu.square_diff",
        22: "mdsfu.cmp",
        23: "mdsfu.select",
        24: "mdsfu.cmp_select",
    }


class md_linear_op(TiuCmd):
    description = "md linear"
    opcode = 10
    eu_type = {
        1: "mdlinear.mac",
        20: "mdlinear.square_sum",
        21: "mdlinear.square_diff",
    }


class lma_op(TiuCmd):
    description = "local memory arrangement"
    opcode = 11
    eu_type = {19: "lmem_arrangement"}


class decompress_op(TiuCmd):
    description = "decompress"
    opcode = 12
    eu_type = {19: "decompress"}


class md_cmp_op(TiuCmd):
    description = "md cmp"
    opcode = 13
    eu_type = {22: "mdcmp.cmp", 23: "mdcmp.select", 24: "mdcmp.cmp_select"}


class vc_op(TiuCmd):
    description = "vector correlation"
    opcode = 14
    eu_type = {
        0: "vc.mul",
        1: "vc.mac",
        2: "vc.add",
        3: "vc.sub",
        4: "vc.max",
        5: "vc.min",
        6: "vc.shift",
        7: "vc.and",
        8: "vc.or",
        9: "vc.xor",
        10: "vc.select_gt",
        11: "vc.select_eq",
        12: "vc.divide",
        13: "vc.taylor",
        14: "vc.fp32_to_int",
        15: "vc.int_normalize",
        16: "vc.fp32_normalize",
        17: "vc.rsqrt",
        18: "vc.add_tree",
        19: "vc.copy",
        20: "vc.square_sum",
    }


class dma_tensor(DmaCmd):
    description = "DMA tensor"
    opcode = 1
    sp_fun = {
        0: "dma",
        1: "dma.trans",
        2: "dma.lrn_shift",
        3: "dma.format",
        4: "dma.constant",
        5: "dma.cw_trans",
        6: "dma.winograd",
        7: "dma.filter",
    }


# build tiu and dma search tree
# search by cmd_short, tsk_typ, tsk_eu_type
tiu_index: Dict[Tuple[int, int], OpInfo] = {}

for k, v in tiu_cls.items():
    if len(v["tsk_eu_typ"]) == 0:
        tsk_eu_typ = {0: "none"}
    else:
        tsk_eu_typ = v["tsk_eu_typ"]

    if isinstance(tsk_eu_typ, range):
        tsk_eu_typ = {i: f"ana_{i}" for i in tsk_eu_typ}

    for eu_type, eu_name in tsk_eu_typ.items():
        tiu_index[(v["tsk_typ"], eu_type)] = OpInfo(k, eu_name)
