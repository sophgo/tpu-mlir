# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# keep This file clean and neat.

try:
    from .regdef_1684 import tiu_reg_def, dma_reg_def
    from .opparam_1684 import opparam_converter
    from .op_support import extract_buf, TIUBase, DMABase, NamedDict, Engine
except:
    from regdef_1684 import tiu_reg_def, dma_reg_def
    from opparam_1684x import opparam_converter
    from op_support import extract_buf, TIUBase, DMABase, NamedDict, Engine

import numpy as np

# global data and type
# ------------------------------------------------------------
tiu_cls = dict()
dma_cls = dict()
tiu_sys = None
dma_sys = None

# ------------------------------------------------------------
# registry function
# ------------------------------------------------------------
def base_registry(cmd_type, reg_def, cls):
    fields, bits = zip(*reg_def)
    # high bit is the upper bit (open interval).
    setattr(cls, "reg_def", {"fields": fields, "high_bit": bits})
    class_name = cls.__name__
    cmd_type.setdefault(cls.opcode, set()).add(cls)
    if class_name in opparam_converter:
        setattr(cls, "_set_op", staticmethod(opparam_converter[class_name]))
    return cls


def tiu_registry(cls):
    return base_registry(tiu_cls, tiu_reg_def, cls)


def dma_registry(cls):
    return base_registry(dma_cls, dma_reg_def, cls)


# ------------------------------------------------------------
# TIU definition
# ------------------------------------------------------------
class tiu_base(TIUBase):
    length = 1024
    opcode_bits = (37, 41)
    # extension
    eu_type = ()
    eu_bits = (41, 46)

    def _decode(self):
        cmd_bits = buffer_to_bits(self.cmd)
        self.reg = NamedDict(decode_reg(cmd_bits, self.reg_def))
        self.cmd_id = self.reg.cmd_id_tpu
        self.cmd_id_dep = self.reg.cmd_id_gdma
        self.op_name = self.eu_type[self.reg.tsk_eu_typ]

    def _is_comp(self, cmd_buf):
        if len(cmd_buf) * 8 < self.length:
            return False
        if extract_buf(cmd_buf, self.opcode_bits) != self.opcode:
            return False
        if extract_buf(cmd_buf, self.eu_bits) not in self.eu_type:
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


@tiu_registry
class conv_op(tiu_base):
    description = "convolution neuron"
    opcode = 0
    eu_type = {0: "conv", 1: "conv"}


@tiu_registry
class pord_op(tiu_base):
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


@tiu_registry
class mm_op(tiu_base):
    description = "matrix multiply"
    opcode = 2
    eu_type = {0: "mm.mul", 1: "mm.mac"}


@tiu_registry
class ar_op(tiu_base):
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


@tiu_registry
class mm2_op(tiu_base):
    description = "matrix multiply2"
    opcode = 4
    eu_type = {18: "mm2"}


@tiu_registry
class cc_op(tiu_base):
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


@tiu_registry
class lut_op(tiu_base):
    description = "table lookup"
    opcode = 6
    eu_type = {0: "lut", 19: "lut"}


@tiu_registry
class md_sum_op(tiu_base):
    description = "md sum"
    opcode = 7
    eu_type = {18: "mdsum"}


@tiu_registry
class md_scalar_op(tiu_base):
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


@tiu_registry
class md_sfu_op(tiu_base):
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


@tiu_registry
class md_linear_op(tiu_base):
    description = "md linear"
    opcode = 10
    eu_type = {
        1: "mdlinear.mac",
        20: "mdlinear.square_sum",
        21: "mdlinear.square_diff",
    }


@tiu_registry
class lma_op(tiu_base):
    description = "local memory arrangement"
    opcode = 11
    eu_type = {19: "lmem_arrangement"}


@tiu_registry
class decompress_op(tiu_base):
    description = "decompress"
    opcode = 12
    eu_type = {19: "decompress"}


@tiu_registry
class md_cmp_op(tiu_base):
    description = "md cmp"
    opcode = 13
    eu_type = {22: "mdcmp.cmp", 23: "mdcmp.select", 24: "mdcmp.cmp_select"}


@tiu_registry
class vc_op(tiu_base):
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


# ------------------------------------------------------------
# GDMA definition
# ------------------------------------------------------------
class dma_base(DMABase):
    length = 1024
    description = "GDMA Operation."
    opcode_bits = (0, 1)
    fun_bits = (32, 35)
    sp_fun = ()

    def _decode(self):
        cmd_bits = buffer_to_bits(self.cmd)
        self.reg = NamedDict(decode_reg(cmd_bits, self.reg_def))
        self.cmd_id = self.reg.cmd_id
        self.cmd_id_dep = self.reg.eng0_sync_id
        if self.sp_fun:
            self.op_name = self.sp_fun[self.reg.special_func]

    def _is_comp(self, cmd_buf):
        if len(cmd_buf) * 8 < self.length:
            return False
        if extract_buf(cmd_buf, self.opcode_bits) != self.opcode:
            return False
        sp_fun_id = extract_buf(cmd_buf, self.fun_bits)
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


@dma_registry
class dma_tensor(dma_base):
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


# cache for convert binary to unsigned integer.
_TABLE = 2 ** np.arange(64, dtype=np.uint64)

def buffer_to_bits(buffer):
    cmmand_buf = np.frombuffer(buffer, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")

def packbits(arr):
    if arr.size > 64:
        return 0
    return int(arr.dot(_TABLE[: arr.size]))

def decode_reg(buffer, des_reg):
    bits_sec = np.split(buffer, des_reg["high_bit"])  # slow
    value = (packbits(x) for x in bits_sec)  # slow
    return dict(zip(des_reg["fields"], value))

def op_factory(engine_type):

    if engine_type == Engine.TIU:
        opcode_bits = tiu_base.opcode_bits
        cmd_set = tiu_cls
    elif engine_type == Engine.DMA:
        opcode_bits = dma_base.opcode_bits
        cmd_set = dma_cls
    else:
        raise ValueError(f"cannot decode engine type: {engine_type}")

    def is_end(cmd_buf, operation):
        cmd_buf_bits = buffer_to_bits(cmd_buf)
        is_less_1024 = len(cmd_buf_bits) < 1025
        if is_less_1024 and not np.any(cmd_buf_bits):
            return True
        return False

    def decoder(cmd_buf):
        nonlocal opcode_bits, cmd_set
        cmd_key = extract_buf(cmd_buf, opcode_bits)
        if cmd_key in cmd_set:
            for op in cmd_set[cmd_key]:
                if op.is_comp(cmd_buf):
                    return op.decode(cmd_buf)
        raise ValueError(f"cannot decode cmd: {cmd_buf}")

    return decoder, is_end



def merge_instruction(tiu, dma):
    main_cmd, inserted_cmd = dma, tiu
    # remove the system command
    def get_end(cmd):
        if len(cmd) == 0:
            return 0
        sys = (tiu_sys, dma_sys)
        if all(sys):
            if isinstance(cmd[-1], sys):
                return -1
        else:
            return len(cmd)
    # remove system instruction
    main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
    inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[: get_end(inserted_cmd)]]
    # "sorted" is stable, which keeps the inserted commands
    # after the main instructions.
    cmd = main_id + inserted_id
    cmd_sorted = sorted(cmd, key=lambda x: x[0])
    return [x[1] for x in cmd_sorted]
