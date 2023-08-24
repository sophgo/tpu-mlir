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
    from . import regdef_1686
    from .opparam_1686 import opparam_converter
    from .op_support import extract_buf, reg_decoder_factory, TIUBase, DMABase, Engine
except:
    import regdef_1686
    from opparam_1686 import opparam_converter
    from op_support import extract_buf, reg_decoder_factory, TIUBase, DMABase, Engine

import ctypes
import numpy as np
from enum import Enum

# global data and type
# ------------------------------------------------------------
tiu_cls = dict()
dma_cls = dict()

# ------------------------------------------------------------
# registry function
# ------------------------------------------------------------


def base_registry(cmd_type, sheet_name, cls):
    reg_def = regdef_1686.reg_def[sheet_name]
    setattr(cls, "length", reg_def[-1][-1])
    setattr(cls, "reg_def", reg_decoder_factory(reg_def))
    cmd_type.setdefault(cls.opcode, set()).add(cls)
    if sheet_name in opparam_converter:
        setattr(cls, "_set_op", staticmethod(opparam_converter[sheet_name]))
    return cls


def tiu_registry(sheet_name):
    def decorate(cls):
        return base_registry(tiu_cls, sheet_name, cls)

    return decorate


def dma_registry(sheet_name):
    def decorate(cls):
        return base_registry(dma_cls, sheet_name, cls)

    return decorate
# ------------------------------------------------------------
# TIU definition
# ------------------------------------------------------------


class tiu_base(TIUBase):

    opcode_bits = (41, 45)
    # extension
    eu_type = ()
    eu_bits = (45, 50)
    short_cmd = False  # long_code by default

    def _decode(self):
        self.reg = ctypes.cast(self.cmd, ctypes.POINTER(self.reg_def)).contents
        if self.opcode == 12:
            self.cmd_id_dep = 0
        else:
            self.cmd_id_dep = self.reg.cmd_id_dep
        self.op_name = self.eu_type[self.reg.tsk_eu_typ]

    def _is_comp(self, cmd_reg):
        if len(cmd_reg) * 8 < self.length:
            return False
        if self.short_cmd is not None and bool(extract_buf(cmd_reg, [0, 1])) != self.short_cmd:
            return False
        if extract_buf(cmd_reg, self.opcode_bits) != self.opcode:
            return False
        if extract_buf(cmd_reg, self.eu_bits) not in self.eu_type:
            return False
        return True

    def ops(self, is_arch):
        raise NotImplementedError()

    def __repr__(self):
        ci = self.core_id
        if self.operands == []:
            if self.attribute:
                attribute = f" {self.attribute}".replace(
                    ":", " =").replace("'", "")
                return (
                    # f"core_id: {self.core_id} " +
                    f"%B{self.cmd_id}C{ci} = \"{self.op_name}\""
                    + f"(%D{self.cmd_id_dep}C{ci})"
                    + attribute
                )
            else:
                return self.description
        res_name, res_type_t = zip(*((x.name, x.type_str)
                                   for x in self.results))
        opd_name, opd_type_t = zip(*((x.name, x.type_str)
                                   for x in self.operands))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(
                ":", " =").replace("'", "")

        return (
            # f"core_id: {self.core_id} " +
            f"{', '.join(res_name)}, %B{self.cmd_id}C{ci} = \"{self.op_name}\""
            + f"({', '.join(opd_name)}, %D{self.cmd_id_dep}C{ci})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({', '.join(res_type_t)}, none)"
        )


@tiu_registry("CONV")
class conv_op(tiu_base):
    opcode = 0
    eu_type = {0: "conv.normal"}
    description = "convolution"

    def ops(self, is_arch=False):
        return 0


@tiu_registry("sCONV")
class sconv_op(conv_op):
    short_cmd = True
    description = "short convolution"


@tiu_registry("MM")
class mm_op(tiu_base):
    opcode = 2
    eu_type = {1: "mm.normal"}
    description = "matrix multiply"

    def ops(self, is_arch=False):
        return 0


@tiu_registry("sMM")
class smm_op(mm_op):
    short_cmd = True
    description = "short matrix multiply"


@tiu_registry("MM2")
class mm2_op(tiu_base):
    opcode = 2
    eu_type = {4: "mm2.nn", 5: "mm2.nt", 6: "mm2.tt"}
    description = "matrix multiply2"

    def ops(self, is_arch=False):
        return 0


@tiu_registry("sMM2")
class smm2_op(mm2_op):
    short_cmd = True
    description = "short matrix multiply2"


@tiu_registry("CMP")
class cmp_op(tiu_base):
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
        return 0


@tiu_registry("sCMP")
class scmp_op(cmp_op):
    short_cmd = True
    description = "short fused_cmpare"


@tiu_registry("SFU")
class sfu_op(tiu_base):
    opcode = 9
    eu_type = {
        12: "sfu.taylor_4x",
        13: "sfu.taylor",
        15: "sfu.normalize",
        17: "sfu.rsqrt",
    }
    description = "special_function"

    def ops(self, is_arch=False):
        return 0


@tiu_registry("sSFU")
class ssfu_op(sfu_op):
    short_cmd = True
    description = "short special_function"


@tiu_registry("LIN")
class lin_op(tiu_base):
    opcode = 10
    eu_type = {1: "lin.mac", 20: "lin.square_sum", 21: "lin.square_diff"}
    description = "fused_linear"

    def ops(self, is_arch):
        return 0


@tiu_registry("sLIN")
class slin_op(lin_op):
    short_cmd = True
    description = "short fused_linear"


@tiu_registry("VC")
class vc_op(tiu_base):
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
        return 0


@tiu_registry("sVC")
class svc_op(vc_op):
    short_cmd = True
    description = "short vector correlation"


@tiu_registry("AR")
class ar_op(tiu_base):
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
        return 0


@tiu_registry("sAR")
class sar_op(ar_op):
    short_cmd = True
    description = "short arithmetic"


@tiu_registry("PorD")
class pord_op(tiu_base):
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
        return 0


@tiu_registry("sPorD")
class spord_op(pord_op):
    short_cmd = True
    description = "short depthwise or pooling"


@tiu_registry("RQ&DQ")
class rqdq_op(tiu_base):
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
        return 0


@tiu_registry("sRQ&sDQ")
class srqdq_op(rqdq_op):
    short_cmd = True
    description = "short RQ && DQ"


@tiu_registry("SG")
class sg_op(tiu_base):
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
        return 0


@tiu_registry("sSG")
class ssg_op(sg_op):
    short_cmd = True
    description = "short scatter_gather"


@tiu_registry("SGL")
class sgl_op(tiu_base):
    opcode = 6
    eu_type = {17: "sgl.pe_s_gather_line", 18: "sgl.pe_s_scatter_line"}
    description = "scatter_gather_line"

    def ops(self, is_arch):
        return 0


@tiu_registry("sSGL")
class ssgl_op(sgl_op):
    short_cmd = True
    description = "short scatter_gather_line"


@tiu_registry("CW&BC")
class transbc_op(tiu_base):
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
        return 0


@tiu_registry("sCW&sBC")
class stransbc_op(transbc_op):
    short_cmd = True
    description = "short TRANS && BC"


@tiu_registry("SYS_TR_ACC")
class tiu_sys_tr_acc(tiu_base):
    opcode = 12
    short_cmd = None
    eu_bits = (45, 48)
    eu_type = {
        0: "system_tr_wr.wr_imm",
    }
    description = "system tr wr"

    def ops(self, is_arch):
        return 1


@tiu_registry("SYS")
class tiu_sys(tiu_base):
    opcode = 15
    short_cmd = None
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
        self.reg = ctypes.cast(self.cmd, ctypes.POINTER(self.reg_def)).contents
        self.cmd_id_dep = self.reg.cmd_id_dep
        if self.sp_fun:
            self.op_name = self.sp_fun[self.reg.cmd_special_function]

    def _is_comp(self, cmd_reg):
        if len(cmd_reg) * 8 < self.length:
            return False
        if self.short_cmd is not None and bool(extract_buf(cmd_reg, [3, 4])) != self.short_cmd:
            return False
        if extract_buf(cmd_reg, self.opcode_bits) != self.opcode:
            return False
        sp_fun_id = extract_buf(cmd_reg, self.fun_bits)
        if self.sp_fun and (sp_fun_id not in self.sp_fun):
            return False
        return True

    def __repr__(self):
        ci = self.core_id
        if self.operands == []:
            if self.attribute:
                attribute = f" {self.attribute}".replace(
                    ":", " =").replace("'", "")
                return (
                    # f"core_id: {self.core_id} " +
                    f"%D{self.cmd_id}C{ci} = \"{self.op_name}\""
                    + f"(%B{self.cmd_id_dep}C{ci})"
                    + attribute
                )
            else:
                return self.description
        opd_name, opd_type_t = zip(*((x.name, x.type_str)
                                   for x in self.operands))
        res_name, res_type_t = zip(*((x.name, x.type_str)
                                   for x in self.results))
        attribute = ""
        if self.attribute:
            attribute = f" {self.attribute}".replace(
                ":", " =").replace("'", "")

        return (
            # f"core_id: {self.core_id} " +
            f"{', '.join(res_name)}, %D{self.cmd_id}C{ci} = \"{self.op_name}\""
            + f"({', '.join(opd_name)}, %B{self.cmd_id_dep}C{ci})"
            + attribute
            + f" : ({', '.join(opd_type_t)}, none) -> ({res_type_t[0]}, none)"
        )

    def ops(self, is_arch):
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


@dma_registry("DMA_masked_select")
class dma_masked_select(dma_base):
    opcode = 2
    sp_fun = {
        0: "dma.masked_select",
        1: "dma.masked_select.ncw",
    }
    description = "DMA masked select"


@dma_registry("DMA_general")
class dma_general(dma_base):
    opcode = 3
    sp_fun = {
        0: "dma.general",
        1: "dma.general.broadcast",
    }
    description = "DMA general"


@dma_registry("DMA_cw_transpose")
class dma_cw_transpose(dma_base):
    opcode = 4
    sp_fun = {0: "dma.cw_transpose"}
    description = "DMA CW Transpose"


@dma_registry("DMA_nonzero")
class dma_nonzero(dma_base):
    opcode = 5
    sp_fun = {0: "dma.nonzero"}
    description = "DMA nonzero"


@dma_registry("sDMA_sys")
class dma_sys(dma_base):
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


@dma_registry("DMA_gather")
class dma_gather(dma_base):
    opcode = 7
    sp_fun = {0: "gdma.gather"}
    description = "DMA gather"


@dma_registry("DMA_scatter")
class dma_scatter(dma_base):
    opcode = 8
    sp_fun = {0: "gdma.scatter"}
    description = "DMA scatter"


@dma_registry("DMA_reverse")
class dma_scatter(dma_base):
    opcode = 9
    sp_fun = {
        0: "dma.reverse.w",
        1: "dma.reverse.h",
        2: "dma.reverse.c",
        3: "dma.reverse.n",
    }
    description = "DMA reverse"


@dma_registry("DMA_compress")
class dma_scatter(dma_base):
    opcode = 10
    sp_fun = {
        0: "dma.compress.non_random_access",
        1: "dma.compress.random_access",
    }
    description = "DMA compress"


@dma_registry("DMA_decompress ")
class dma_scatter(dma_base):
    opcode = 11
    sp_fun = {
        0: "dma.decompress.non_random_access",
        1: "dma.decompress.random_access",
    }
    description = "DMA decompress"


def op_factory(engine_type):
    cmd_id = 1
    if engine_type == Engine.TIU:
        opcode_bits = tiu_base.opcode_bits
        cmd_set = tiu_cls

        def sys_end(operation):
            if isinstance(operation, tiu_sys) and \
                    operation.reg.tsk_eu_typ == 31:
                return True
            else:
                return False
    elif engine_type == Engine.DMA:
        opcode_bits = dma_base.opcode_bits
        cmd_set = dma_cls

        def sys_end(operation):
            if isinstance(operation, dma_sys) and \
                    operation.reg.cmd_special_function == 0:
                return True
            else:
                return False
    else:
        raise ValueError(f"cannot decode engine type: {engine_type}")

    def is_end(cmd_buf, operation):
        nonlocal sys_end
        is_sys = sys_end(operation)
        is_less_1024 = len(cmd_buf) * 8 < 1025
        if is_sys and is_less_1024 and not np.any(np.frombuffer(cmd_buf, np.uint8)):
            return True
        return False

    def decoder(cmd_buf):
        nonlocal opcode_bits, cmd_set, cmd_id
        cmd_key = extract_buf(cmd_buf, opcode_bits)
        if cmd_key in cmd_set:
            for op in cmd_set[cmd_key]:
                if op.is_comp(cmd_buf):
                    operation = op.decode(cmd_buf)
                    operation.cmd_id = cmd_id
                    cmd_id += 1
                    return operation
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

    def fix_tgcr_cmd_id_dp(tiu_cmd):
        for i, v in enumerate(tiu_cmd):
            if v.opcode == 12:
                v.cmd_id_dep = (
                    tiu_cmd[i + 1].cmd_id_dep
                    if tiu_cmd[i + 1].cmd_id_dep != 0
                    else tiu_cmd[i + 2].cmd_id_dep
                )

    fix_tgcr_cmd_id_dp(inserted_cmd[: get_end(inserted_cmd)])
    # remove system instruction
    main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
    inserted_id = [(i.cmd_id_dep, i)
                   for i in inserted_cmd[: get_end(inserted_cmd)]]
    # "sorted" is stable, which keeps the inserted commands
    # after the main instructions.
    cmd = main_id + inserted_id
    cmd_sorted = sorted(cmd, key=lambda x: x[0])
    return [x[1] for x in cmd_sorted]


"""
This essentially simulates the engine synchronization mechanism of cmodel/hardware

Think of the instruction list for each core as a state machine.
The principle of consumption instructions:
if it is a normal instruction direct consumption, if it is a sys instruction,
it will be consumed according to the rules
1. The send command can be directly consumed
2. The wait command requires send_cnt to meet the requirements before it can be consumed
Assuming that a core instruction is in an unconsumable state
(wait is not up to the standard or all instructions have been consumed),
switch the current state machine

The sender sends a message to the message queue,
and the message queue adds 1 to the sent_cnt field
where the ID is located after receiving the message,
and sets the remain_wait_cnt field to wait_cnt;
the receiver polls whether the sent_cnt at the
corresponding position in the message queue has
met the requirements, if it is satisfied,
decrease reamin_wait_cnt by one.
When it is reduced to 0, this position is cleared to 0,
and the process of message synchronization is completed so far
"""


class SYS_TYPE(Enum):
    SEND = 0
    WAIT = 1

class Status(Enum):
    # for msg use
    CONSUMED = 0
    WAITING = 1
    # for machine use
    RUNNING = 2
    MACHINE_DONE = 3



class CmdStateMachine:
    def __init__(self, cmds, msg_queue):
        self.cmds = cmds
        self.idx = 0
        self.msg_queue = msg_queue

    def consume_cmd(self):
        if self.consume_done():
            return None, Status.MACHINE_DONE
        cmd = self.cmds[self.idx]
        sys = (tiu_sys, dma_sys)
        if isinstance(cmd, sys):
            return self.consume_sys(cmd)
        else:
            self.idx += 1
        return cmd, Status.RUNNING

    def consume_sys(self, cmd):
        cmd, ret = self.msg_queue.consume_sys(cmd)
        if ret == Status.CONSUMED:
            self.idx += 1
            return cmd, Status.RUNNING
        else:
            return None, ret

    def consume_done(self):
        return self.idx >= len(self.cmds)


class Msg:
    def __init__(self):
        self.sent_cnt = 0
        self.wait_cnt = 0


class MsgQueue:
    def __init__(self):
        self.msges = [Msg()] * 128

    def get_cmd_type(self, cmd):
        if isinstance(cmd, tiu_sys):
            if cmd.reg.tsk_eu_typ == 8:
                return SYS_TYPE.SEND
            elif cmd.reg.tsk_eu_typ == 9:
                return SYS_TYPE.WAIT
            else:
                raise ValueError(f"cmd type error: {cmd}")
        elif isinstance(cmd, dma_sys):
            if cmd.reg.cmd_special_function == 3:
                return SYS_TYPE.SEND
            elif cmd.reg.cmd_special_function == 4:
                return SYS_TYPE.WAIT
            else:
                raise ValueError(f"cmd type error: {cmd}")
        else:
            raise ValueError(f"cmd type error: {cmd}")

    def get_msg_id(self, cmd):
        return cmd.attribute["msg_id"]

    def get_msg_cnt(self, cmd):
        return cmd.attribute["cnt"]

    def consume_sys(self, cmd):
        sys = (tiu_sys, dma_sys)
        assert isinstance(cmd, sys)
        if self.get_cmd_type(cmd) == SYS_TYPE.SEND:
            return self.consume_send(cmd)
        elif self.get_cmd_type(cmd) == SYS_TYPE.WAIT:
            return self.consume_wait(cmd)

    def consume_send(self, cmd):
        msg_id = self.get_msg_id(cmd)
        self.msges[msg_id].sent_cnt += 1
        self.msges[msg_id].remain_wait_cnt = self.get_msg_cnt(cmd)
        return cmd, Status.CONSUMED

    def consume_wait(self, cmd):
        msg_id = self.get_msg_id(cmd)
        if self.get_msg_cnt(cmd) != self.msges[msg_id].sent_cnt:
            return None, Status.WAITING
        else:
            self.msges[msg_id].remain_wait_cnt -= 1
            if self.msges[msg_id].remain_wait_cnt == 0:
                self.msges[msg_id].sent_cnt = 0
            return cmd, Status.CONSUMED


class MultiCoreCmd:
    def __init__(self, core_cmds):
        self.core_cmds = core_cmds
        self.core_num = len(core_cmds)
        self.cur_core_id = 0
        self.msg_queue = MsgQueue()
        self.cmd_state_machines = [CmdStateMachine(
            core_cmds[i], self.msg_queue) for i in range(self.core_num)]
        self.current_machine = self.cmd_state_machines[self.cur_core_id]
        self.cmds = []

    def switch_machine(self):
        self.cur_core_id = (self.cur_core_id + 1) % self.core_num
        self.current_machine = self.cmd_state_machines[self.cur_core_id]

    def all_consume_done(self):
        for i in self.cmd_state_machines:
            if not i.consume_done():
                return False
        return True

    def consume_cmd(self):
        while(not self.all_consume_done()):
            cmd, ret = self.current_machine.consume_cmd()
            self.switch_machine()
            if ret not in  (Status.MACHINE_DONE, Status.WAITING) :
                self.cmds.append(cmd)
        return self.cmds
