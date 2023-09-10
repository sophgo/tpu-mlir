#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from debugger import disassembler
from debugger import opdef_1684x
from debugger import context
import numpy as np


def get_nop_conv():
    import ctypes

    class Nop:
        cmd = None
        reg = None

    nop = Nop()
    nop.cmd = bytes(np.zeros(16, dtype=np.uint64))
    nop.reg = ctypes.cast(nop.cmd, ctypes.POINTER(opdef_1684x.conv_op.reg_def)).contents

    cmd = nop.reg
    cmd.cmd_short = 0
    cmd.cmd_id = 0
    cmd.cmd_id_dep = 0
    cmd.tsk_typ = 0
    cmd.tsk_eu_typ = 0
    cmd.eu_half_en = 0
    cmd.tsk_opd_num = 2
    cmd.pad_mode = 0
    cmd.cmd_id_en = 1
    cmd.tsk_lane_num = 0
    cmd.res0_n = 1
    cmd.res0_c = 64
    cmd.res0_h = 64 * 8
    cmd.res0_w = 64
    cmd.opd0_n = 1
    cmd.opd0_c = 64
    cmd.opd0_h = 64 * 8
    cmd.opd0_w = 64
    cmd.opd1_n = 1
    cmd.opd1_c = 64
    cmd.opd1_h = 1
    cmd.opd1_w = 1
    cmd.res0_n_str = 64 * 64 * 8
    cmd.res0_c_str = 64 * 64 * 8
    cmd.res0_h_str = 64
    cmd.res0_w_str = 1
    cmd.opd0_n_str = 64 * 64 * 8
    cmd.opd0_c_str = 64 * 64 * 8
    cmd.opd0_h_str = 64
    cmd.opd0_w_str = 1
    cmd.opd1_n_str = 1
    cmd.opd1_c_str = 1
    cmd.opd1_h_str = 1
    cmd.opd1_w_str = 1
    cmd.res0_addr = 131072
    cmd.opd0_addr = 0
    cmd.opd1_addr = 0
    return nop


def padding_compute(bmodel):
    cmd_group = bmodel.net[0].parameter[0].cmd_group[0]
    tiu_cmd = cmd_group.tiu_cmd
    ctx = context.Context(bmodel.chip)
    tiu = list(ctx.decoder.decode_tiu_buf(tiu_cmd))
    tiu_ex = []

    for x in tiu[:-1]:
        # opdef_1684x.conv_op.reg_def
        conv_cmd = get_nop_conv()
        conv_cmd.reg.cmd_id = x.cmd_id - 1
        tiu_ex.extend([conv_cmd] * 20 + [x])
    # the command id of the last one should be the same as number of instructions.
    tiu[-1].reg.cmd_id = len(tiu_ex) + 1
    tiu_ex.append(tiu[-1])
    tiu_buf = []
    for x in tiu_ex:
        tiu_buf.extend(x.cmd)
    size = len(tiu_buf)
    padding = (size + 127) // 128 * 128 - size  # 128 bytes alignment
    tiu_buf.extend(bytes([0] * padding))
    cmd_group.tiu_cmd = tiu_buf
    cmd_group.tiu_num = len(tiu_ex)
    return tiu_ex


if __name__ == "__main__":
    import sys

    bmodel = disassembler.BModel(sys.argv[1])
    padding_compute(bmodel)
    bmodel.serialize(sys.argv[2])
