#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import ctypes
from copy import copy
import os
from debugger import disassembler
from debugger.atomic_dialect import BModel2MLIR
from debugger.target_common import CMDType


def main():
    parser = argparse.ArgumentParser(
        description="Truncate BModel and generate a new BModel."
    )
    parser.add_argument(
        "bmodel",
        help="The input BModel.",
    )
    parser.add_argument(
        "out_file",
        help="Truncated bmodel.",
    )
    parser.add_argument(
        "tp",
        help="Target Truncate-Point concat with {core id:tiu/dma:cmd_id}.",
    )

    args = parser.parse_args()
    return args


def trans_cmds_to_buf(cmds, engine_type):
    buf_list = []
    for cmd in cmds:
        reg = copy(cmd.reg)
        if engine_type == 1:  # dma
            u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)
            buf = bytes(u32_buf)
        else:
            buf = bytes(reg)
        buf_list.append(buf)
    return b"".join(buf_list)


def get_index_by_cmd_id(cmds, target_cmd_id, target_cmd_type):
    target_cmd_type = get_cmd_type(target_cmd_type)

    for cmd_index, cmd in enumerate(cmds):
        if cmd.cmd_type == target_cmd_type and cmd.cmd_id == target_cmd_id:
            return cmd_index
        else:
            return -1


def get_cmd_type(target_cmd_type):
    if target_cmd_type == "tiu":
        target_cmd_type = CMDType.tiu
    elif target_cmd_type == "dma":
        target_cmd_type = CMDType.dma
    else:
        raise RuntimeError("Not supported cmd type!")
    return target_cmd_type


def padding_to_128_align(buf):
    size = len(buf)
    padding = (size + 127) // 128 * 128 - size  # 128 bytes alignment
    buf += bytes([0] * padding)
    return buf


def serialize(args, bmodel, func_id, reg_id, block_id, block):
    target_core_id, targt_cmd_type, target_cmd_id = args.tp.split(":")

    if bmodel.chip == "BM1684X":
        from debugger.target_1684x.regdef import SYSID_reg, sDMA_sys_reg
    elif bmodel.chip in ("BM1688", "CV186X"):
        from debugger.target_1688.regdef import SYS_reg, sDMA_sys_reg
    elif bmodel.chip == "SG2260":
        from debugger.target_2260.regdef import SYS_reg, sDMA_sys_reg

    assert block.subnet.run_mode == block.subnet.run_mode.TPU_STATIC
    truncate_msgcore_id = 0
    if bmodel.core_num > 1:
        assert len(block.cores_cmds) > 0
        # get truncate_msgcore_id
        for core_id, core_cmds in enumerate(block.cores_cmds):
            if core_id == int(target_core_id):
                for msgcore_id, msgcore in enumerate(core_cmds.msgcores):
                    if targt_cmd_type == "tiu":
                        boundary_type, start_id, end_id = msgcore.get_CmdId_boundary(CMDType.tiu)
                    else:
                        boundary_type, start_id, end_id = msgcore.get_CmdId_boundary(CMDType.dma)

                    if start_id <= int(target_cmd_id) <= end_id:
                        truncate_msgcore_id = msgcore_id
                        break

        # truncate
        for core_id, core_cmds in enumerate(block.cores_cmds):
            new_all = []
            new_tiu = []
            new_dma = []
            # truncate point in no_sys_cmds
            if boundary_type == 0:
                assert int(target_core_id) == 0
                new_all = sum(core_cmds.msgcores[:truncate_msgcore_id], [])
                if core_id == 0:
                    target_index = get_index_by_cmd_id(
                        core_cmds.msgcores[truncate_msgcore_id].no_sys_cmds,
                        int(target_cmd_id),
                        targt_cmd_type,
                    )
                    new_all += core_cmds.msgcores[truncate_msgcore_id].no_sys_cmds[: target_index + 1]

            # truncate point in sys_cmds
            elif boundary_type == 1:
                new_all = sum([msgcore.total_cmds for msgcore in core_cmds.msgcores[: truncate_msgcore_id + 1]],[])
                # add tiu/dma wait msg cmd
                if truncate_msgcore_id < len(core_cmds.msgcores) - 1:
                    new_all += core_cmds.msgcores[truncate_msgcore_id + 1].total_cmds[:2]

            for cmd in new_all:
                if cmd.cmd_type == CMDType.tiu:
                    new_tiu.append(cmd)
                elif cmd.cmd_type == CMDType.dma:
                    new_dma.append(cmd)

            print(f"new bmodel core{core_id} include {len(new_tiu)} tius, {len(new_dma)} dmas")
            tiu_buf = trans_cmds_to_buf(new_tiu, 0)
            dma_buf = trans_cmds_to_buf(new_dma, 1)

            tiu_sys_end = block.cmds[core_id].tiu[-1].reg
            dma_sys_end = block.cmds[core_id].dma[-1].reg
            tiu_sys_end_buf = block.cmds[core_id].tiu[-1].buf
            dma_sys_end_buf = block.cmds[core_id].dma[-1].buf
            if bmodel.chip == "BM1684X":
                assert isinstance(tiu_sys_end, SYSID_reg)
            else:
                assert isinstance(tiu_sys_end, SYS_reg)
            assert isinstance(dma_sys_end, sDMA_sys_reg)
            tiu_sys_end.cmd_id_dep = 0
            dma_sys_end.cmd_id_dep = 0
            tiu_buf += tiu_sys_end_buf
            dma_buf += dma_sys_end_buf

            tiu_buf = padding_to_128_align(tiu_buf)
            dma_buf = padding_to_128_align(dma_buf)

            bmodel.net[func_id].parameter[reg_id].sub_net[block_id].core_commands[core_id].gdma_tiu_commands[0].tiu_cmd.bytes = tiu_buf
            bmodel.net[func_id].parameter[reg_id].sub_net[block_id].core_commands[core_id].gdma_tiu_commands[0].tiu_num = (len(new_tiu) + 1)
            bmodel.net[func_id].parameter[reg_id].sub_net[block_id].core_commands[core_id].gdma_tiu_commands[0].dma_cmd.bytes = dma_buf
            bmodel.net[func_id].parameter[reg_id].sub_net[block_id].core_commands[core_id].gdma_tiu_commands[0].dma_num = (len(new_dma) + 1)
    else:
        assert len(block.cmds) == 1
        new_tiu = []
        new_dma = []

        for cmd in block.cmds[0].all:
            if cmd.cmd_type == CMDType.tiu:
                new_tiu.append(cmd)
            elif cmd.cmd_type == CMDType.dma:
                new_dma.append(cmd)

            if cmd.cmd_type == get_cmd_type(targt_cmd_type) and cmd.cmd_id == int(target_cmd_id):
                break

        print(f"new bmodel include {len(new_tiu)} tius, {len(new_dma)} dmas")
        tiu_buf = trans_cmds_to_buf(new_tiu, 0)
        dma_buf = trans_cmds_to_buf(new_dma, 1)

        tiu_sys_end = block.cmds[0].tiu[-1].reg
        dma_sys_end = block.cmds[0].dma[-1].reg
        tiu_sys_end_buf = block.cmds[0].tiu[-1].buf
        dma_sys_end_buf = block.cmds[0].dma[-1].buf
        if bmodel.chip == "BM1684X":
            assert isinstance(tiu_sys_end, SYSID_reg)
        else:
            assert isinstance(tiu_sys_end, SYS_reg)
        assert isinstance(dma_sys_end, sDMA_sys_reg)
        tiu_sys_end.cmd_id_dep = 0
        dma_sys_end.cmd_id_dep = 0
        tiu_buf += tiu_sys_end_buf
        dma_buf += dma_sys_end_buf

        tiu_buf = padding_to_128_align(tiu_buf)
        dma_buf = padding_to_128_align(dma_buf)

        bmodel.net[func_id].parameter[reg_id].sub_net[block_id].cmd_group[0].tiu_cmd.bytes = tiu_buf
        bmodel.net[func_id].parameter[reg_id].sub_net[block_id].cmd_group[0].tiu_num = len(new_tiu) + 1
        bmodel.net[func_id].parameter[reg_id].sub_net[block_id].cmd_group[0].dma_cmd.bytes = dma_buf
        bmodel.net[func_id].parameter[reg_id].sub_net[block_id].cmd_group[0].dma_num = len(new_dma) + 1
    return bmodel


def truncate(bmodel, args):
    MlirModel = BModel2MLIR(bmodel)
    for func_id, function in enumerate(MlirModel.functions):
        for reg_id, region in enumerate(function.regions):
            for block_id, block in enumerate(region.blocks):
                bmodel = serialize(args, bmodel, func_id, reg_id, block_id, block)

    bmodel.serialize(args.out_file)


if __name__ == "__main__":
    args = main()
    bmodel_file = os.path.abspath(args.bmodel)
    bmodel = disassembler.BModel(bmodel_file)
    truncate(bmodel, args)
