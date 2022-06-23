#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
import sys
from collections import namedtuple
import numpy as np
import tpu1684x_opdef
import bmodel_fbs
import itertools

# Example:
"""
# bmodel file
code = TPUCMD(file_name)
code.cmd[0].all

# bdc binary file
bdc_code = decode_bdc(file_name)

# gdma binary file
gama_code = decode_gdma(file_name)
"""


def read_file(cmd_file):
    cmmand_buf = np.fromfile(cmd_file, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


def read_buf(cmd_buf):
    cmmand_buf = np.frombuffer(cmd_buf, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


class BmodelReader:
    header_t = np.dtype(
        [
            ("magic", np.uint32),
            ("header_size", np.uint32),
            ("flatbuffers_size", np.uint32),
            ("binary_size", np.uint32),
            ("reserved", np.uint32, 12),
        ]
    )

    class cmd_group_cls:
        def __init__(self, fbs: bmodel_fbs.CmdGroup, cmd_buf):
            self.bdc_num = fbs.BdcNum()
            self.gdma_num = fbs.GdmaNum()
            self.binary_bdc = (fbs.BinaryBdc().Start(), fbs.BinaryBdc().Size())  # type: ignore
            self.binary_gdma = (fbs.BinaryGdma().Start(), fbs.BinaryGdma().Size())  # type: ignore
            self.bdc_cmd = cmd_buf[self.binary_bdc[0] : sum(self.binary_bdc)]
            self.gdma_cmd = cmd_buf[self.binary_gdma[0] : sum(self.binary_gdma)]

        def __repr__(self):
            return f"bdc_num: {self.bdc_num}\ngdma_num: {self.gdma_num}"

    def __init__(self, bmodel_file):
        self.head = None
        self.binary_desc = None
        self.binary = None
        with open(bmodel_file, "rb") as file_obj:
            file_obj.seek(0, 0)
            self.head = np.frombuffer(
                file_obj.read(self.header_t.itemsize), dtype=self.header_t
            )
            self.binary_desc = file_obj.read(self.head["flatbuffers_size"][0])
            self.binary = file_obj.read(self.head["binary_size"][0])
        bmodel = bmodel_fbs.Model.GetRootAsModel(self.binary_desc, 0)
        self.cmd_group = []
        for n in range(bmodel.NetLength()):
            net = bmodel.Net(n)
            if net is None:
                return
            for p in range(net.ParameterLength()):
                param = net.Parameter(p)
                if param is None:
                    return
                for s in range(param.CmdGroupLength()):
                    cmd = param.CmdGroup(s)
                    if cmd is None:
                        return
                    self.cmd_group.append(self.cmd_group_cls(cmd, self.binary))


class TPUCMD:
    _cmd = namedtuple("cmd", ["bdc", "gdma", "all"])

    def __init__(self, bmodel_file):
        self.bmodel = BmodelReader(bmodel_file)
        self.cmd = []
        for cmd in self.bmodel.cmd_group:
            bdc_cmd = read_buf(cmd.bdc_cmd)
            gdma_cmd = read_buf(cmd.gdma_cmd)
            bdc = itertools.islice(self.decode_bdc(bdc_cmd), cmd.bdc_num)
            gdma = itertools.islice(self.decode_gdma(gdma_cmd), cmd.gdma_num)
            bdc = list(bdc)
            gdma = list(gdma)
            self.cmd.append(self._cmd(bdc, gdma, self.merge_cmd(gdma, bdc)))

    @staticmethod
    def __decode(cmd_buf, cmd_bits, cmd_set, sys_end):
        code = None
        cur = 0
        l, h = cmd_bits
        while cmd_buf.size > 0:
            cmd_key = tpu1684x_opdef.packbits(cmd_buf[l:h])
            if cmd_key in cmd_set:
                recognize = False
                for op in cmd_set[cmd_key]:
                    if op.is_comp(cmd_buf):
                        # check whether this command is recognized by the operation
                        code = op(cmd_buf)
                        yield code
                        # consume this command_code
                        cmd_buf = cmd_buf[op.len :]
                        cur += op.len
                        recognize = True
                        break
                is_sys = isinstance(code, sys_end)
                is_less_1024 = cmd_buf.size < 1025
                is_all_zeros = np.all(cmd_buf == 0)
                if is_sys and is_less_1024 and is_all_zeros:
                    break  # all the BDC have been processed
                if not recognize:
                    raise ValueError(
                        "Can not decode cmd, with opcode: {}, at {}.".format(
                            cmd_key, cur
                        )
                    )
            else:
                raise ValueError(
                    "Can not decode cmd, with opcode: {}, at {}.".format(cmd_key, cur)
                )

    @staticmethod
    def decode_bdc(cmd_buf):
        return TPUCMD.__decode(
            cmd_buf,
            tpu1684x_opdef.bdc_base.cmd_bits,
            tpu1684x_opdef.bdc_cmd,
            tpu1684x_opdef.sysid_op,
        )

    @staticmethod
    def decode_gdma(cmd_buf):
        return TPUCMD.__decode(
            cmd_buf,
            tpu1684x_opdef.dma_base.cmd_bits,
            tpu1684x_opdef.dma_cmd,
            tpu1684x_opdef.sdma_sys,
        )

    @staticmethod
    def merge_cmd(main_cmd, inserted_cmd):
        # remove the system command
        main_id = [(m.cmd_id, m) for m in main_cmd[:-1]]
        inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[:-1]]
        # "sorted" is stable, which keeps the inserted commands
        # after the main instructions.
        cmd = main_id + inserted_id
        cmd_sorted = sorted(cmd, key=lambda x: x[0])
        return [x[1] for x in cmd_sorted]


def decode_bdc(file_name):
    a = read_file(file_name)
    return [x for x in TPUCMD.decode_bdc(a)]


def decode_gdma(file_name):
    a = read_file(file_name)
    return [x for x in TPUCMD.decode_gdma(a)]


if __name__ == "__main__":
    args = sys.argv
    assert (
        len(args) == 2
    ), f"The input should be a bmodel file. but more arguments are provided {args}"
    tpu_cmd = TPUCMD(args[1])
    for idx, cmd in enumerate(tpu_cmd.cmd):
        fmt_cmd = ["\n    " + str(x) for x in cmd.all]
        fmt_cmd = "".join(fmt_cmd) + "\n"
        print(f"Net({idx}) {{{fmt_cmd}}}")
