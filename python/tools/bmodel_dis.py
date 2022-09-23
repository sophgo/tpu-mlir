#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import sys
from collections import namedtuple
import numpy as np
from utils.bmodel_dis import opdef_1684x
from utils.bmodel_dis import bmodel_fbs
import itertools

# Example:
"""
# bmodel file
code = TPUCMD(file_name)


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

        def get_cmd(param, _fields):
            field, *_fields = _fields
            mult = []
            for s in range(getattr(param, field + "Length")()):
                cmd = getattr(param, field)(s)
                # if cmd is None:
                #     return mult
                if _fields == []:
                    mult.append(self.cmd_group_cls(cmd, self.binary))
                else:
                    mult.append(get_cmd(cmd, _fields))
            return mult

        # net{ stage{ subnet{ cmdgroup... }... }... }...
        fields = ["Net", "Parameter", "SubNet", "CmdGroup"]
        self.nets = get_cmd(bmodel, fields)


class TPUCMD:
    _cmd = namedtuple("cmd", ["bdc", "gdma", "all"])

    def decode_cmd(self, cmd):
        bdc_cmd = read_buf(cmd.bdc_cmd)
        gdma_cmd = read_buf(cmd.gdma_cmd)
        bdc = itertools.islice(self.decode_bdc(bdc_cmd), cmd.bdc_num)
        gdma = itertools.islice(self.decode_gdma(gdma_cmd), cmd.gdma_num)
        bdc = list(bdc)
        gdma = list(gdma)
        return self._cmd(bdc, gdma, self.merge_cmd(gdma, bdc))

    def __init__(self, bmodel_file):
        self.bmodel = BmodelReader(bmodel_file)

        def get_cmd(cmd, id):
            for idx, v in enumerate(cmd):
                id[-1] = idx
                if isinstance(v, list):
                    id.append(idx)
                    yield from get_cmd(v, id)
                else:
                    yield (id, self.decode_cmd(v))

        self.cmd = get_cmd(self.bmodel.nets, [0])

    @staticmethod
    def __decode(cmd_buf, cmd_bits, cmd_set, sys_end):
        code = None
        cur = 0
        l, h = cmd_bits
        while cmd_buf.size > 0:
            cmd_key = opdef_1684x.packbits(cmd_buf[l:h])
            if cmd_key in cmd_set:
                recognize = False
                for op in cmd_set[cmd_key]:
                    if op.is_comp(cmd_buf):
                        # check whether this command is recognized by the operation
                        code = op.decode(cmd_buf)
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
            opdef_1684x.bdc_base.cmd_bits,
            opdef_1684x.bdc_cmd,
            opdef_1684x.sysid_op,
        )

    @staticmethod
    def decode_gdma(cmd_buf):
        return TPUCMD.__decode(
            cmd_buf,
            opdef_1684x.dma_base.cmd_bits,
            opdef_1684x.dma_cmd,
            opdef_1684x.sdma_sys,
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


def unified_diff(a, b, fromfile="", tofile="", n=3, format="mlir"):
    r"""
    Compare two BModel of operations; generate the delta as a unified diff.

    Unified diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.
    """
    import difflib

    def fmt_op(op):
        if format == "raw":
            return str(op.attr)
        if format == "mlir":
            return str(op)
        if format == "bits":
            return "".join((str(x) for x in op.cmd))

    lineterm = "\n"
    started = False
    for group in difflib.SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            yield f"--- {fromfile}"
            yield f"+++ {tofile}"

        first, last = group[0], group[-1]
        file1_range = difflib._format_range_unified(first[1], last[2])
        file2_range = difflib._format_range_unified(first[3], last[4])
        yield "@@ -{} +{} @@{}".format(file1_range, file2_range, lineterm)

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in a[i1:i2]:
                    yield "    " + fmt_op(line)
                continue
            if tag in {"replace", "delete"}:
                for line in a[i1:i2]:
                    yield "-   " + fmt_op(line)
            if tag in {"replace", "insert"}:
                for line in b[j1:j2]:
                    yield "+   " + fmt_op(line)
        yield ""


import argparse

parser = argparse.ArgumentParser(description="BModel disassembler.")
parser.add_argument(
    "bmodels",
    type=str,
    nargs="+",
    help="The path of BModels. If one BModel is provided, the assemble code will be printed. Compare the Bmodels if two models provided.",
)
parser.add_argument(
    "--fmt",
    dest="format",
    choices=["mlir", "raw", "bits"],
    default="mlir",
    help="The format of format operations.",
)
parser.add_argument(
    "--N",
    dest="N",
    type=int,
    default=3,
    help="The number of context lines.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if len(args.bmodels) == 1:
        tpu_cmd = TPUCMD(args.bmodels[0])
        for idx, cmd in tpu_cmd.cmd:
            fmt_cmd = ["\n    " + str(x) for x in cmd.all]
            fmt_cmd = "".join(fmt_cmd) + "\n"
            fun_name = "graph" + "".join((str(x) for x in idx))
            print(f"func.func @{fun_name}() {{{fmt_cmd}}}")

    if len(args.bmodels) == 2:
        tpu_cmd_a = TPUCMD(args.bmodels[0])
        tpu_cmd_b = TPUCMD(args.bmodels[1])
        is_same = True
        for (idx, cmd_a), (_, cmd_b) in zip(tpu_cmd_a.cmd, tpu_cmd_b.cmd):
            fmt_cmd = [
                "\n" + x
                for x in unified_diff(
                    cmd_a.all,
                    cmd_b.all,
                    args.bmodels[0],
                    args.bmodels[1],
                    n=args.N,
                    format=args.format,
                )
            ]
            fun_name = "graph" + "".join((str(x) for x in idx))
            if fmt_cmd != []:
                is_same = False
                fmt_cmd = "".join(fmt_cmd[:-1]) + "\n"
                print(f"func.func @{fun_name}() {{{fmt_cmd}}}")
        if is_same:
            print(f""""{args.bmodels[0]}" and "{args.bmodels[1]}" are the same!""")
            exit(0)
        else:
            exit(1)
    parser.error("Too many BModels.")
