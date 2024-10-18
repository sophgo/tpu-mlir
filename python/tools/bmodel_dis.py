#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# from debugger.context import Context
import debugger.disassembler as dis
from debugger.atomic_dialect import decode_cmdgroup
from typing import Tuple, Iterator
from debugger.target_common import use_backend


def decode_tiu_file(tiu_file, device: str):
    tiu = open(tiu_file, "rb").read()

    with use_backend(device.upper()) as context:
        return context.decoder.decode_tiu_cmds(tiu)


def decode_dma_file(dma_file, device):
    dma = open(dma_file, "rb").read()

    with use_backend(device.upper()) as context:
        return context.decoder.decode_dma_cmds(dma)


def BModel2MLIR(bmodel_file):
    from debugger.atomic_dialect import BModel2MLIR

    bmodel = dis.BModel(bmodel_file)
    return BModel2MLIR(bmodel)


def BModelCMDIter(bmodel: dis.BModel) -> Iterator[dis.SubNet]:
    for net in bmodel.net:
        for param in net.parameter:
            for subnet in param.sub_net:
                yield subnet


def BModel2Reg(bmodel_file):
    bmodel = dis.BModel(bmodel_file)
    for subnet in BModelCMDIter(bmodel):
        subnet_id = subnet.id
        for gid, cmds in enumerate(subnet.cmd_group):
            formated_id = f"core(0).subnet({subnet_id}).group({gid})"
            yield formated_id, decode_cmdgroup(bmodel.context, cmds, subnet_id, 0)
        for core_id, _cmds in enumerate(subnet.core_commands):
            for gid, cmds in enumerate(_cmds.gdma_tiu_commands):
                formated_id = f"core({core_id}).subnet({subnet_id}).group({gid})"
                yield formated_id, decode_cmdgroup(
                    bmodel.context, cmds, subnet_id, core_id
                )


def BModel2Bin(bmodel_file):
    import math

    class FName:
        core_id = 0
        subnet_id = 0
        gid = 0
        length = 0
        suffix = ""

        def __str__(self):
            return bmodel_file.split(".")[0] + f"{self.suffix}.{self.core_id}"

    fname = FName()
    bmodel = dis.BModel(bmodel_file)
    for subnet in BModelCMDIter(bmodel):
        fname.core_id = 0
        fname.subnet_id = subnet.id
        for fname.gid, cmds in enumerate(subnet.cmd_group):
            fname.length, fname.suffix = cmds.tiu_num, ".BD"
            with open(str(fname), "wb") as f:
                f.write(bytes(cmds.tiu_cmd))
            fname.length, fname.suffix = cmds.dma_num, ".GDMA"
            with open(str(fname), "wb") as f:
                f.write(bytes(cmds.dma_cmd))
        for fname.core_id, _cmds in enumerate(subnet.core_commands):
            for fname.gid, cmds in enumerate(_cmds.gdma_tiu_commands):
                fname.length, fname.suffix = cmds.tiu_num, ".BD"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds.tiu_cmd))
                fname.length, fname.suffix = cmds.dma_num, ".GDMA"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds.dma_cmd))
            for fname.gid, cmds in enumerate(_cmds.sdma_commands):
                # cmds contains system-end.
                fname.length, fname.suffix = math.ceil(len(cmds) / 96), ".SDMA"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))
            for fname.gid, cmds in enumerate(_cmds.hau_commands):
                fname.length, fname.suffix = math.ceil(len(cmds) / 80), ".HAU"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))
            for fname.gid, cmds in enumerate(_cmds.cdma_commands):
                fname.length, fname.suffix = math.ceil(len(cmds) / 120), ".HAU"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))


def unified_diff(a, b, fromfile="", tofile="", n=3, format="mlir"):
    r"""
    Compare the operations of two BModel; generate the delta as a unified diff.

    Unified diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.
    """
    import difflib

    fmt_op = {
        "raw": lambda op: str(op.attr),
        "mlir": lambda op: str(op),
        "bits": lambda op: "".join((str(x) for x in op.cmd)),
    }
    fmt = fmt_op[format]

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
                    yield "    " + fmt(line)
                continue
            if tag in {"replace", "delete"}:
                for line in a[i1:i2]:
                    yield "-   " + fmt(line)
            if tag in {"replace", "insert"}:
                for line in b[j1:j2]:
                    yield "+   " + fmt(line)
        yield ""


def __main():
    import argparse

    parser = argparse.ArgumentParser(description="BModel disassembler.")
    parser.add_argument(
        "bmodels",
        type=str,
        nargs="+",
        help="The path of BModels. If one BModel is provided, the assemble code will be printed. Compare the Bmodels if two models provided.",
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=["mlir", "reg", "bits", "bin", "reg-set", "version"],
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
    args = parser.parse_args()
    if len(args.bmodels) == 1:
        if args.format == "mlir":
            module = BModel2MLIR(args.bmodels[0])
            print(module, flush=True)
        elif args.format == "version":
            bmodel = dis.BModel(args.bmodels[0])
            print(bmodel.version)
        elif args.format == "reg" or args.format == "reg-set":
            import json

            visited = set()

            def not_visited(op):
                if args.format == "reg":
                    return True
                res = op not in visited
                visited.add(op)
                return res

            module = BModel2Reg(args.bmodels[0])
            outs = {
                _id: {
                    "tiu": [
                        {
                            "name": getattr(x, "op_name", x.name),
                            "cmd": dict(x.reg),
                            "extra": {
                                "cmd_id": x.cmd_id,
                                "cmd_type": x.cmd_type.name,
                                "core_id": x.core_id,
                                "subnet_id": x.subnet_id,
                            },
                        }
                        for x in ops.tiu
                        if not_visited(getattr(x, "op_name", x.name))
                    ],
                    "dma": [
                        {
                            "name": getattr(x, "op_name", x.name),
                            "cmd": dict(x.reg),
                            "extra": {
                                "cmd_id": x.cmd_id,
                                "cmd_type": x.cmd_type.name,
                                "core_id": x.core_id,
                                "subnet_id": x.subnet_id,
                            },
                        }
                        for x in ops.dma
                        if not_visited(getattr(x, "op_name", x.name))
                    ],
                }
                for _id, ops in module
            }
            print(json.dumps(outs, indent=2, ensure_ascii=False), flush=True)

        elif args.format == "bin":
            BModel2Bin(args.bmodels[0])
        else:
            raise NotImplementedError("Not supports bits mode.")
        exit(0)

    # FIX ME: the code below doe not compatible with the new Module.
    if len(args.bmodels) == 2:
        tpu_cmd_a = BModel2MLIR(args.bmodels[0])
        tpu_cmd_b = BModel2MLIR(args.bmodels[1])
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


if __name__ == "__main__":
    __main()
