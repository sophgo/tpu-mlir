#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from debugger.context import Context
import debugger.disassembler as dis


def decode_tiu_file(tiu_file, device):
    tiu = open(tiu_file, "rb").read()
    context = Context(device.upper())
    return context.decoder.decode_tiu_buf(tiu)


def decode_dma_file(dma_file, device):
    dma = open(dma_file, "rb").read()
    context = Context(device.upper())
    return context.decoder.decode_dma_buf(dma)


def BModel2MLIR(bmodel_file):
    bmodel = dis.BModel(bmodel_file)
    chip = bmodel.chip
    context = Context(chip)
    return context.BModel2MLIR(bmodel)


def BModelCMDIter(bmodel):
    for net in bmodel.net:
        for param in net.parameter:
            for subnet in param.sub_net:
                _id = subnet.id
                for gid, _net in enumerate(getattr(subnet, "cmd_group", [])):
                    yield (0, _id, gid), _net
                for core_id, cmds in enumerate(getattr(subnet, "core_commands", [])):
                    for gid, _net in enumerate(cmds.gdma_tiu_commands):
                        yield (core_id, _id, gid), _net


def BModel2Reg(bmodel_file):
    bmodel = dis.BModel(bmodel_file)
    chip = bmodel.chip
    context = Context(chip)
    decoder = context.decoder
    for identifier, net in BModelCMDIter(bmodel):
        core_id, _id, gid = identifier
        formated_id = f"core({core_id}).subnet({_id}).group({gid})"
        yield formated_id, decoder.decode_bmodel_cmd(net, _id, core_id)


def BModel2Bin(bmodel_file):
    bmodel = dis.BModel(bmodel_file)
    for identifier, net in BModelCMDIter(bmodel):
        core_id, _id, gid = identifier
        formated_id = f".core({core_id}).subnet({_id}).group({gid})."
        with open(bmodel_file + formated_id + "tiu.bin", "wb") as f:
            f.write(net.tiu_cmd)
        with open(bmodel_file + formated_id + "dma.bin", "wb") as f:
            f.write(net.dma_cmd)


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
        choices=["mlir", "reg", "bits", "bin"],
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
        elif args.format == "reg":
            import json

            module = BModel2Reg(args.bmodels[0])
            outs = {
                _id: {
                    "tiu": [dict(x.reg) for x in ops.tiu],
                    "dma": [dict(x.reg) for x in ops.dma],
                }
                for _id, ops in module
            }
            print(json.dumps(outs, indent=2), flush=True)

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
