#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from debugger.context import Context

def decode_tiu_file(tiu_file, device):
    tiu = open(tiu_file, "rb").read()
    context = Context(device.upper())
    return context.decoder.decode_tiu_buf(tiu)

def decode_dma_file(dma_file, device):
    dma = open(dma_file, "rb").read()
    context = Context(device.upper())
    return context.decoder.decode_dma_buf(dma)

def merge_cmd(tiu_cmd, dma_cmd, device):
    tiu_cmd = list(tiu_cmd)
    dma_cmd = list(dma_cmd)
    context = Context(device.upper())
    return context.decoder.merge_instruction(tiu_cmd, dma_cmd)

def decode_bin(tiu_file, dma_file, device):
    tiu_cmd = decode_tiu_file(tiu_file, device)
    dma_cmd = decode_dma_file(dma_file, device)
    return merge_cmd(tiu_cmd, dma_cmd, device)

def __main():
    import argparse

    parser = argparse.ArgumentParser(description="BModel disassembler regression test")
    parser.add_argument(
        "--tiu_cmd_bin",
        type=str,
        help="The path of tiu cmd bin.",
    )
    parser.add_argument(
        "--dma_cmd_bin",
        type=str,
        help="The path of dma cmd bin.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device name.",
    )
    parser.add_argument(
        "--cmd_file",
        type=str,
        help="The output file.",
    )
    args = parser.parse_args()
    decoded_cmds = decode_bin(args.tiu_cmd_bin, args.dma_cmd_bin, args.device)

    cmd_file = open(args.cmd_file, "rb").read().decode()
    # hen the multi-core expression form is finally determined,
    # then open the regression
    if args.device != "bm1686":
        assert cmd_file == str(decoded_cmds)

if __name__ == "__main__":
    __main()
