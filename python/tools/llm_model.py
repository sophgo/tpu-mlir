#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""
LLM BModel memory usage analysis tool.

Analyzes memory usage of an LLM bmodel file by calling ``model_tool --print``
and parsing the output. Memory is categorized into four parts:

  1. Neuron   — neuron_size
  2. Coeff    — binary_coeff sizes (deduplicated by start)
  3. KV Cache — IO tensor sizes of block_cache_ networks
  4. Instruct — binary_ir + cmd_group binary_bdc/binary_gdma
                (deduplicated by start, excluding sub_net)

Usage:
    python llm_model.py model.bmodel
    python llm_model.py --text model_info.txt   # parse saved text output
"""

import sys
import os
import json
import subprocess
import argparse

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _js_to_json(text):
    """Convert JavaScript-like object notation (unquoted keys) to valid JSON."""
    result = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            if in_string:
                # close string unless escaped
                if i == 0 or text[i - 1] != '\\':
                    in_string = False
            else:
                in_string = True
            result.append(ch)
            i += 1
        elif not in_string and ch == '0' and i + 1 < n and text[i + 1] in 'xX':
            # hex literal: 0x... → convert to decimal for JSON
            j = i + 2
            while j < n and (text[j] in '0123456789abcdefABCDEF'):
                j += 1
            result.append(str(int(text[i:j], 16)))
            i = j
        elif not in_string and (ch.isalpha() or ch == '_'):
            # collect word
            j = i
            while j < n and (text[j].isalnum() or text[j] == '_'):
                j += 1
            word = text[i:j]
            # look ahead for ':'  → this word is an object key
            k = j
            while k < n and text[k] == ' ':
                k += 1
            if k < n and text[k] == ':':
                result.append('"')
                result.append(word)
                result.append('"')
            else:
                result.append(word)
            i = j
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def parse_model_info(text):
    """Parse model_tool --print output into a Python dict."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_js_to_json(text))


def get_model_info(bmodel_path):
    """Run ``model_tool --print <bmodel>`` and return parsed dict."""
    cmd = ["model_tool", "--print", bmodel_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error running model_tool: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return parse_model_info(proc.stdout)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _align4k(size):
    """Round *size* up to the nearest multiple of 4096."""
    return (size + 4095) & ~4095


def _fmt(size_bytes):
    """Return human-readable size string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024 ** 3:.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


# ---------------------------------------------------------------------------
# Data type mapping
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    0: "FP32",
    1: "FP16",
    2: "INT8",
    3: "UINT8",
    4: "INT16",
    5: "UINT16",
    6: "INT32",
    7: "UINT32",
    8: "BFP16",
    9: "INT4",
    10: "UINT4",
    11: "FP20",
    12: "F8E5M2",
    13: "F8E4M3",
    -1: "UNKNOWN",
}


def _dtype(code):
    return DTYPE_MAP.get(code, f"DTYPE_{code}")


def _shape_str(tensor):
    dims_list = tensor.get("shape", [])
    if dims_list:
        dims = dims_list[0].get("dim", [])
        return "x".join(str(d) for d in dims)
    return ""


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------


def analyze_memory(data):
    """Analyze and print memory usage breakdown of the bmodel."""
    nets = data.get("net", [])

    # 1. Neuron space: neuron_size
    neuron_size = _align4k(data.get("neuron_size", 0))

    # 2. Coeff space: binary_coeff.size per net, deduplicated by start
    coeff_seen = {}  # start → size
    for net in nets:
        for param in net.get("parameter", []):
            coeff_mem = param.get("coeff_mem")
            if not coeff_mem:
                continue
            bc = coeff_mem.get("binary_coeff", {})
            start = bc.get("start", 0)
            size = bc.get("size", 0)
            if size <= 0:
                continue
            coeff_seen[start] = _align4k(size)
    coeff_size = sum(coeff_seen.values())

    # 3. KV space: sum of each input/output tensor size for block_cache_ nets
    kv_size = 0
    for net in nets:
        if not net["name"].startswith("block_cache_"):
            continue
        net_kv = 0
        for param in net.get("parameter", []):
            for t in param.get("input_tensor", []):
                net_kv += _align4k(t.get("size", 0))
            for t in param.get("output_tensor", []):
                net_kv += _align4k(t.get("size", 0))
        kv_size += net_kv

    # 4. Instruction space: binary_ir.size (per net) + cmd_group binary_bdc/binary_gdma
    #    Deduplicate bdc/gdma by start. Do NOT count sub_net.
    ir_total = 0
    bdc_seen = {}  # start → size
    gdma_seen = {}  # start → size
    for net in nets:
        for param in net.get("parameter", []):
            bir = param.get("binary_ir", {})
            ir_total += _align4k(bir.get("size", 0))
            for cg in param.get("cmd_group", []):
                bdc = cg.get("binary_bdc", {})
                gdma = cg.get("binary_gdma", {})
                bs, bsz = bdc.get("start", 0), bdc.get("size", 0)
                gs, gsz = gdma.get("start", 0), gdma.get("size", 0)
                if bsz > 0:
                    bdc_seen[bs] = _align4k(bsz)
                if gsz > 0:
                    gdma_seen[gs] = _align4k(gsz)
    inst_size = ir_total + sum(bdc_seen.values()) + sum(gdma_seen.values())

    # 5. Other space:
    #    a) embedding output tensor size
    #    b) io_size of all non-block_cache nets
    embed_out_size = 0
    non_bc_io_size = 0
    for net in nets:
        name = net.get("name", "")
        is_bc = name.startswith("block_cache_")
        for param in net.get("parameter", []):
            if not is_bc:
                non_bc_io_size += _align4k(param.get("io_size", 0))
            if name == "embedding":
                for t in param.get("output_tensor", []):
                    embed_out_size += _align4k(t.get("size", 0))
    # c) deepstack space: if vit has >1 output, non_bc_io_size * (vit_output_num - 1)
    vit_output_num = 0
    for net in nets:
        if net.get("name", "") == "vit":
            for param in net.get("parameter", []):
                vit_output_num += len(param.get("output_tensor", []))
            break
    deepstack_size = embed_out_size * (vit_output_num - 1) if vit_output_num > 1 else 0
    other_size = embed_out_size + non_bc_io_size + deepstack_size

    total = neuron_size + coeff_size + kv_size + inst_size + other_size

    # ----- report -----
    W = 60
    print("=" * W)
    print("  BModel Basic Info")
    print("=" * W)
    print(f"  Version: {data.get('version', 'N/A')}")
    print(f"  Time:    {data.get('time', 'N/A').strip()}")
    print(f"  Chip:    {data.get('chip', 'N/A')}")
    print()
    print("=" * W)
    print("  BModel Memory Usage Analysis")
    print("=" * W)
    print(f"  Neuron:    {_fmt(neuron_size):>12}  ({neuron_size:,} B)")
    print(f"  Coeff:     {_fmt(coeff_size):>12}  ({coeff_size:,} B)")
    print(f"  KV Cache:  {_fmt(kv_size):>12}  ({kv_size:,} B)")
    print(f"  Instruct:  {_fmt(inst_size):>12}  ({inst_size:,} B)")
    print(f"  Other:     {_fmt(other_size):>12}  ({other_size:,} B)")
    print(f"    - Embedding Output:  {_fmt(embed_out_size):>12}  ({embed_out_size:,} B)")
    print(f"    - IO (non-cache):    {_fmt(non_bc_io_size):>12}  ({non_bc_io_size:,} B)")
    if deepstack_size > 0:
        print(
            f"    - Deepstack (x{vit_output_num - 1}):   {_fmt(deepstack_size):>12}  ({deepstack_size:,} B)"
        )
    print(f"  {'─' * (W - 4)}")
    print(f"  Total:               {_fmt(total):>12}  ({total:,} B)")
    print("=" * W)

    # ----- components -----
    print()
    print("=" * W)
    print("  BModel Components")
    print("=" * W)

    def _print_tensors(net):
        param = net.get("parameter", [{}])[0]
        for t in param.get("input_tensor", []):
            print(
                f"        input:  {t['name']:30s}  [{_shape_str(t)}]  {_dtype(t.get('data_type', -1))}"
            )
        for t in param.get("output_tensor", []):
            print(
                f"        output: {t['name']:30s}  [{_shape_str(t)}]  {_dtype(t.get('data_type', -1))}"
            )

    # Group consecutive block_N / block_cache_N into ranges
    i = 0
    while i < len(nets):
        name = nets[i]["name"]
        # detect block_cache_N or block_N pattern
        for prefix in ("block_cache_", "block_"):
            if name.startswith(prefix):
                j = i + 1
                while j < len(nets) and nets[j]["name"].startswith(prefix):
                    j += 1
                idx_end = int(nets[j - 1]["name"][len(prefix):])
                if j - i > 1:
                    print(f"    {prefix}0 ~ {prefix}{idx_end}  ({j - i} nets)")
                else:
                    print(f"    {name}")
                _print_tensors(nets[i])
                i = j
                break
        else:
            print(f"    {name}")
            _print_tensors(nets[i])
            i += 1
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Analyze LLM BModel memory usage")
    ap.add_argument("bmodel", help="Path to bmodel file (or text file with --text)")
    ap.add_argument("--text",
                    action="store_true",
                    help="Treat input as saved model_tool --print output (text file)")
    args = ap.parse_args()

    if not os.path.exists(args.bmodel):
        print(f"Error: file not found: {args.bmodel}", file=sys.stderr)
        sys.exit(1)

    if args.text:
        with open(args.bmodel, "r") as f:
            data = parse_model_info(f.read())
    else:
        data = get_model_info(args.bmodel)

    analyze_memory(data)


if __name__ == "__main__":
    main()
