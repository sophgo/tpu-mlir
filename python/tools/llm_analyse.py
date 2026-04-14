#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""
LLM MLIR analysis tool.

Discovers and analyzes all top-level MLIR files generated for an LLM model,
producing a single Excel workbook with:
  - Overview sheet: chip parameters, per-module summary, phase totals
  - Per-module sheets: detailed per-operator analysis (Roofline model)

Transformer blocks (block_0..N, block_cache_0..N) are grouped; the first
block is analyzed as representative and multiplied by the block count.

Directory structure expected:
    <model_dir>/<module_name>/<module_name>.mlir
    e.g. qwen3.5-0.8b_bf16_seq2048_bm1684x_1dev_static/block_0/block_0.mlir

Usage:
    python llm_analyse.py <model_dir> -t 32 -b 64
    python llm_analyse.py <model_dir> -t 32 -b 64 -d w4f16 -o result.xlsx
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Tuple, Optional
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlir_analyse import (
    parse_mlir_file,
    calc_flops,
    calc_data_volume,
    KEY_OPS,
    SKIP_OPS,
    fmt_num,
    fmt_bytes,
    fmt_time,
)

# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------
PREFILL_MODULES = {
    "vit", "embedding", "block", "block(LinearAttention)", "block(FullAttention)", "lm_head"
}
DECODE_MODULES = {
    "embedding_cache", "block_cache", "block_cache(LinearAttention)", "block_cache(FullAttention)",
    "lm_head"
}
MODULE_ORDER = [
    "vit",
    "embedding",
    "block",
    "lm_head",
    "embedding_cache",
    "block_cache",
]

# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


def discover_modules(model_dir: str,
                     num_layers: int,
                     model_type: str = "") -> List[Tuple[str, str, int]]:
    """Discover and group MLIR files under model_dir.

    Returns list of (module_name, mlir_path, count) in logical order.
    Transformer blocks are grouped: block -> (block_0.mlir, N).
    For qwen3_5_text, block_0/block_3 are kept separate (3:1 ratio).
    """
    block_files: Dict[int, str] = {}
    cache_files: Dict[int, str] = {}
    other_files: Dict[str, str] = {}

    for entry in sorted(os.listdir(model_dir)):
        sub_dir = os.path.join(model_dir, entry)
        if not os.path.isdir(sub_dir):
            continue
        mlir_path = os.path.join(sub_dir, f"{entry}.mlir")
        if not os.path.isfile(mlir_path):
            continue
        m_block = re.match(r"^block_(\d+)$", entry)
        m_cache = re.match(r"^block_cache_(\d+)$", entry)
        if m_block:
            block_files[int(m_block.group(1))] = mlir_path
        elif m_cache:
            cache_files[int(m_cache.group(1))] = mlir_path
        else:
            other_files[entry] = mlir_path

    is_qwen3_5 = model_type == "qwen3_5_text"
    modules = []
    for name in MODULE_ORDER:
        if name == "block":
            if not is_qwen3_5:
                modules.append(("block", block_files[0], num_layers))
            else:
                modules.append(("block(LinearAttention)", block_files[0], num_layers * 3 // 4))
                modules.append(("block(FullAttention)", block_files[3], num_layers // 4))
        elif name == "block_cache":
            if not is_qwen3_5:
                modules.append(("block_cache", cache_files[0], num_layers))
            else:
                modules.append(
                    ("block_cache(LinearAttention)", cache_files[0], num_layers * 3 // 4))
                modules.append(("block_cache(FullAttention)", cache_files[3], num_layers // 4))
        elif name in other_files:
            modules.append((name, other_files.pop(name), 1))

    for name in sorted(other_files):
        modules.append((name, other_files[name], 1))

    return modules


# ---------------------------------------------------------------------------
# Single-module analysis
# ---------------------------------------------------------------------------


def analyse_module(filepath: str, dtype_mode: str = "f16"):
    """Parse and analyse a single MLIR file.

    Returns (rows_data, totals) where totals has keys: flops, read, write, io.
    """
    ops, _, ssa_op_map = parse_mlir_file(filepath)
    compute_ops = [op for op in ops if op.op_type not in SKIP_OPS]

    rows_data = []
    total_flops = total_read = total_write = 0

    for op in compute_ops:
        opn = op.op_type.split(".")[-1]
        if opn == "MatMul" and ssa_op_map and len(op.operands) > 1:
            if ssa_op_map.get(op.operands[1], "") == "top.Weight":
                opn = f"MatMul ({dtype_mode})"
        flops = calc_flops(op)
        rb, wb = calc_data_volume(op, dtype_mode, ssa_op_map)
        inp_shapes = ", ".join(t.shape_str() if t else "none" for t in op.input_types)
        out_shapes = ", ".join(t.shape_str() if t else "none" for t in op.output_types)
        base_opn = op.op_type.split(".")[-1]
        rows_data.append(
            dict(
                opn=opn,
                base_opn=base_opn,
                loc=op.loc_name,
                inp_shapes=inp_shapes,
                out_shapes=out_shapes,
                flops=flops,
                rb=rb,
                wb=wb,
                total_io=rb + wb,
                is_key=base_opn in KEY_OPS,
            ))
        total_flops += flops
        total_read += rb
        total_write += wb

    totals = dict(flops=total_flops,
                  read=total_read,
                  write=total_write,
                  io=total_read + total_write)
    return rows_data, totals


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


def export_llm_excel(modules_data,
                     chip_tops,
                     bw_gbps,
                     out_dir,
                     llm_path,
                     dtype_mode="f16",
                     vector_tops=None,
                     uarch_rate=0.8,
                     bw_util=0.7,
                     parallelism=0.5,
                     model_config=None,
                     seq_length=0,
                     max_pixels="",
                     cmdline=""):
    """Create combined LLM analysis Excel workbook.

    modules_data: list of (name, count, rows_data, totals)
    model_config: dict of LLM architecture info from AutoConfig
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

    if vector_tops is None:
        vector_tops = chip_tops / 8.0

    # ---------- Styles ----------
    wb = Workbook()
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    hdr_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    key_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    sum_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
    edit_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    phase_fill = PatternFill(start_color="F4B084", end_color="F4B084", fill_type="solid")
    # Overview section styles
    param_hdr_fill = PatternFill(start_color="3B3838", end_color="3B3838", fill_type="solid")
    param_hdr_font = Font(bold=True, color="FFFFFF", size=11)
    ratio_hdr_fill = PatternFill(start_color="BF8F00", end_color="BF8F00", fill_type="solid")
    ratio_hdr_font = Font(bold=True, color="FFFFFF", size=11)
    mod_hdr_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    mod_hdr_font = Font(bold=True, color="FFFFFF", size=11)
    phase_font = Font(bold=True, color="833C0B")
    bold = Font(bold=True)
    thin = Border(*(Side(style="thin"), ) * 4)
    center = Alignment(horizontal="center", wrap_text=True)

    def _write_header(ws, headers):
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = center
            cell.border = thin

    def _set_col_widths(ws, widths):
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w

    # Formula references to Overview editable cells
    TOPS_REF = "Overview!$B$2"
    BW_REF = "Overview!$B$3"
    VECTOR_TOPS_REF = "Overview!$B$4"
    CU_REF = "Overview!$B$5"
    BU_REF = "Overview!$B$6"
    PAR_REF = "Overview!$B$8"
    CPU_CALL_REF = "Overview!$B$9"
    PREPROCESS_TIME_REF = "Overview!$B$10"
    FATTENTION_RATIO_REF = "Overview!$B$12"
    GATHER_RATIO_REF = "Overview!$B$13"
    PERMUTE_CONCAT_RATIO_REF = "Overview!$B$14"
    CHUNKGATEDDELTARULE_RATIO_REF = "Overview!$B$15"

    def _compute_formula(gops_cell, tops_ref=TOPS_REF):
        return f"=IF({tops_ref}=0,0,{gops_cell}/({tops_ref}*{CU_REF})*1000)"

    def _memory_formula(io_cell):
        return f"=IF({BW_REF}=0,0,{io_cell}/({BW_REF}*{BU_REF})*1000)"

    # ============ Overview sheet (default / first) ============
    ws0 = wb.active
    ws0.title = "Overview"

    # Row 1: header
    for c, h in enumerate(["Parameter", "Value", "Unit"], 1):
        cell = ws0.cell(row=1, column=c, value=h)
        cell.font = param_hdr_font
        cell.fill = param_hdr_fill
        cell.border = thin

    # Rows 2-10: editable parameters
    params = [
        ("Chip Compute Power", chip_tops, "TOPS", "#,##0.##"),
        ("Chip Bandwidth", bw_gbps, "GB/s", "#,##0.##"),
        ("Vector Compute Power", vector_tops, "TOPS", "#,##0.##"),
        ("uArch Rate", uarch_rate, "", "0%"),
        ("Bandwidth Utilization", bw_util, "", "0%"),
        ("Parallelism", parallelism, "", "0%"),
        ("Serialism", "=1-B7", "", "0%"),
        ("Cpu Call", 200, "us", "#,##0"),
        ("Preprocess Time", 0.1, "s", "#,##0.000"),
    ]
    for r, (label, val, unit, fmt) in enumerate(params, 2):
        ws0.cell(row=r, column=1, value=label)
        c = ws0.cell(row=r, column=2, value=val)
        c.fill = edit_fill
        c.font = Font(bold=True, color="006100")
        c.number_format = fmt
        ws0.cell(row=r, column=3, value=unit)

    # Row 11: Special Ratio header + rows 12-15
    for c, h in enumerate(["Special Ratio", "Value"], 1):
        cell = ws0.cell(row=11, column=c, value=h)
        cell.font = ratio_hdr_font
        cell.fill = ratio_hdr_fill
        cell.border = thin
    special_ratios = [
        ("FAttention Ratio", 3.0, "0%"),
        ("Gather Ratio", 5.0, "0%"),
        ("Permute/Concat Ratio", 2.0, "0%"),
        ("ChunkGatedDeltaRule Ratio", 3.0, "0%"),
    ]
    for r, (label, val, fmt) in enumerate(special_ratios, 12):
        ws0.cell(row=r, column=1, value=label)
        c = ws0.cell(row=r, column=2, value=val)
        c.fill = edit_fill
        c.font = Font(bold=True, color="006100")
        c.number_format = fmt

    # Row 17: Module summary table header
    sum_hdr = 17
    sum_headers = [
        "Module", "Count", "GOPs", "Total I/O (MB)", "Est. Time (s)", "Total Time (s)", "Phase"
    ]
    for c, h in enumerate(sum_headers, 1):
        cell = ws0.cell(row=sum_hdr, column=c, value=h)
        cell.font = mod_hdr_font
        cell.fill = mod_hdr_fill
        cell.alignment = center
        cell.border = thin

    # ============ Create module sheets & fill Overview rows ============
    op_headers = [
        "No.",
        "Op Type",
        "Input Shapes",
        "Output Shapes",
        "GOPs",
        "Read (MB)",
        "Write (MB)",
        "Total I/O (MB)",
        "Compute (us)",
        "Memory (us)",
        "Est. Time (us)",
        "Bottleneck",
        "Name",
    ]
    mod_start = sum_hdr + 1
    prefill_rows = []
    decode_rows = []
    cur_ov_row = mod_start
    has_vit = False

    for mi, (mod_name, count, rows_data, totals) in enumerate(modules_data):
        # --- Create module sheet ---
        ws = wb.create_sheet(mod_name)
        _write_header(ws, op_headers)

        for idx, d in enumerate(rows_data, 1):
            r = idx + 1
            for c, v in enumerate([
                    idx,
                    d["opn"],
                    d["inp_shapes"],
                    d["out_shapes"],
                    d["flops"] / 1e9,
                    d["rb"] / 1024.0 / 1024.0,
                    d["wb"] / 1024.0 / 1024.0,
                    d["total_io"] / 1024.0 / 1024.0,
            ], 1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.border = thin
                if d["is_key"]:
                    cell.fill = key_fill
                if c in (5, 6, 7, 8):
                    cell.number_format = "#,##0.000"

            tops_ref = TOPS_REF if d["is_key"] else VECTOR_TOPS_REF
            # I: Compute(us)
            cell_i = ws.cell(row=r, column=9)
            cell_i.value = _compute_formula(f"E{r}", tops_ref)
            cell_i.number_format = "#,##0.000"
            cell_i.border = thin
            # J: Memory(us)
            cell_j = ws.cell(row=r, column=10)
            cell_j.value = _memory_formula(f"H{r}")
            cell_j.number_format = "#,##0.000"
            cell_j.border = thin
            # K: Est.Time = MAX(I,J) + Serialism * MIN(I,J), with special OP ratio
            cell_k = ws.cell(row=r, column=11)
            base_time = f"MAX(I{r},J{r})+{PAR_REF}*MIN(I{r},J{r})"
            if d["base_opn"] == "FAttention":
                cell_k.value = f"=({base_time})*{FATTENTION_RATIO_REF}"
            elif d["base_opn"] == "Gather":
                cell_k.value = f"=({base_time})*{GATHER_RATIO_REF}"
            elif d["base_opn"] in ("Permute", "Concat"):
                cell_k.value = f"=({base_time})*{PERMUTE_CONCAT_RATIO_REF}"
            elif d["base_opn"] == "ChunkGatedDeltaRule":
                cell_k.value = f"=({base_time})*{CHUNKGATEDDELTARULE_RATIO_REF}"
            else:
                cell_k.value = f"={base_time}"
            cell_k.number_format = "#,##0.000"
            cell_k.border = thin
            # L: Bottleneck
            ws.cell(row=r, column=12, value=f'=IF(I{r}>=J{r},"Compute","Memory")').border = thin
            # N: Name
            ws.cell(row=r, column=13, value=d["loc"]).border = thin

            if d["is_key"]:
                for c in range(9, 14):
                    ws.cell(row=r, column=c).fill = key_fill

        # Summary row
        sr = len(rows_data) + 3
        for c in range(1, len(op_headers) + 1):
            cell = ws.cell(row=sr, column=c)
            cell.fill = sum_fill
            cell.font = bold
            cell.border = thin
        ws.cell(row=sr, column=1, value="TOTAL")
        fd, ld = 2, len(rows_data) + 1
        ws.cell(row=sr, column=5, value=f"=SUM(E{fd}:E{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=6, value=f"=SUM(F{fd}:F{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=7, value=f"=SUM(G{fd}:G{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=8, value=f"=SUM(H{fd}:H{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=11, value=f"=SUM(K{fd}:K{ld})").number_format = "#,##0.000"
        ws.freeze_panes = "A2"
        _set_col_widths(ws, [5, 18, 22, 16, 10, 10, 10, 12, 12, 12, 12, 11, 10, 20])

        # --- Fill Overview module row ---
        ov_r = cur_ov_row
        cur_ov_row += 1
        ws0.cell(row=ov_r, column=1, value=mod_name).border = thin
        # Count (editable for blocks)
        c_count = ws0.cell(row=ov_r, column=2, value=count)
        c_count.border = thin
        c_count.number_format = "0"
        if count > 1:
            c_count.fill = edit_fill
            c_count.font = Font(bold=True, color="006100")
        # C: GOPs (reference sheet total)
        c_gops = ws0.cell(row=ov_r, column=3)
        c_gops.value = f"='{mod_name}'!E{sr}"
        c_gops.number_format = "#,##0.000"
        c_gops.border = thin
        # D: Total I/O (MB)
        c_io = ws0.cell(row=ov_r, column=4)
        c_io.value = f"='{mod_name}'!H{sr}"
        c_io.number_format = "#,##0.000"
        c_io.border = thin
        # E: Est. Time (s) per instance (module sheet K is in us, /1e6 -> s)
        c_est = ws0.cell(row=ov_r, column=5)
        c_est.value = f"='{mod_name}'!K{sr}/1000000"
        c_est.number_format = "#,##0.000000"
        c_est.border = thin
        # F: Total Time (s) = Est.Time * Count + Count * CpuCall(us) / 1e6
        c_total = ws0.cell(row=ov_r, column=6)
        c_total.value = f"=E{ov_r}*B{ov_r}+B{ov_r}*{CPU_CALL_REF}/1000000"
        c_total.number_format = "#,##0.000000"
        c_total.border = thin
        # G: Phase
        if mod_name in PREFILL_MODULES and mod_name in DECODE_MODULES:
            phase = "Both"
        elif mod_name in PREFILL_MODULES:
            phase = "Prefill"
        elif mod_name in DECODE_MODULES:
            phase = "Decode"
        else:
            phase = ""
        ws0.cell(row=ov_r, column=7, value=phase).border = thin

        if mod_name in PREFILL_MODULES:
            prefill_rows.append(ov_r)
        if mod_name in DECODE_MODULES:
            decode_rows.append(ov_r)
        if mod_name == "vit":
            has_vit = True

    # ============ Phase summary rows ============
    phase_r = cur_ov_row + 1
    for pr, (label, rows) in enumerate([
        ("Prefill Total", prefill_rows),
        ("Decode Total", decode_rows),
    ]):
        r = phase_r + pr
        for c in range(1, len(sum_headers) + 1):
            cell = ws0.cell(row=r, column=c)
            cell.fill = phase_fill
            cell.font = phase_font
            cell.border = thin
        ws0.cell(row=r, column=1, value=label)
        if rows:
            total_time_formula = "+".join(f"F{x}" for x in rows)
            if label == "Prefill Total" and has_vit:
                total_time_formula += f"+{PREPROCESS_TIME_REF}"
            ws0.cell(
                row=r,
                column=3,
                value="=" + "+".join(f"C{x}*B{x}" for x in rows),
            ).number_format = "#,##0.000"
            ws0.cell(
                row=r,
                column=4,
                value="=" + "+".join(f"D{x}*B{x}" for x in rows),
            ).number_format = "#,##0.000"
            ws0.cell(
                row=r,
                column=6,
                value="=" + total_time_formula,
            ).number_format = "#,##0.000000"

    # ============ TTFT and Tokens/s ============
    ttft_r = phase_r + 3
    # Styles for result section
    result_fill = PatternFill(start_color="2F75B5", end_color="2F75B5", fill_type="solid")
    result_font = Font(bold=True, color="FFFFFF", size=12)
    util_label_font = Font(bold=True, color="333333", size=10)
    util_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    util_font = Font(bold=True, color="1F4E79", size=10)

    # -- TTFT row --
    c_ttft_label = ws0.cell(row=ttft_r, column=1, value="TTFT (s)")
    c_ttft_label.font = result_font
    c_ttft_label.fill = result_fill
    c_ttft_label.border = thin
    c_ttft = ws0.cell(row=ttft_r, column=2)
    c_ttft.value = f"=F{phase_r}"
    c_ttft.number_format = "#,##0.000000"
    c_ttft.font = result_font
    c_ttft.fill = result_fill
    c_ttft.border = thin
    # Prefill compute utilization
    c_pcl = ws0.cell(row=ttft_r, column=3, value="MFU Util")
    c_pcl.font = util_label_font
    c_pcl.fill = util_fill
    c_pcl.border = thin
    c_pcu = ws0.cell(row=ttft_r, column=4)
    c_pcu.value = f"=IF(F{phase_r}=0,0,C{phase_r}/{TOPS_REF}/1000/F{phase_r})"
    c_pcu.number_format = "0.00%"
    c_pcu.font = util_font
    c_pcu.fill = util_fill
    c_pcu.border = thin
    # Prefill BW utilization
    c_pbl = ws0.cell(row=ttft_r, column=5, value="BW Util")
    c_pbl.font = util_label_font
    c_pbl.fill = util_fill
    c_pbl.border = thin
    c_pbu = ws0.cell(row=ttft_r, column=6)
    c_pbu.value = f"=IF(F{phase_r}=0,0,D{phase_r}/{BW_REF}/1000/F{phase_r})"
    c_pbu.number_format = "0.00%"
    c_pbu.font = util_font
    c_pbu.fill = util_fill
    c_pbu.border = thin

    # -- Tokens/s row --
    c_tps_label = ws0.cell(row=ttft_r + 1, column=1, value="Tokens/s")
    c_tps_label.font = result_font
    c_tps_label.fill = result_fill
    c_tps_label.border = thin
    c_tps = ws0.cell(row=ttft_r + 1, column=2)
    c_tps.value = f"=IF(F{phase_r+1}=0,0,1/F{phase_r+1})"
    c_tps.number_format = "#,##0.00"
    c_tps.font = result_font
    c_tps.fill = result_fill
    c_tps.border = thin
    # Decode compute utilization
    c_dcl = ws0.cell(row=ttft_r + 1, column=3, value="MFU Util")
    c_dcl.font = util_label_font
    c_dcl.fill = util_fill
    c_dcl.border = thin
    c_dcu = ws0.cell(row=ttft_r + 1, column=4)
    c_dcu.value = f"=IF(F{phase_r+1}=0,0,C{phase_r+1}/{TOPS_REF}/1000/F{phase_r+1})"
    c_dcu.number_format = "0.00%"
    c_dcu.font = util_font
    c_dcu.fill = util_fill
    c_dcu.border = thin
    # Decode BW utilization
    c_dbl = ws0.cell(row=ttft_r + 1, column=5, value="BW Util")
    c_dbl.font = util_label_font
    c_dbl.fill = util_fill
    c_dbl.border = thin
    c_dbu = ws0.cell(row=ttft_r + 1, column=6)
    c_dbu.value = f"=IF(F{phase_r+1}=0,0,D{phase_r+1}/{BW_REF}/1000/F{phase_r+1})"
    c_dbu.number_format = "0.00%"
    c_dbu.font = util_font
    c_dbu.fill = util_fill
    c_dbu.border = thin

    # ============ Model Architecture section ============
    arch_r = ttft_r + 3
    if model_config:
        ws0.cell(row=arch_r, column=1, value="Model Architecture").font = Font(bold=True, size=12)
        arch_fields = [
            ("Model", llm_path),
            ("Architecture", model_config.get("architectures", "")),
            ("Num Hidden Layers", model_config.get("num_hidden_layers", "")),
            ("Hidden Size", model_config.get("hidden_size", "")),
            ("Num Attention Heads", model_config.get("num_attention_heads", "")),
            ("Num Key Value Heads", model_config.get("num_key_value_heads", "")),
            ("Intermediate Size", model_config.get("intermediate_size", "")),
            ("Vocab Size", model_config.get("vocab_size", "")),
            ("Head Dim", model_config.get("head_dim", "")),
            ("Command", cmdline if cmdline else ""),
            ("Dtype Mode", dtype_mode),
            ("Seq Length", seq_length if seq_length else ""),
            ("Max Pixels", max_pixels if max_pixels else ""),
        ]
        # Filter out empty values
        arch_fields = [(k, v) for k, v in arch_fields if v != ""]
        for r_off, (k, v) in enumerate(arch_fields, 1):
            ws0.cell(row=arch_r + r_off, column=1, value=k).border = thin
            c_val = ws0.cell(row=arch_r + r_off, column=2, value=v)
            c_val.border = thin
            if isinstance(v, (int, float)):
                c_val.number_format = "#,##0"
        info_r = arch_r + len(arch_fields) + 3
    else:
        info_r = arch_r + 2
        ws0.cell(row=info_r - 1, column=1, value="Model").border = thin
        ws0.cell(row=info_r - 1, column=2, value=llm_path).border = thin

    # ============ Info section ============
    info_items = [
        ("Note", "Modify B2-B10 and B12-B15 (green cells) to update all performance estimates."),
        ("Note", "Block counts (green) in the module table are also editable."),
        ("Note", "Est.Time = max(Compute, Memory) + Serialism * min(Compute, Memory)"),
        ("Note", "block/block_cache are analyzed from the first block as representative."),
    ]
    for r_off, (k, v) in enumerate(info_items):
        ws0.cell(row=info_r + r_off, column=1, value=k)
        ws0.cell(row=info_r + r_off, column=2, value=v)

    _set_col_widths(ws0, [26, 28, 14, 16, 16, 16, 10])
    file = os.path.join(out_dir, f"{os.path.basename(out_dir)}.xlsx")
    wb.save(file)
    print(f"\nExcel saved: {file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # yapf: disable
    parser = argparse.ArgumentParser(
        description="Analyze LLM MLIR modules: per-operator FLOPs, data volume, "
                    "estimated runtime (Roofline model)")
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument("-t", "--tops", type=float, required=True,
                        help="Chip compute power in TOPS")
    parser.add_argument("-b", "--bandwidth", type=float, required=True,
                        help="Chip memory bandwidth in GB/s")
    parser.add_argument("-q", "--quantize", default="f16",
                        choices=["f16", "w8f16", "w4f16","bf16", "w8bf16", "w4bf16"],
                        help="Quantization mode (default: f16)")
    parser.add_argument("-v", "--vector_tops", type=float, default=None,
                        help="Vector compute power in TOPS (default: tops/8)")
    parser.add_argument("-r", "--uarch_rate", type=float, default=0.8,
                        help="uArch Rate ratio (default: 0.8)")
    parser.add_argument("-u", "--bw_util", type=float, default=0.8,
                        help="Bandwidth utilization ratio (default: 0.8)")
    parser.add_argument("-p", "--parallelism", type=float, default=0.5,
                        help="Parallelism ratio for Est.Time (default: 0.5)")
    parser.add_argument('--max_input_length', type=int, default=0,
                        help='max input length for prefill, default 0 means the same as seq_length')
    parser.add_argument("-o", "--out_dir", required=True,
                        help="Output directory path (default: <out_dir>/<out_dir>_analysis.xlsx)")
    args = parser.parse_args()
    # yapf: enable
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, "text_config"):
        llm_config = config.text_config
    else:
        llm_config = config
    # ------------------------------------------------------------------
    # TODO: Add llm_converter.py call here to generate MLIRs if needed
    # e.g. convert_model(args.model_path, args.model_dir, ...)
    # ------------------------------------------------------------------
    if config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
        max_pixels = "672,896"
    else:
        max_pixels = "768,768"
    cmds = [
        "llm_convert.py", f"-m {args.model_path}", f"-s {args.seq_length}", f"-q {args.quantize}",
        "-c bm1684x", f"--out_dir {args.out_dir}", "--only_mlir", f"--max_pixels {max_pixels}"
    ]
    if args.max_input_length > 0:
        cmds.append(f"--max_input_length {args.max_input_length}")
    print("\nRunning LLM conversion to generate MLIR files...")
    cmd = " ".join(cmds)
    print(f"Command: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during LLM conversion: {e}", file=sys.stderr)
        sys.exit(1)

    model_type = getattr(llm_config, "model_type", "")
    # Extract model config info
    model_config = {}
    for key in [
            "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "vocab_size", "head_dim"
    ]:
        val = getattr(llm_config, key, None)
        if val is not None:
            model_config[key] = val
    archs = getattr(config, "architectures", None)
    if archs:
        model_config["architectures"] = archs[0] if len(archs) == 1 else ", ".join(archs)
    # Use num_hidden_layers to correct block counts if available
    num_layers = model_config.get("num_hidden_layers")
    # Step 1: Discover modules
    mlir_dir = os.path.join(args.out_dir, "tmp_mlir_analyse")
    print(f"Scanning: {mlir_dir}")
    modules = discover_modules(mlir_dir, llm_config.num_hidden_layers, model_type)
    if not modules:
        print("No MLIR files found.", file=sys.stderr)
        sys.exit(1)
    # Correct block counts from config if available
    if num_layers:
        if model_type == "qwen3_5_text":
            dense_count = round(num_layers * 3 / 4)
            sparse_count = num_layers - dense_count
            modules = [(n, p, dense_count if n in ("block_0", "block_cache_0") else
                        sparse_count if n in ("block_3", "block_cache_3") else c)
                       for n, p, c in modules]
        else:
            modules = [(n, p, num_layers if n in ("block", "block_cache") and c > 1 else c)
                       for n, p, c in modules]
    print(f"Found {len(modules)} module(s): "
          f"{', '.join(f'{n}(x{c})' if c > 1 else n for n, _, c in modules)}")

    # Step 2: Analyse each module
    modules_data = []
    for name, path, count in modules:
        print(f"  Analysing: {name} ({os.path.basename(path)})")
        dtype = args.quantize if name.startswith("block") else "f16"
        rows_data, totals = analyse_module(path, dtype)
        modules_data.append((name, count, rows_data, totals))

    # Step 3: Export Excel
    cmdline = "python " + " ".join(sys.argv)
    export_llm_excel(modules_data, args.tops, args.bandwidth, args.out_dir, args.model_path,
                     args.quantize, args.vector_tops, args.uarch_rate, args.bw_util,
                     args.parallelism, model_config, args.seq_length, max_pixels, cmdline)


if __name__ == "__main__":
    main()
