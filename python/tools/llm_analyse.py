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
        from openpyxl.formatting.rule import CellIsRule
        from openpyxl.utils import get_column_letter
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.formatting.rule import CellIsRule
        from openpyxl.utils import get_column_letter

    if vector_tops is None:
        vector_tops = chip_tops / 8.0

    # ---------- Color palette ----------
    C_TITLE = "1F4E79"  # deep blue - main title
    C_SECTION = "2E75B6"  # medium blue - section banners
    C_COLHDR = "4472C4"  # column headers
    C_RESULT = "1F4E79"  # result highlight
    C_EDIT = "E2EFDA"  # editable (soft green)
    C_EDIT_TXT = "375623"
    C_KEY = "FFF2CC"  # key operator (soft yellow)
    C_SUM = "DDEBF7"  # summary row (soft blue)
    C_PHASE = "FCE4D6"  # phase total (soft peach)
    C_PHASE_TXT = "833C0B"
    C_ZEBRA = "F7F9FC"  # alternating row
    C_UTIL = "D9E7F5"  # utilization value background

    # ---------- Styles ----------
    wb = Workbook()
    title_font = Font(bold=True, color="FFFFFF", size=14)
    title_fill = PatternFill(start_color=C_TITLE, end_color=C_TITLE, fill_type="solid")
    section_font = Font(bold=True, color="FFFFFF", size=11)
    section_fill = PatternFill(start_color=C_SECTION, end_color=C_SECTION, fill_type="solid")
    col_hdr_font = Font(bold=True, color="FFFFFF", size=10)
    col_hdr_fill = PatternFill(start_color=C_COLHDR, end_color=C_COLHDR, fill_type="solid")
    key_fill = PatternFill(start_color=C_KEY, end_color=C_KEY, fill_type="solid")
    zebra_fill = PatternFill(start_color=C_ZEBRA, end_color=C_ZEBRA, fill_type="solid")
    sum_fill = PatternFill(start_color=C_SUM, end_color=C_SUM, fill_type="solid")
    edit_fill = PatternFill(start_color=C_EDIT, end_color=C_EDIT, fill_type="solid")
    edit_font = Font(bold=True, color=C_EDIT_TXT)
    phase_fill = PatternFill(start_color=C_PHASE, end_color=C_PHASE, fill_type="solid")
    phase_font = Font(bold=True, color=C_PHASE_TXT)
    result_fill = PatternFill(start_color=C_RESULT, end_color=C_RESULT, fill_type="solid")
    result_font = Font(bold=True, color="FFFFFF", size=12)
    util_fill = PatternFill(start_color=C_UTIL, end_color=C_UTIL, fill_type="solid")
    util_font = Font(bold=True, color=C_TITLE, size=10)
    bold = Font(bold=True)
    thin = Border(*(Side(style="thin", color="BFBFBF"), ) * 4)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)

    def _set_col_widths(ws, widths):
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w

    def _banner(ws, row, text, span, fill=section_fill, font=section_font, height=22):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
        cell = ws.cell(row=row, column=1, value=text)
        cell.font = font
        cell.fill = fill
        cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[row].height = height

    # ============ Overview sheet ============
    ws0 = wb.active
    ws0.title = "Overview"
    TOTAL_COLS = 7

    # Row 1: Title banner
    _banner(ws0,
            1,
            "  LLM Performance Analysis",
            TOTAL_COLS,
            fill=title_fill,
            font=title_font,
            height=30)

    # --- Performance Summary (rows 3-5) ---
    _banner(ws0, 3, "Performance Summary", TOTAL_COLS)

    # We fill rows 4-5 with TTFT/Tokens/s; values reference phase totals below (computed later).
    # Placeholder cells styled now, formulas injected after we know phase_r.
    for r in (4, 5):
        for c in range(1, TOTAL_COLS + 1):
            ws0.cell(row=r, column=c).border = thin
        ws0.row_dimensions[r].height = 22
    ws0.cell(row=4, column=1, value="TTFT (s)").font = result_font
    ws0.cell(row=4, column=1).fill = result_fill
    ws0.cell(row=4, column=1).alignment = center
    ws0.cell(row=5, column=1, value="Tokens/s").font = result_font
    ws0.cell(row=5, column=1).fill = result_fill
    ws0.cell(row=5, column=1).alignment = center
    for r in (4, 5):
        vcell = ws0.cell(row=r, column=2)
        vcell.font = result_font
        vcell.fill = result_fill
        vcell.alignment = center
        # Utilization labels
        for col, label in ((3, "MFU Util"), (5, "BW Util")):
            lc = ws0.cell(row=r, column=col, value=label)
            lc.font = util_font
            lc.fill = util_fill
            lc.alignment = center
        for col in (4, 6):
            vc = ws0.cell(row=r, column=col)
            vc.font = util_font
            vc.fill = util_fill
            vc.alignment = center
            vc.number_format = "0.00%"

    # --- Hardware & Utilization (rows 7 onward) ---
    _banner(ws0, 7, "Hardware & Utilization", TOTAL_COLS)
    for c, h in enumerate(["Parameter", "Value", "Unit"], 1):
        cell = ws0.cell(row=8, column=c, value=h)
        cell.font = col_hdr_font
        cell.fill = col_hdr_fill
        cell.alignment = center
        cell.border = thin

    # params start at row 9
    params = [
        ("Chip Compute Power", chip_tops, "TOPS", "#,##0.##"),
        ("Vector Compute Power", vector_tops, "TOPS", "#,##0.##"),
        ("Chip Bandwidth", bw_gbps, "GB/s", "#,##0.##"),
        ("uArch Rate", uarch_rate, "", "0%"),
        ("Bandwidth Utilization", bw_util, "", "0%"),
        ("Parallelism", parallelism, "", "0%"),
        ("Serialism", "=1-B14", "", "0%"),
        ("CPU Call", 100, "us", "#,##0"),
        ("Preprocess Time", 0.1, "s", "#,##0.000"),
    ]
    PARAM_START = 9
    for i, (label, val, unit, fmt) in enumerate(params):
        r = PARAM_START + i
        ws0.cell(row=r, column=1, value=label).border = thin
        c = ws0.cell(row=r, column=2, value=val)
        c.border = thin
        c.number_format = fmt
        c.alignment = center
        # All except computed Serialism are editable
        if label != "Serialism":
            c.fill = edit_fill
            c.font = edit_font
        else:
            c.font = bold
        uc = ws0.cell(row=r, column=3, value=unit)
        uc.border = thin
        uc.alignment = center

    # Formula references (must match absolute positions above)
    TOPS_REF = "Overview!$B$9"
    VECTOR_TOPS_REF = "Overview!$B$10"
    BW_REF = "Overview!$B$11"
    CU_REF = "Overview!$B$12"
    BU_REF = "Overview!$B$13"
    PAR_REF = "Overview!$B$14"
    CPU_CALL_REF = "Overview!$B$16"
    PREPROCESS_TIME_REF = "Overview!$B$17"

    # --- Special Ratios (rows 19+) ---
    RATIO_BANNER = PARAM_START + len(params) + 1  # 19
    _banner(ws0, RATIO_BANNER, "Special Op Ratios", TOTAL_COLS)
    for c, h in enumerate(["Operation", "Ratio"], 1):
        cell = ws0.cell(row=RATIO_BANNER + 1, column=c, value=h)
        cell.font = col_hdr_font
        cell.fill = col_hdr_fill
        cell.alignment = center
        cell.border = thin

    special_ratios = [
        ("FAttention (Prefill)", 3.0),
        ("FAttention (Decode)", 2.0),
        ("Gather", 5.0),
        ("Permute/Concat", 2.0),
        ("ChunkGatedDeltaRule", 3.0),
        ("RecurrentGatedDeltaRule", 2.0),
    ]
    RATIO_START = RATIO_BANNER + 2  # 21
    for i, (label, val) in enumerate(special_ratios):
        r = RATIO_START + i
        ws0.cell(row=r, column=1, value=label).border = thin
        c = ws0.cell(row=r, column=2, value=val)
        c.fill = edit_fill
        c.font = edit_font
        c.number_format = "0%"
        c.border = thin
        c.alignment = center
    FATTENTION_RATIO_REF = f"Overview!$B${RATIO_START}"
    FATTENTION_DECODE_RATIO_REF = f"Overview!$B${RATIO_START+1}"
    GATHER_RATIO_REF = f"Overview!$B${RATIO_START+2}"
    PERMUTE_CONCAT_RATIO_REF = f"Overview!$B${RATIO_START+3}"
    CHUNKGATEDDELTARULE_RATIO_REF = f"Overview!$B${RATIO_START+4}"
    RECURRENTGATEDDELTARULE_RATIO_REF = f"Overview!$B${RATIO_START+5}"

    # --- Module Breakdown ---
    MOD_BANNER = RATIO_START + len(special_ratios) + 1  # 28
    _banner(ws0, MOD_BANNER, "Module Breakdown", TOTAL_COLS)
    sum_hdr = MOD_BANNER + 1
    sum_headers = [
        "Module", "Count", "GOPs", "I/O (MB)", "Est. Time (s)", "Total Time (s)", "Phase"
    ]
    for c, h in enumerate(sum_headers, 1):
        cell = ws0.cell(row=sum_hdr, column=c, value=h)
        cell.font = col_hdr_font
        cell.fill = col_hdr_fill
        cell.alignment = center
        cell.border = thin

    def _compute_formula(gops_cell, tops_ref=TOPS_REF):
        return f"=IF({tops_ref}=0,0,{gops_cell}/({tops_ref}*{CU_REF})*1000)"

    def _memory_formula(io_cell):
        return f"=IF({BW_REF}=0,0,{io_cell}/({BW_REF}*{BU_REF})*1000)"

    # Per-module sheet columns (no "No." column - Excel row numbers suffice)
    # 1:Op Type 2:Input Shapes 3:Output Shapes 4:GOPs 5:Read 6:Write 7:I/O
    # 8:Compute(us) 9:Memory(us) 10:Est.Time 11:Bottleneck 12:Name
    op_headers = [
        "Op Type",
        "Input Shapes",
        "Output Shapes",
        "GOPs",
        "Read (MB)",
        "Write (MB)",
        "I/O (MB)",
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
        for c, h in enumerate(op_headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = col_hdr_font
            cell.fill = col_hdr_fill
            cell.alignment = center
            cell.border = thin
        ws.row_dimensions[1].height = 28
        is_decode_mod = mod_name in DECODE_MODULES and mod_name not in PREFILL_MODULES

        for idx, d in enumerate(rows_data, 1):
            r = idx + 1
            use_zebra = (idx % 2 == 0) and not d["is_key"]
            row_fill = key_fill if d["is_key"] else (zebra_fill if use_zebra else None)
            values = [
                d["opn"],
                d["inp_shapes"],
                d["out_shapes"],
                d["flops"] / 1e9,
                d["rb"] / 1024.0 / 1024.0,
                d["wb"] / 1024.0 / 1024.0,
                d["total_io"] / 1024.0 / 1024.0,
            ]
            for c, v in enumerate(values, 1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.border = thin
                if row_fill is not None:
                    cell.fill = row_fill
                if c == 4:
                    cell.number_format = "#,##0.000000"
                elif c in (5, 6, 7):
                    cell.number_format = "#,##0.000"

            tops_ref = TOPS_REF if d["is_key"] else VECTOR_TOPS_REF
            # H: Compute(us) - from GOPs column D
            cell_h = ws.cell(row=r, column=8)
            cell_h.value = _compute_formula(f"D{r}", tops_ref)
            cell_h.number_format = "#,##0.000"
            cell_h.border = thin
            # I: Memory(us) - from I/O column G
            cell_i = ws.cell(row=r, column=9)
            cell_i.value = _memory_formula(f"G{r}")
            cell_i.number_format = "#,##0.000"
            cell_i.border = thin
            # J: Est.Time = MAX(H,I)+PAR*MIN(H,I), with special OP ratio
            cell_j = ws.cell(row=r, column=10)
            base_time = f"MAX(H{r},I{r})+{PAR_REF}*MIN(H{r},I{r})"
            if d["base_opn"] == "FAttention":
                fa_ref = FATTENTION_DECODE_RATIO_REF if is_decode_mod else FATTENTION_RATIO_REF
                cell_j.value = f"=({base_time})*{fa_ref}"
            elif d["base_opn"] == "Gather":
                cell_j.value = f"=({base_time})*{GATHER_RATIO_REF}"
            elif d["base_opn"] in ("Permute", "Concat"):
                cell_j.value = f"=({base_time})*{PERMUTE_CONCAT_RATIO_REF}"
            elif d["base_opn"] == "ChunkGatedDeltaRule":
                cell_j.value = f"=({base_time})*{CHUNKGATEDDELTARULE_RATIO_REF}"
            elif d["base_opn"] == "RecurrentGatedDeltaRule":
                cell_j.value = f"=({base_time})*{RECURRENTGATEDDELTARULE_RATIO_REF}"
            else:
                cell_j.value = f"={base_time}"
            cell_j.number_format = "#,##0.000"
            cell_j.border = thin
            # K: Bottleneck
            cell_k = ws.cell(row=r, column=11, value=f'=IF(H{r}>=I{r},"Compute","Memory")')
            cell_k.border = thin
            cell_k.alignment = center
            # L: Name
            ws.cell(row=r, column=12, value=d["loc"]).border = thin

            if row_fill is not None:
                for c in range(8, 13):
                    ws.cell(row=r, column=c).fill = row_fill

        # Conditional formatting on Bottleneck column (K)
        nrows = len(rows_data)
        if nrows > 0:
            rng = f"K2:K{nrows + 1}"
            compute_rule = CellIsRule(operator="equal",
                                      formula=['"Compute"'],
                                      font=Font(bold=True, color="1F4E79"),
                                      fill=PatternFill(start_color="BDD7EE",
                                                       end_color="BDD7EE",
                                                       fill_type="solid"))
            memory_rule = CellIsRule(operator="equal",
                                     formula=['"Memory"'],
                                     font=Font(bold=True, color="9C0006"),
                                     fill=PatternFill(start_color="FFC7CE",
                                                      end_color="FFC7CE",
                                                      fill_type="solid"))
            ws.conditional_formatting.add(rng, compute_rule)
            ws.conditional_formatting.add(rng, memory_rule)

        # Summary row
        sr = len(rows_data) + 3
        for c in range(1, len(op_headers) + 1):
            cell = ws.cell(row=sr, column=c)
            cell.fill = sum_fill
            cell.font = bold
            cell.border = thin
        ws.cell(row=sr, column=1, value="TOTAL").alignment = center
        fd, ld = 2, len(rows_data) + 1
        ws.cell(row=sr, column=4, value=f"=SUM(D{fd}:D{ld})").number_format = "#,##0.000000"
        ws.cell(row=sr, column=5, value=f"=SUM(E{fd}:E{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=6, value=f"=SUM(F{fd}:F{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=7, value=f"=SUM(G{fd}:G{ld})").number_format = "#,##0.000"
        ws.cell(row=sr, column=10, value=f"=SUM(J{fd}:J{ld})").number_format = "#,##0.000"
        ws.freeze_panes = "A2"
        _set_col_widths(ws, [20, 24, 18, 11, 11, 11, 11, 12, 12, 12, 12, 24])

        # --- Fill Overview module row ---
        ov_r = cur_ov_row
        cur_ov_row += 1
        ws0.cell(row=ov_r, column=1, value=mod_name).border = thin
        # Count (editable for blocks)
        c_count = ws0.cell(row=ov_r, column=2, value=count)
        c_count.border = thin
        c_count.number_format = "0"
        c_count.alignment = center
        if count > 1:
            c_count.fill = edit_fill
            c_count.font = edit_font
        # C: GOPs
        c_gops = ws0.cell(row=ov_r, column=3)
        c_gops.value = f"='{mod_name}'!D{sr}"
        c_gops.number_format = "#,##0.000000"
        c_gops.border = thin
        # D: I/O (MB)
        c_io = ws0.cell(row=ov_r, column=4)
        c_io.value = f"='{mod_name}'!G{sr}"
        c_io.number_format = "#,##0.000"
        c_io.border = thin
        # E: Est. Time (s)
        c_est = ws0.cell(row=ov_r, column=5)
        c_est.value = f"='{mod_name}'!J{sr}/1000000"
        c_est.number_format = "#,##0.000000"
        c_est.border = thin
        # F: Total Time (s)
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
        c_phase = ws0.cell(row=ov_r, column=7, value=phase)
        c_phase.border = thin
        c_phase.alignment = center

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
            cell.alignment = center
        ws0.cell(row=r, column=1, value=label).alignment = left
        if rows:
            total_time_formula = "+".join(f"F{x}" for x in rows)
            if label == "Prefill Total" and has_vit:
                total_time_formula += f"+{PREPROCESS_TIME_REF}"
            ws0.cell(row=r, column=3,
                     value="=" + "+".join(f"C{x}*B{x}"
                                          for x in rows)).number_format = "#,##0.000000"
            ws0.cell(row=r, column=4,
                     value="=" + "+".join(f"D{x}*B{x}" for x in rows)).number_format = "#,##0.000"
            ws0.cell(row=r, column=6, value="=" + total_time_formula).number_format = "#,##0.000000"

    # ============ Back-fill TTFT / Tokens/s at top (rows 4, 5) ============
    # TTFT row (4)
    ws0.cell(row=4, column=2, value=f"=F{phase_r}").number_format = "#,##0.000000"
    ws0.cell(row=4, column=4, value=f"=IF(F{phase_r}=0,0,C{phase_r}/{TOPS_REF}/1000/F{phase_r})")
    ws0.cell(row=4, column=6, value=f"=IF(F{phase_r}=0,0,D{phase_r}/{BW_REF}/1000/F{phase_r})")
    # Tokens/s row (5)
    ws0.cell(row=5, column=2,
             value=f"=IF(F{phase_r+1}=0,0,1/F{phase_r+1})").number_format = "#,##0.00"
    ws0.cell(row=5,
             column=4,
             value=f"=IF(F{phase_r+1}=0,0,C{phase_r+1}/{TOPS_REF}/1000/F{phase_r+1})")
    ws0.cell(row=5,
             column=6,
             value=f"=IF(F{phase_r+1}=0,0,D{phase_r+1}/{BW_REF}/1000/F{phase_r+1})")
    # Re-apply number format for util cells (set before formula insertion was overwritten)
    for r in (4, 5):
        for c in (4, 6):
            ws0.cell(row=r, column=c).number_format = "0.00%"
            ws0.cell(row=r, column=c).font = util_font
            ws0.cell(row=r, column=c).fill = util_fill
            ws0.cell(row=r, column=c).alignment = center
        ws0.cell(row=r, column=2).font = result_font
        ws0.cell(row=r, column=2).fill = result_fill
        ws0.cell(row=r, column=2).alignment = center

    # ============ Model Architecture ============
    arch_banner = phase_r + 3
    if model_config:
        _banner(ws0, arch_banner, "Model Architecture", TOTAL_COLS)
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
            ("Seq Length", seq_length if seq_length else ""),
            ("Max Pixels", max_pixels if max_pixels else ""),
            ("Command", cmdline if cmdline else ""),
        ]
        arch_fields = [(k, v) for k, v in arch_fields if v != ""]
        for r_off, (k, v) in enumerate(arch_fields, 1):
            kc = ws0.cell(row=arch_banner + r_off, column=1, value=k)
            kc.border = thin
            kc.font = bold
            # Span value over remaining columns for readability
            ws0.merge_cells(start_row=arch_banner + r_off,
                            start_column=2,
                            end_row=arch_banner + r_off,
                            end_column=TOTAL_COLS)
            c_val = ws0.cell(row=arch_banner + r_off, column=2, value=v)
            c_val.border = thin
            c_val.alignment = left
            if isinstance(v, (int, float)):
                c_val.number_format = "#,##0"
        info_banner = arch_banner + len(arch_fields) + 2
    else:
        info_banner = arch_banner

    # ============ Notes ============
    _banner(ws0, info_banner, "Notes", TOTAL_COLS)
    notes = [
        "Green cells (parameters, ratios, block counts) are editable; all estimates auto-update.",
        "Est.Time = max(Compute, Memory) + Serialism * min(Compute, Memory).",
        "block / block_cache use the first block as representative, multiplied by Count.",
        "Key operators are highlighted in yellow; use chip TOPS (else vector TOPS) for compute.",
    ]
    for i, txt in enumerate(notes, 1):
        ws0.merge_cells(start_row=info_banner + i,
                        start_column=1,
                        end_row=info_banner + i,
                        end_column=TOTAL_COLS)
        cell = ws0.cell(row=info_banner + i, column=1, value=f"• {txt}")
        cell.alignment = left
        cell.font = Font(italic=True, color="595959")

    # Freeze top summary rows for navigation
    ws0.freeze_panes = "A6"
    _set_col_widths(ws0, [26, 16, 14, 14, 16, 16, 12])
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
    parser.add_argument('--max_pixels', type=str, default="",
                        help='max input pixels for vision models, default "" means no vision input')
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
    if args.max_pixels:
        max_pixels = args.max_pixels
    elif config.model_type in ["qwen2_5_vl", "qwen2_vl"]:
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
