#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Shared helpers for the operator-level Python regression tests.

The three test entry points ``test_onnx.py``, ``test_torch.py`` and
``test_mlir.py`` historically duplicated a fair amount of infrastructure:
chip-support lookup, calibration-table generation, random data, working
directory handling, and the ``test_one_case_in_all`` / ``test_all`` runners.
This module consolidates them so every tester behaves the same way and the
per-file files stay focused on the operator cases they exercise.

The chip-support tables in each tester intentionally remain dense tuples of
``Y``/``N`` flags so that adding/auditing a row stays a one-line operation.
This module provides a :func:`make_chip_resolver` helper to turn a tuple of
flags into a clean per-chip boolean.
"""

from __future__ import annotations

import os
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.timer import Timer
from utils.auto_remove import clean_kmp_files

__all__ = [
    "Y",
    "N",
    "make_chip_resolver",
    "rand_data",
    "generate_random",
    "square_rooted",
    "cosine_similarity",
    "make_simple_calibration_table",
    "change_directory",
    "run_all_cases",
]

# Conventional aliases used in the per-row support tables.
Y: bool = True
N: bool = False


# ---------------------------------------------------------------------------
# Chip support resolver
# ---------------------------------------------------------------------------
def make_chip_resolver(
    chip_columns: Sequence[str],
    aliases: Optional[Mapping[str, str]] = None,
) -> Callable[[str, Sequence[bool]], bool]:
    """Return a resolver that maps a chip name + per-row Y/N flags to a bool.

    Args:
        chip_columns: Ordered chip names corresponding to the flags that follow
            the test function in every row of a ``test_cases`` dict.
        aliases: Optional mapping ``{actual_chip -> column_chip}`` used when
            several physical chips share the same support column (e.g. all
            ``cv18*`` variants share the ``cv183x`` column).

    Returns:
        A function ``resolve(chip, flags) -> bool``. ``flags`` must have the
        same length as ``chip_columns``; the leading test-function entry of
        the row must be sliced off by the caller.
    """
    columns: Tuple[str, ...] = tuple(chip_columns)
    alias_map: Dict[str, str] = dict(aliases or {})

    def resolve(chip: str, flags: Sequence[bool]) -> bool:
        if len(flags) != len(columns):
            raise ValueError(
                f"flags has length {len(flags)} but {len(columns)} chip columns are configured")
        column_chip = alias_map.get(chip, chip)
        if column_chip not in columns:
            return False
        return bool(flags[columns.index(column_chip)])

    return resolve


# ---------------------------------------------------------------------------
# Random data
# ---------------------------------------------------------------------------
_INT_DTYPES = ("int32", "uint32", "int16", "uint16", "int8", "uint8")


def rand_data(
    shape: Sequence[int],
    dtype: str,
    min_val: float = -10,
    max_val: float = 10,
    seed: Optional[int] = None,
    int_satu: bool = False,
) -> np.ndarray:
    """Generate random data with the requested shape and dtype.

    Float arrays are sampled from ``randn`` and clipped to ``[min_val, max_val]``.
    Integer arrays are sampled uniformly in ``[0, 127)``; when ``int_satu`` is
    set the result is additionally clipped to ``[min_val, max_val]``.
    """
    if seed is not None:
        np.random.seed(seed)

    if dtype in ("float32", "float16"):
        data = np.random.randn(*shape).astype(dtype)
        return np.clip(data, min_val, max_val)

    if dtype in _INT_DTYPES:
        data = np.random.randint(0, 127, size=shape).astype(dtype)
        if int_satu:
            return np.clip(data, min_val, max_val)
        return data

    raise ValueError(f"Unsupported data type: {dtype}")


def generate_random(
    shape: Sequence[int],
    dtype: str = "float32",
    min_val: float = -10,
    max_val: float = 10,
) -> np.ndarray:
    """Uniform random helper used by the Torch tester.

    Returns an array of ``shape`` filled with samples from
    ``Uniform(min_val, max_val)`` cast to ``dtype``. The legacy behaviour of
    casting ``bool`` via ``int`` is preserved.
    """
    scale = max_val - min_val
    data = np.random.rand(*shape) * scale + min_val
    if dtype == "bool":
        data = data.astype("int")
    return data.astype(dtype)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def square_rooted(x: np.ndarray) -> float:
    """L2 norm helper used by :func:`cosine_similarity`."""
    return float(np.sqrt(np.sum(np.power(x, 2))))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Round-to-3 cosine similarity between two flat arrays."""
    numerator = float(np.sum(x * y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------
_INT_QUANT_MODES = ("int8", "int4", "w4int8")
_FP8_QUANT_MODES = ("f8e4m3", "f8e5m2")


def make_simple_calibration_table(
    tensors: Mapping[str, np.ndarray],
    table_name: str,
    quant_mode: Optional[str] = None,
) -> None:
    """Write a "naive" calibration table for the tester.

    The table contains one row per tensor with ``name threshold min max`` where
    ``threshold = 1.1 * max(|min|, |max|) + 0.01``. For fp8 modes the table is
    suffixed with ``_<mode>`` and gains an fp8 header line; for non-int / non-fp8
    modes the function returns silently (the torch tester always wants the
    table written, so it passes ``quant_mode=None``).
    """
    if quant_mode in _FP8_QUANT_MODES:
        table_name = f"{table_name}_{quant_mode}"
    elif quant_mode is not None and quant_mode not in _INT_QUANT_MODES:
        return

    with open(table_name, "w") as f:
        if quant_mode in _FP8_QUANT_MODES:
            f.write("#tpu-mlir-fp8 caliration table\n")
        for name, tensor in tensors.items():
            flat = tensor.flatten()
            max_val = float(np.max(flat))
            min_val = float(np.min(flat))
            if max_val == min_val:
                max_val += 0.01
            threshold = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
            f.write(f"{name} {threshold} {min_val} {max_val}\n")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
@contextmanager
def change_directory(path: str):
    """Temporarily ``chdir`` into ``path`` (created if missing)."""
    original_dir = os.getcwd()
    try:
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Generic test runner
# ---------------------------------------------------------------------------
class _TesterProtocol:
    """Structural type expected by :func:`run_all_cases`.

    Any tester that exposes:
        - ``chip``: ``str``
        - ``test_cases``: ``Mapping[str, Tuple[Callable, ...]]``
        - ``check_support(case_name) -> bool``
        - ``test_single(case_name) -> None``
    is accepted. (Kept as a documentation-only base class so we don't pull in
    ``typing.Protocol`` for compatibility.)
    """


def _run_one(
    tester: Any,
    case: str,
    error_cases: List[str],
    success_cases: List[str],
    *,
    print_traceback: bool = False,
) -> None:
    timer = Timer()
    try:
        tester.test_single(case)
    except Exception:
        if print_traceback:
            traceback.print_exc()
        error_cases.append(f"{case}:{int(timer.elapsed_time())}s")
        return
    success_cases.append(f"{case}:{int(timer.elapsed_time())}s")


def run_all_cases(
    tester: Any,
    *,
    label: str,
    print_traceback: bool = False,
    print_failure_command: bool = False,
    script_name: Optional[str] = None,
) -> List[str]:
    """Iterate every supported case on ``tester`` and report success / failure.

    Args:
        tester: An object exposing ``test_cases``, ``check_support`` and
            ``test_single`` (see :class:`_TesterProtocol`).
        label: Short tag printed in the final banner, e.g. ``"test_onnx.py"``.
        print_traceback: When ``True`` (Torch tester behaviour) failures print
            a full traceback before being recorded.
        print_failure_command: When ``True`` (ONNX tester behaviour) the
            failure summary includes a reproduce command per failed case.
        script_name: Script path used in the reproduce command. Defaults to
            ``label`` when omitted.

    Returns:
        The list of ``"<case>:<elapsed>s"`` failure entries.
    """
    error_cases: List[str] = []
    success_cases: List[str] = []
    for case in tester.test_cases:
        if tester.check_support(case):
            _run_one(tester, case, error_cases, success_cases, print_traceback=print_traceback)
    print(f"Success: {success_cases}")
    print(f"Failure: {error_cases}")
    if error_cases:
        print(f"====== {label} --chip {tester.chip} TEST Failed ======")
        if print_failure_command:
            script = script_name or label
            for entry in error_cases:
                case_name = entry.split(":")[0]
                print(f"{script} --chip {tester.chip} --case {case_name} failed")
    else:
        print(f"====== {label} --chip {tester.chip} TEST Success ======")
    clean_kmp_files()
    return error_cases
