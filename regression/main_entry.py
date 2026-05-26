#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Top-level regression driver invoked by ``regression/run.sh`` / ``run_all.sh``.

Composes shell command lines for each individual op/model test, then dispatches
them through GNU ``parallel`` so the per-commit CI can fan-out across all CPU
cores. Each test set is implemented as a ``run_*`` method that enqueues
commands into ``self.commands``; ``execute_commands`` flushes the queue.

Per-test chip support is queried from the corresponding tester's
``check_support`` method — this module no longer unpacks the
``tester.test_cases`` tuple layout positionally.
"""

import argparse
import enum
import os
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional

# Make sibling python test packages importable as bare modules
# (``test_onnx``, ``test_torch`` etc. live in ``python/test`` and on PATH at
# runtime, but importing them here requires sys.path setup).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _extra in (
        os.path.join(_REPO_ROOT, "python", "test"),
        os.path.join(_REPO_ROOT, "python", "tools", "train", "test"),
        os.path.join(_REPO_ROOT, "third_party", "customlayer", "test"),
):
    if _extra not in sys.path:
        sys.path.append(_extra)

REGRESSION_PATH = os.getenv("REGRESSION_PATH")

from chip import (  # noqa: E402  (sys.path tweak above)
    basic_model_list, basic_multi_core_model_list, chip_support, full_model_list,
    full_multi_core_model_list, models_for_chip, multi_core_info,
)

# ANSI color codes for human-readable command echoing.
_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


class Status(enum.IntEnum):
    """Status codes shared with ``dailyrelease_cvimodel.py``."""
    PASSED = 0
    FAILED = 1
    TIMEOUT = 2


# Chip lists used by the op-level tests. Kept here (rather than in chip.py)
# because they reflect *which chips we currently run in CI* for that test
# family, which is a CI policy decision, not an intrinsic chip capability.
_ONNX_CHIPS = ("bm1684x", "bm1688", "bm1684", "cv183x", "cv184x")
_TORCH_CHIPS = ("bm1684x", "bm1688", "cv184x")
_CUSTOM_TPULANG_CHIPS = ("bm1684x", "bm1688")
_TPULANG_CHIPS = ("bm1684x", "bm1688")
_MASKRCNN_CHIPS = ("bm1684x", )

# Script-test buckets — kept as class-level constants for easy editing.
_SCRIPT_BASIC = (
    "test1",
    "test2",
    "test5",
    "test9",
    "test_llm0",
    "test12",
    "test_modelzoo",
    "test_encrypt",
    "test_pruning",
    "struct_optimize_pattern_test",
    # "struct_optimize_pattern_test2",
    "test_profile",
)
_SCRIPT_EXTEND = (
    "test3",
    "test4",
    "test6",
    "test7",
    "test8",
    "test10",
    "test_llm1",
    "test_tdb",
)

# LLM smoke-test models that ship with the toolchain.
_LLM_MODELS = (
    ("Qwen2.5-VL-3B-Instruct-GPTQ-Int4", "w4bf16"),
    ("InternVL3-1B-AWQ", "w4f16"),
)


def _log_path(log_dir: str, *parts: str) -> str:
    """Build a log file path under ``log_dir`` from ``parts``."""
    return os.path.join(log_dir, "_".join(parts) + ".log")


class MAIN_ENTRY(object):
    """Build and run the per-commit / nightly regression command queue."""

    def __init__(self, test_type: str):
        self.test_type = test_type
        self.is_basic = test_type == "basic"

        self.test_set = {
            "onnx": self.run_op_onnx_test,
            "torch": self.run_op_torch_test,
            "script": self.run_script_test,
            "model": self.run_model_test,
            "multi_core_model": self.run_multi_core_test,
            "cuda": self.run_cuda_test,
        }

        self.results = []
        self.time_cost = []
        self.max_workers = int(os.cpu_count() or 1)
        print(f"### max_workers: {self.max_workers}")
        self.log_dir = os.path.join(REGRESSION_PATH, "regression_op_log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.task_file = os.path.join(self.log_dir, f"regression_{test_type}_task.txt")
        self.task_log = os.path.join(self.log_dir, f"regression_{test_type}_task.log")
        self.commands: List[str] = []

    # ----- low-level helpers ------------------------------------------------

    def print_log(self, log_file: str) -> None:
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())

    def run_command(self, command: List[str]) -> None:
        """Run ``command`` synchronously; exit the process on failure."""
        printable = " ".join(command)
        print(f"{_GREEN}Executing command: \n{printable}{_RESET}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"{_RED}Error: Command failed with return code {e.returncode}{_RESET}")
            print(f"{_RED}Failed command: {printable}{_RESET}")
            sys.exit(e.returncode)

    def execute_commands(self) -> None:
        """Flush the queued commands through GNU ``parallel``."""
        if not self.commands:
            return
        with open(self.task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()
        halt_now = "--halt now,fail=1" if self.is_basic else ""
        parallel_cmd = [
            "parallel",
            f"-j {self.max_workers}",
            halt_now,
            "--verbose",
            f"--joblog {self.task_log}",
            f"< {self.task_file}",
        ]
        self.run_command(["bash", "-c", " ".join(parallel_cmd)])

    def _enqueue(self, command: str, log_name: str) -> None:
        """Append ``command`` with stdout redirected to ``<log_dir>/<log_name>.log``."""
        log = os.path.join(self.log_dir, f"{log_name}.log")
        self.commands.append(f"{command} > {log}\n")

    def send_regression_net(self, model_name: str, chip: str, num_core: int) -> None:
        run_model = os.path.join(REGRESSION_PATH, "run_model.py")
        cmd = (f"python {run_model} {model_name} --chip {chip} "
               f"--mode {self.test_type} --num_core {num_core}")
        self._enqueue(cmd, f"run_model_{model_name}_{chip}_{num_core}core")

    def send_script_test(self, source: str) -> None:
        script_path = os.path.join(REGRESSION_PATH, "script_test", f"{source}.sh")
        self._enqueue(f"bash {script_path}", f"script_test_{source}")

    def _enqueue_tester_cases(self,
                              tester,
                              script_name: str,
                              log_prefix: str,
                              chips: Iterable[str],
                              extra_flags: str = "") -> None:
        """Enqueue ``script_name --case CASE --chip CHIP`` for every supported case.

        ``tester`` must be a freshly-constructed instance whose ``chip`` we are
        free to mutate. Cases are selected via the tester's own
        ``check_support`` method, which keeps the chip support matrix
        encapsulated in the tester module.
        """
        for chip in chips:
            tester.chip = chip
            for case in tester.test_cases.keys():
                if not tester.check_support(case):
                    continue
                cmd = f"{script_name} --case {case} --chip {chip} {extra_flags}".rstrip()
                self._enqueue(cmd, f"{log_prefix}_{case}_{chip}")

    # ----- script tests -----------------------------------------------------

    def run_script_test(self) -> None:
        print("======= script test ======")
        sources = list(_SCRIPT_BASIC)
        if not self.is_basic:
            sources += list(_SCRIPT_EXTEND)
        for source in sources:
            self.send_script_test(source)

    def run_cuda_test(self) -> None:
        if self.is_basic:
            self.send_script_test("test_cuda")

    # ----- op-level tests ---------------------------------------------------

    def run_op_onnx_test(self) -> None:
        import test_onnx
        simple = "--simple" if self.is_basic else ""
        onnx_tester = test_onnx.ONNX_IR_TESTER()
        try:
            self._enqueue_tester_cases(onnx_tester,
                                       "test_onnx.py",
                                       "test_onnx",
                                       _ONNX_CHIPS,
                                       extra_flags=simple)
            # bm1690-only fp8 cases (no chip-support gating: all run on bm1690).
            for case in onnx_tester.cases_fp8:
                cmd = f"test_onnx.py --case {case} --chip bm1690 {simple}".rstrip()
                self._enqueue(cmd, f"test_onnx_{case}_bm1690")
        finally:
            del onnx_tester

    def run_op_torch_test(self) -> None:
        import test_custom_tpulang
        import test_torch
        simple = "--simple" if self.is_basic else ""

        torch_tester = test_torch.TORCH_IR_TESTER()
        try:
            self._enqueue_tester_cases(torch_tester,
                                       "test_torch.py",
                                       "test_torch",
                                       _TORCH_CHIPS,
                                       extra_flags=simple)
        finally:
            del torch_tester

        custom_tester = test_custom_tpulang.CUSTOM_TPULANG_TESTER()
        custom_py = test_custom_tpulang.__file__
        try:
            self._enqueue_tester_cases(custom_tester,
                                       f"python3 {custom_py}",
                                       "test_custom_tpulang",
                                       _CUSTOM_TPULANG_CHIPS,
                                       extra_flags=simple)
        finally:
            del custom_tester

    # ----- model-level tests ------------------------------------------------

    def run_model_test(self, multi_core: bool = False) -> None:
        # Smoke LLM conversion before fanning out the model regression set.
        self.run_llm_test()

        if multi_core:
            model_list = basic_multi_core_model_list if self.is_basic else full_multi_core_model_list
        else:
            model_list = basic_model_list if self.is_basic else full_model_list

        for chip in chip_support.keys():
            num_core = 1
            if multi_core:
                if chip not in multi_core_info:
                    continue
                num_core = multi_core_info[chip]
            for model in models_for_chip(model_list, chip):
                self.send_regression_net(model, chip, num_core)

        # ----- additional op-driven model tests (bm1684x only) -----
        import test_MaskRCNN
        maskrcnn_tester = test_MaskRCNN.MaskRCNN_IR_TESTER()
        try:
            for chip in _MASKRCNN_CHIPS:
                maskrcnn_tester.chip = chip
                for case in maskrcnn_tester.test_cases.keys():
                    cmd = f"test_MaskRCNN.py --case {case} --chip {chip}"
                    self._enqueue(cmd, f"test_MaskRCNN_{case}_{chip}")
        finally:
            del maskrcnn_tester

        # ----- tpulang (nightly only) -----
        if not self.is_basic:
            import test_tpulang
            tpulang_tester = test_tpulang.TPULANG_IR_TESTER()
            try:
                for chip in _TPULANG_CHIPS:
                    tpulang_tester.chip = chip
                    for case in tpulang_tester.test_function.keys():
                        if not tpulang_tester.check_support(case):
                            continue
                        cmd = f"test_tpulang.py --case {case} --chip {chip}"
                        self._enqueue(cmd, f"test_tpulang_{case}_{chip}")
            finally:
                del tpulang_tester

    def run_multi_core_test(self) -> None:
        self.run_model_test(multi_core=True)

    # ----- LLM conversion smoke test ---------------------------------------

    def run_llm_test(self) -> None:
        nnmodels_path = os.getenv("NNMODELS_PATH")
        if not nnmodels_path:
            print("NNMODELS_PATH unset; skipping LLM smoke test.")
            return
        for model, quant_type in _LLM_MODELS:
            model_path = os.path.join(nnmodels_path, "llm_models", model)
            self.run_command([
                "llm_convert.py",
                "-m",
                model_path,
                "-s",
                "2048",
                "-c",
                "bm1684x",
                "--out_dir",
                "llm_output",
                "--max_pixels",
                "672,896",
            ])
            input_ref = os.path.join(model_path, "block_cache_0_input.npz")
            output_ref = os.path.join(model_path, "block_cache_0_output.npz")
            bmodel = os.path.join(
                "llm_output",
                f"{model.lower()}_{quant_type}_seq2048_bm1684x_1dev_static",
                "block_cache_0",
                "block_cache_0.bmodel",
            )
            self.run_command([
                "model_runner.py",
                "--input",
                input_ref,
                "--model",
                bmodel,
                "--output",
                "output.npz",
            ])
            self.run_command(["npz_tool.py", "compare", output_ref, "output.npz"])

    # ----- top-level driver -------------------------------------------------

    def run_all(self, test_set: Iterable[str]) -> None:
        default_workers = int(os.cpu_count() or 1)
        for test in test_set:
            # torch tests fork heavy worker subprocesses themselves; throttle.
            self.max_workers = max(1, default_workers // 8) if test == "torch" else default_workers
            self.test_set[test]()
        self.execute_commands()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TEST_SET_CHOICES = ("onnx", "torch", "script", "model", "multi_core_model", "cuda")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=("all", "basic"),
                        help="'all' runs all modes; 'basic' runs f16 + int8_sym only (per-commit CI)")
    parser.add_argument("--test_set", default=list(_TEST_SET_CHOICES), type=str.lower,
                        nargs="+", choices=_TEST_SET_CHOICES,
                        help="run only the listed test sets")
    # yapf: enable
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    lock_file = "/tmp/bmchip_mux.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
    os.environ["CMODEL_LOCKFILE"] = lock_file

    out_dir = os.path.join(REGRESSION_PATH, "regression_out")
    cur_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    try:
        driver = MAIN_ENTRY(args.test_type)
        driver.run_all(args.test_set)
        print(f"TEST {args.test_type} {args.test_set} PASSED")
    finally:
        os.chdir(cur_dir)
        shutil.rmtree(out_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
