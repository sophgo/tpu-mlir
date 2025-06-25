#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import sys
import os
import shutil

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'test'))
sys.path.append(test_dir)
train_test_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'python', 'tools', 'train', 'test'))
sys.path.append(train_test_dir)
test_custom_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'customlayer', 'test'))
sys.path.append(test_custom_dir)
REGRESSION_PATH = os.getenv('REGRESSION_PATH')

from chip import *
import argparse
import subprocess


class MAIN_ENTRY(object):

    def __init__(self, test_type):
        self.test_type = test_type
        self.is_basic = test_type == "basic"

        # yapf: enable
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
        self.max_workers = os.cpu_count()
        print(f"### max_workers: {self.max_workers}")
        self.log_dir = os.path.join(REGRESSION_PATH, "regression_op_log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.task_file = os.path.join(self.log_dir, f"regression_{self.test_type}_tast.txt")
        self.task_log = os.path.join(self.log_dir, f"regression_{self.test_type}_task.log")
        self.commands = []

    def print_log(self, log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            context = f.read()
            print(context)

    def run_command(self, command):
        GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m"
        try:
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}"
                  )  # Print the command in green
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # Print the error message in red
            print(f"{RED_COLOR}Error: Command failed with return code {e.returncode}{RESET_COLOR}")
            print(f"{RED_COLOR}Failed command: {' '.join(command)}{RESET_COLOR}")
            # Exit the program with the same return code as the failed command
            sys.exit(e.returncode)

    def execute_commands(self):
        if len(self.commands) == 0:
            return
        with open(self.task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()
        halt_now = ""
        if self.is_basic:
            halt_now = f"--halt now,fail=1"
        parallel_cmd = [
            "parallel", f"-j {self.max_workers}", halt_now, "--verbose",
            f"--joblog {self.task_log}", f"< {self.task_file}"
        ]
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)])

    def send_regression_net(self, model_name, chip, num_core):
        run_model = os.path.join(REGRESSION_PATH, "run_model.py")
        self.commands.append(
            f"python {run_model} {model_name} --chip {chip} --mode {self.test_type} --num_core {num_core} > {self.log_dir}/run_model_{model_name}_{chip}_{num_core}core.log\n"
        )

    def send_script_test(self, source):
        script_path = os.path.join(REGRESSION_PATH, f"script_test/{source}.sh")
        self.commands.append(f"bash {script_path} > {self.log_dir}/script_test_{source}.log\n")

    def run_script_test(self):
        # run scripts under $REGRESSION_OUT/script_test
        print("======= script test ======")
        self.script_basic = [
            "test1", "test2", "test5", "test9", "test_llm0", "test12", 'test_modelzoo',
            'test_encrypt', "test_pruning"
        ]
        self.script_extend = [
            "test3", "test4", "test6", "test7", "test8", "test10", "test_llm1", "test_tdb"
        ]
        sources = self.script_basic
        if not self.is_basic:
            sources += self.script_extend

        for source in sources:
            self.send_script_test(source)

    def run_cuda_test(self):
        if self.is_basic:
            self.send_script_test("test_cuda")

    def run_op_onnx_test(self):
        import test_onnx
        # send onnx test
        onnx_tester = test_onnx.ONNX_IR_TESTER()
        chips = ["bm1684x", "bm1688", "bm1684", "cv183x", "mars3"]
        simple = "--simple" if self.is_basic else ""
        for chip in chips:
            for case, (_, bm1684_support, bm1684x_support, bm1688_support, cv183x_support,
                       bm1690_support, mars3_support) in onnx_tester.test_cases.items():
                if chip == "bm1684" and not bm1684_support:
                    continue
                if chip == "bm1684x" and not bm1684x_support:
                    continue
                if chip == "bm1688" and not bm1688_support:
                    continue
                if chip == "cv183x" and not cv183x_support:
                    continue
                if chip == "mars3" and not mars3_support:
                    continue
                self.commands.append(
                    f"test_onnx.py --case {case} --chip {chip} {simple} > {self.log_dir}/test_onnx_{case}_{chip}.log\n"
                )
        # send 1690 fp8
        for case in onnx_tester.cases_fp8:
            self.commands.append(
                f"test_onnx.py --case {case} --chip bm1690 {simple} > {self.log_dir}/test_onnx_{case}_bm1690.log\n"
            )
        del onnx_tester

    def run_op_torch_test(self):
        import test_torch
        import test_tpulang
        import test_custom_tpulang
        # send torch test
        torch_tester = test_torch.TORCH_IR_TESTER()
        chips = ["bm1684", "bm1684x", "bm1688", "cv183x", "mars3"]
        simple = "--simple" if self.is_basic else ""
        for chip in chips:
            for case, (_, bm1684_support, bm1684x_support, bm1688_support, cv183x_support,
                       mars3_support) in torch_tester.test_cases.items():
                if chip == "bm1684" and not bm1684_support:
                    continue
                if chip == "bm1684x" and not bm1684x_support:
                    continue
                if chip == "bm1688" and not bm1688_support:
                    continue
                if chip == "cv183x" and not cv183x_support:
                    continue
                if chip == "mars3" and not mars3_support:
                    continue
                self.commands.append(
                    f"test_torch.py --case {case} --chip {chip} {simple} > {self.log_dir}/test_torch_{case}_{chip}.log\n"
                )
        del torch_tester
        # send tpulang test
        if not self.is_basic:
            tpulang_tester = test_tpulang.TPULANG_IR_TESTER()
            chips = ["bm1684x", "bm1688"]
            for chip in chips:
                for case, (_, bm1684x_support,
                           bm1688_support) in tpulang_tester.test_function.items():
                    if chip == "bm1684x" and not bm1684x_support:
                        continue
                    if chip == "bm1688" and not bm1688_support:
                        continue
                    self.commands.append(
                        f"test_tpulang.py --case {case} --chip {chip} {simple} > {self.log_dir}/test_tpulang_{case}_{chip}.log\n"
                    )
            del tpulang_tester
        # send custom tpulang test
        custom_tester = test_custom_tpulang.CUSTOM_TPULANG_TESTER()
        custom_py = test_custom_tpulang.__file__
        chips = ["bm1684x", "bm1688"]
        for chip in chips:
            for case, (_, bm1684x_support, bm1688_support) in custom_tester.test_cases.items():
                if chip == "bm1684x" and not bm1684x_support:
                    continue
                if chip == "bm1688" and not bm1688_support:
                    continue
                self.commands.append(
                    f"python3 {custom_py} --case {case} --chip {chip} {simple} > {self.log_dir}/test_custom_tpulang_{case}_{chip}.log\n"
                )
        del custom_tester

    def run_model_test(self, multi_core: bool = False):
        # run llm test
        self.run_llm_test()
        # run model list
        model_list = None
        if multi_core:
            model_list = basic_multi_core_model_list if self.is_basic else full_multi_core_model_list
        else:
            model_list = basic_model_list if self.is_basic else full_model_list

        for idx, chip in enumerate(chip_support.keys()):
            num_core = 1
            if multi_core:
                if chip not in multi_core_info.keys():
                    continue
                num_core = multi_core_info[chip]

            cur_model_list = [
                model_name for model_name, do_test in model_list.items() if do_test[idx]
            ]

            for model in cur_model_list:
                self.send_regression_net(model, chip, num_core)
        # ===== run other test =======
        import test_MaskRCNN
        # send MaskRCNN test only bm1684x
        maskrcnn_tester = test_MaskRCNN.MaskRCNN_IR_TESTER()
        for case in maskrcnn_tester.test_cases.keys():
            self.commands.append(
                f"test_MaskRCNN.py --case {case} --chip bm1684x > {self.log_dir}/test_MaskRCNN_{case}_bm1684x.log\n"
            )
        del maskrcnn_tester
        # send 1690 fx
        import test_fx
        for chip in ["bm1684x", "bm1688", "bm1690"]:
            fx_tester = test_fx.FX_IR_TESTER(chip=chip)
            for case in fx_tester.test_cases.keys():
                if fx_tester.check_support(case):
                    self.commands.append(
                        f"test_fx.py --case {case} --chip {chip}> {self.log_dir}/test_fx_{chip}_{case}.log\n"
                    )
                else:
                    continue

            del fx_tester

    def run_multi_core_test(self):
        self.run_model_test(multi_core=True)

    def run_llm_test(self):
        NNMODELS_PATH = os.getenv('NNMODELS_PATH')
        LLM_MODELS = ["Qwen2.5-VL-3B-Instruct-GPTQ-Int4", "InternVL3-1B-AWQ"]
        for model in LLM_MODELS:
            MODEL_PATH = os.path.join(NNMODELS_PATH, "llm_models", model)
            self.run_command([
                "llm_convert.py", "-m", MODEL_PATH, "-s", "2048", "-q", "w4bf16", "-c", "bm1684x",
                "--out_dir", "llm_output", "--max_pixels", "672,896"
            ])
            # check result
            input_ref = os.path.join(MODEL_PATH, "block_cache_0_input.npz")
            output_ref = os.path.join(MODEL_PATH, "block_cache_0_output.npz")
            bmodel = os.path.join("llm_output",
                                  model.lower() + "_w4bf16_seq2048_bm1684x_1dev",
                                  "block_cache_0.bmodel")
            self.run_command([
                "model_runner.py", "--input", input_ref, "--model", bmodel, "--output", "output.npz"
            ])
            self.run_command(["npz_tool.py", "compare", output_ref, "output.npz"])

    def run_all(self, test_set):
        for test in test_set:
            self.test_set[test]()

        self.execute_commands()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=['all', 'basic'],
                        help="whether do all model test, 'all' runs all modes, 'basic' runs basic models f16 and int8 sym only")
    choices = ["onnx", "torch", "script", "model", "multi_core_model", "cuda"]
    parser.add_argument("--test_set", default=choices, type=str.lower, nargs="+", choices=choices,
                        help="run test set individually.")
    # yapf: enable
    args = parser.parse_args()
    LOCK_FILE = "/tmp/bmchip_mux.lock"
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    os.environ["CMODEL_LOCKFILE"] = LOCK_FILE
    dir = os.path.join(REGRESSION_PATH, "regression_out")
    cur_dir = os.getcwd()
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    main_entry = MAIN_ENTRY(args.test_type)

    main_entry.run_all(args.test_set)

    print(f"TEST {args.test_type} {args.test_set} PASSED")
    os.chdir(cur_dir)
    shutil.rmtree(dir)
