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
train_test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'tools', 'train', 'test'))
sys.path.append(train_test_dir)
test_custom_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'customlayer', 'test'))
sys.path.append(test_custom_dir)

regression_out_path = os.path.join(os.getenv("REGRESSION_PATH"), "regression_out")
assert regression_out_path
regression_log_path = os.path.join(regression_out_path, "../regression_op_log")
os.makedirs(regression_log_path, exist_ok=True)

from utils.timer import Timer
from chip import *
from run_model import MODEL_RUN
import test_tpulang
import test_torch
import test_onnx
import test_fx
import test_custom_tpulang
import argparse
import logging
import subprocess
from utils.mlir_shell import _os_system_log

SUCCESS = 0
FAILURE = 1

class Status:
    PASSED = 'PASSED'
    FAILED = 'FAILED'
    TIMEOUT = 'TIMEOUT'


def timeit(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} ran in {end - start:.2f} seconds")
        return result

    return wrapper


class MAIN_ENTRY(object):

    def __init__(self, test_type, disable_thread: bool, concise_log=False):
        self.test_type = test_type
        self.disable_thread = disable_thread
        self.current_dir = os.getcwd()
        self.is_basic = test_type == "basic"
        self.concise_log = concise_log
        # yapf: disable
        self.op0_test_types = {
            # op_source: (tester, test_all_func, chips)
            "onnx_int4": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_int4,  ["bm1688"]),
            "onnx_fp8": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_fp8,  ["bm1690"]),
            "onnx":     (test_onnx.ONNX_IR_TESTER,       test_onnx.test_all,   ["bm1684", "bm1684x", "bm1688", "cv183x", "mars3"]),
            "fx":       (test_fx.FX_IR_TESTER,           test_fx.test_all, ["bm1690"]),
        }
        self.op1_test_types = {
            # op_source: (tester, test_all_func, chips)
            "torch":    (test_torch.TORCH_IR_TESTER,     test_torch.test_all,  ["bm1684", "bm1684x", "bm1688", "cv183x", "mars3"]),
            "tpulang":  (test_tpulang.TPULANG_IR_TESTER, test_tpulang.test_all, ["bm1684x", "bm1688"]),
            "custom_tpulang":  (test_custom_tpulang.CUSTOM_TPULANG_TESTER, test_custom_tpulang.test_all, ["bm1684x", "bm1688"]),
        }
        if not self.is_basic:
            del self.op1_test_types["tpulang"]

        self.script_basic = ["test1", "test2","test5","test9","test_llm0","test12",'test_modelzoo','test_encrypt', 'test_MaskRCNN']
        self.script_extend = ["test3","test4","test6","test7","test8","test10","test_llm1"]
        # yapf: enable
        self.test_set = {
            "op0": self.run_op0_test,
            "op1": self.run_op1_test,
            "script": self.run_script_test,
            "model": self.run_model_test,
            "multi_core_model": self.run_multi_core_test,
            "cuda": self.run_cuda_test,
            "tdb": self.run_tdb_test,
        }

        self.results = []
        self.time_cost = []
        cpu_count = os.cpu_count()
        self.max_workers = max(cpu_count, 4)
        self.task_file = os.path.join(self.current_dir,  f"{test_type}_task.txt")
        self.commands = []
        self.logger = logging.getLogger()

    def extend_time_cost_list(self, test_type: str, duration: int):
        cur_time_cost = f"{test_type}: {duration} seconds"
        print(cur_time_cost)
        self.time_cost.append(cur_time_cost)

    def add_result(self, test_name: str, status: bool, time: int = 0, error_cases: list = []):
        self.results.append({
            "name": test_name,
            "status": Status.PASSED if status else Status.FAILED,
            "error_cases": error_cases,
            "time": time
        })

    def print_log(self, log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(lines)

    def run_command(self, command, log_file):
        GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m"
        try:
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}"
                  )  # Print the command in green
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            self.print_log(log_file)
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
        task_log = f"{self.task_file}.log"
        halt_now = ""
        if self.is_basic:
            halt_now = " --halt now,fail=1"
        parallel_cmd = [
            "parallel", f"-j {self.max_workers}", halt_now, "--progress", f"--joblog {task_log}",
            f"< {self.task_file}"
        ]
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)], task_log)

    def send_regression_net(self, model_name, chip, num_core):
        run_model = os.path.expandvars(f"$REGRESSION_PATH/run_model.py")
        self.commands.append(f"python {run_model} {model_name} --chip {chip} --mode {self.test_type} --num_core {num_core} \n")


    def _run_op_test(self, op_source, tester, test_all_func, chip):
        print(f"======= test_{op_source}.py ======")
        case_name = f"{op_source}_test_{chip}"
        t = Timer()
        os.makedirs(case_name, exist_ok=True)
        os.chdir(case_name)
        if op_source == "tflite" or op_source == "fx":
            tester = tester(chip=chip, concise_log=self.concise_log)
        else:
            tester = tester(chip=chip, simple=self.is_basic, concise_log=self.concise_log)
        error_cases = test_all_func(tester)
        self.add_result(case_name, not error_cases, int(t.elapsed_time()), error_cases)
        os.chdir(self.current_dir)
        shutil.rmtree(case_name) # too large
        return not error_cases

    def send_script_test(self, source):
        script_path = os.path.expandvars(f"$REGRESSION_PATH/script_test/{source}.sh")
        self.commands.append(f"bash {script_path} \n")

    def _run_script_test(self, source):
        print(f"======= test script:{source}.sh ======")
        case_name = f"test_script_{source}"
        file_handler = logging.FileHandler(filename=case_name + ".log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        if self.concise_log:
            file_logger = logging.getLogger("logger_to_file")
            file_logger.addHandler(file_handler)
            file_logger.propagate = False
            console_logger = logging.getLogger("logger_to_file.console")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            console_logger.addHandler(console_handler)
            console_logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(file_handler)

        t = Timer()
        os.makedirs(case_name, exist_ok=True)
        os.chdir(case_name)

        success = True
        try:
            _os_system_log(os.path.expandvars("$REGRESSION_PATH/script_test/{}.sh".format(source)))
        except:
            success = False
        if self.concise_log:
            file_logger.removeHandler(file_handler)
            console_logger.removeHandler(console_handler)
            file_handler.close()
            console_handler.close()
        else:
            self.logger.removeHandler(file_handler)
            file_handler.close()
        self.add_result(case_name, success, int(t.elapsed_time()))
        os.chdir(self.current_dir)
        shutil.rmtree(case_name) # too large
        return success

    def run_script_test(self):
        # run scripts under $REGRESSION_OUT/script_test
        print("======= script test ======")

        sources = self.script_basic
        if not self.is_basic:
            sources += self.script_extend

        for source in sources:
            self.send_script_test(source)



    def run_cuda_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= cuda test ======")
        ret = self._run_script_test("test_cuda")
        self.extend_time_cost_list("run_cuda", int(t.elapsed_time()))
        return SUCCESS if ret else FAILURE

    def run_op_test(self, op_test_types):
        ret = True
        for op_source in op_test_types.keys():
            t = Timer()
            tester, test_func, chips = op_test_types[op_source]
            for chip in chips:
                ret = self._run_op_test(op_source, tester, test_func, chip)
                # basic test stops once a test failed
                if not ret and self.is_basic:
                    return FAILURE
            self.extend_time_cost_list(f"run_{op_source}", int(t.elapsed_time()))

        return SUCCESS if ret else FAILURE

    def run_op0_test(self):
        self.run_op_test(self.op0_test_types)

    def run_op1_test(self):
        self.run_op_test(self.op1_test_types)

    def run_model_test(self, multi_core: bool = False):
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

    def run_multi_core_test(self):
        self.run_model_test(multi_core=True)

    def run_tdb_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= TDB test ======")
        ret = self._run_script_test("test_tdb")
        self.extend_time_cost_list(f"run_TDB", int(t.elapsed_time()))

        return SUCCESS if ret else FAILURE

    def run_all(self, test_set):
        t = Timer()

        for test in test_set:
            if self.test_set[test]() and self.is_basic:
                return FAILURE
        self.execute_commands()
        self.time_cost.append(f"total time: {int(t.elapsed_time())} seconds")
        return FAILURE if any(result.get("status") != Status.PASSED
                              for result in self.results) else SUCCESS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=['all', 'basic'],
                        help="whether do all model test, 'all' runs all modes, 'basic' runs basic models f16 and int8 sym only")
    choices = ["op0", "op1", "script", "model", "multi_core_model", "cuda", "tdb"]
    parser.add_argument("--test_set", default=choices, type=str.lower, nargs="+", choices=choices,
                        help="run test set individually.")
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--concise_log", action="store_true", help='do test with concise log')
    # yapf: enable
    args = parser.parse_args()

    dir = os.path.expandvars("${REGRESSION_PATH}/regression_out")
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(message)s')

    main_entry = MAIN_ENTRY(args.test_type, args.disable_thread, args.concise_log)

    exit_status = main_entry.run_all(args.test_set)

    # print last 100 lines log of failed cases first
    for result in main_entry.results:
        if result["name"].startswith(tuple(
                main_entry.op0_test_types.keys())) or result["name"].startswith(
                    tuple(main_entry.op1_test_types.keys())):
            continue
        if result["status"] != Status.PASSED:
            log_file = result["name"] + ".log"
            try:
                with open(log_file, 'r') as file:
                    lines = file.readlines()
                    last_100_lines = lines[-100:]
                    content = ''.join(last_100_lines)
                    print(content)
            except:
                print("Failed to open error log file:{}".format(log_file))

    print("============ Time Consum Summary ============")
    for time in main_entry.time_cost:
        print(time)

    print("============ Passed Cases ============")
    for result in main_entry.results:
        if result["status"] == Status.PASSED:
            print("{} {} [{} s]".format(result["name"], result["status"], result["time"]))

    print("============ Failed Cases ============")
    for result in main_entry.results:
        if result["status"] != Status.PASSED:
            print("{} {} [{} s]".format(result["name"], result["status"], result["time"]))
            if result["error_cases"]:
                print("Failed cases: ", result["error_cases"])

    print("TEST {} {}".format(args.test_type, "PASSED" if not exit_status else "FAILED"))
    exit(exit_status)
