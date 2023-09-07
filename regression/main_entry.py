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

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'test'))
sys.path.append(test_dir)
import time
from chip import *
from run_model import MODEL_RUN
import test_tpulang
import test_torch
import test_tflite
import test_onnx
import argparse
import logging
from utils.mlir_shell import _os_system_log


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


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time


class MAIN_ENTRY(object):
    def __init__(self, test_type, disable_thread: bool):
        self.test_type = test_type
        self.disable_thread = disable_thread
        self.current_dir = os.getcwd()
        self.is_basic = test_type == "basic"
        # yapf: disable
        self.op_test_types = {
            # op_source: (tester, test_all_func, chips)
            "onnx_int4": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_int4,  ["bm1686"]),
            "onnx":     (test_onnx.ONNX_IR_TESTER,       test_onnx.test_all,   ["bm1684", "bm1684x", "bm1686", "cv183x"]),
            "tflite":   (test_tflite.TFLITE_IR_TESTER,   test_tflite.test_all, ["bm1684x", "bm1686"]),
            "torch":    (test_torch.TORCH_IR_TESTER,     test_torch.test_all,  ["bm1684", "bm1684x", "bm1686", "cv183x"]),
            "tpulang":  (test_tpulang.TPULANG_IR_TESTER, test_tpulang.test_all, ["bm1684x"]),
        }
        # yapf: enable
        self.test_set = {
            "op": self.run_op_test,
            "script": self.run_script_test,
            "model": self.run_model_test,
        }

        self.results = []
        self.time_cost = []
        self.logger = logging.getLogger()

    def add_result(self, test_name: str, status: bool, error_cases: list = []):
        self.results.append({
            "name": test_name,
            "status": Status.PASSED if status else Status.FAILED,
            "error_cases": error_cases
        })

    def run_regression_net(self, model_name, chip, finished_list):
        case_name = f"{model_name}_{chip}"
        # set the file for saving output stream
        log_filename = case_name + ".log"

        file_handler = logging.FileHandler(filename=log_filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

        print(f"======= run_models.py {model_name} {chip} {self.test_type}=====")
        dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model_name}_{chip}")
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        regressor = MODEL_RUN(model_name,
                              chip,
                              self.test_type,
                              save_log=True,
                              disable_thread=self.disable_thread)
        ret = regressor.run_full()
        finished_list.append({
            "name": case_name,
            "status": Status.PASSED if ret == 0 else Status.FAILED,
            "error_cases": []
        })
        os.chdir(self.current_dir)

        self.logger.removeHandler(file_handler)
        file_handler.close()
        return ret == 0

    def _run_op_test(self, op_source, tester, test_all_func, chip):
        print(f"======= test_{op_source}.py ======")
        case_name = f"{op_source}_test_{chip}"
        os.makedirs(case_name, exist_ok=True)
        os.chdir(dir)
        if op_source == "tflite" or op_source == "tpulang":
            tester = tester(chip=chip)
        else:
            tester = tester(chip=chip, simple=self.is_basic)
        error_cases = test_all_func(tester)
        self.add_result(case_name, not error_cases, error_cases)
        os.chdir(self.current_dir)

        return not error_cases

    def run_script_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= script test ======")
        case_name = "script_test"

        file_handler = logging.FileHandler(filename=case_name + ".log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

        success = True
        try:
            _os_system_log(
                os.path.expandvars("$REGRESSION_PATH/script_test/run.sh {}".format(self.test_type)))
        except:
            success = False

        self.add_result(case_name, success)

        self.logger.removeHandler(file_handler)
        file_handler.close()
        self.time_cost.append(f"run_script: {int(t.elapsed_time())} seconds")
        return not success

    def run_op_test(self):
        for op_source in self.op_test_types.keys():
            t = Timer()
            tester, test_func, chips = self.op_test_types[op_source]
            for chip in chips:
                success = self._run_op_test(op_source, tester, test_func, chip)
                # basic test stops once a test failed
                if not success and self.is_basic:
                    return 1
            self.time_cost.append(f"run_{op_source}: {int(t.elapsed_time())} seconds")

    def run_model_test(self):
        model_list = basic_model_list if self.is_basic else full_model_list
        for idx, chip in enumerate(chip_support.keys()):
            t = Timer()
            cur_model_list = [
                model_name for model_name, do_test in model_list.items() if do_test[idx]
            ]
            if self.disable_thread:
                finished_list = list()
                for model in cur_model_list:
                    self.run_regression_net(model, chip, finished_list)
                self.results.extend(finished_list)
                if self.is_basic:
                    for result in finished_list:
                        if result["status"] != Status.PASSED:
                            return 1
            else:
                import multiprocessing
                from utils.misc import collect_process
                process_number = multiprocessing.cpu_count() // 2 + 1
                processes = []
                finished_list = multiprocessing.Manager().list()
                error_cases = multiprocessing.Manager().list()
                for model in cur_model_list:
                    name = f"{model}_{chip}"
                    p = multiprocessing.Process(target=self.run_regression_net,
                                                name=name,
                                                args=(model, chip, finished_list))
                    processes.append(p)
                    if len(processes) == process_number:
                        collect_process(processes, error_cases, 1200)
                        processes = []
                collect_process(processes, error_cases, 1200)
                processes = []
                for error in error_cases:
                    if error not in finished_list:
                        finished_list.append({
                            "name": error,
                            "status": Status.TIMEOUT,
                            "error_cases": []
                        })
                self.results.extend(finished_list)
                if self.is_basic:
                    for result in finished_list:
                        if result["status"] != Status.PASSED:
                            return 1
            self.time_cost.append(f"run models for {chip}: {int(t.elapsed_time())} seconds")


    def run_all(self, test_set):
        t = Timer()

        for test in test_set:
            if self.test_set[test]() and self.is_basic:
                return 1

        self.time_cost.append(f"total time: {int(t.elapsed_time())} seconds")
        return 1 if any(result.get("status") != Status.PASSED for result in self.results) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=['all', 'basic'],
                        help="whether do all model test, 'all' runs all modes, 'baisc' runs basic models f16 and int8 sym only")
    choices = ["op", "script", "model"]
    parser.add_argument("--test_set", default=choices, type=str.lower, nargs="+", choices=choices,
                        help="run test set individually.")
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    # yapf: enable
    args = parser.parse_args()

    dir = os.path.expandvars("${REGRESSION_PATH}/regression_out")
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(message)s')

    main_entry = MAIN_ENTRY(args.test_type, args.disable_thread)

    exit_status = main_entry.run_all(args.test_set)

    # print last 100 lines log of failed cases first
    for result in main_entry.results:
        if result["name"].startswith(tuple(main_entry.op_test_types.keys())):
            continue
        if result["status"] != Status.PASSED:
            with open(result["name"] + ".log", 'r') as file:
                lines = file.readlines()
                last_100_lines = lines[-100:]
                content = ''.join(last_100_lines)
                print(content)

    print("============ Time Consum ============")
    for time in main_entry.time_cost:
        print(time)

    print("============ Passed Cases ============")
    for result in main_entry.results:
        if result["status"] == Status.PASSED:
            print("{} {}".format(result["name"], result["status"]))

    print("============ Failed Cases ============")
    for result in main_entry.results:
        if result["status"] != Status.PASSED:
            print("{} {}".format(result["name"], result["status"]))
            if result["error_cases"]:
                print("Failed cases: ", result["error_cases"])

    print("TEST {} {}".format(args.test_type, "PASSED" if not exit_status else "FAILED"))
    exit(exit_status)
