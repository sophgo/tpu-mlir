#!/usr/bin/env python3
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
import multiprocessing
import time
from chip import *
from run_model import MODEL_RUN
import test_tpulang
import test_torch
import test_tflite
import test_onnx
import argparse


class MAIN_ENTRY(object):

    def __init__(self, test_type, disable_thread:bool):
        self.test_type = test_type
        self.disable_thread = disable_thread
        self.current_dir = os.getcwd()
        self.is_basic = test_type == "basic"
        # yapf: disable
        self.op_test_types = {
            # op_source: (tester, test_all_func, chips)
            "onnx_int4": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_int4,  ["bm1686"]),
            "onnx":     (test_onnx.ONNX_IR_TESTER,       test_onnx.test_all,   ["bm1684", "bm1684x", "bm1686", "cv183x"]),
            "tflite":   (test_tflite.TFLITE_IR_TESTER,   test_tflite.test_all, ["bm1684x"]),
            "torch":    (test_torch.TORCH_IR_TESTER,     test_torch.test_all,  ["bm1684x", "bm1686", "cv183x"]),
            "tpulang":  (test_tpulang.TPULANG_IR_TESTER, test_tpulang.test_all, ["bm1684x"]),
        }
        # yapf: enable
        self.results = []
        self.time_cost = []

    def add_result(self, test_name: str, status: bool, error_cases: list = []):
        self.results.append({
            "name": test_name,
            "status": "PASSED" if status else "FAILED",
            "error_cases": error_cases
        })

    def run_regression_net(self, model_name, chip, finished_list):
        print(f"======= run_models.py {model_name} {chip} {self.test_type}=====")
        dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model_name}_{chip}")
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        regressor = MODEL_RUN(model_name, chip, self.test_type, disable_thread=self.disable_thread)
        ret = regressor.run_full()
        finished_list.append({
            "name": f"Test {model_name} {chip}",
            "status": "PASSED" if ret == 0 else "FAILED",
            "error_cases": []
        })
        os.chdir(self.current_dir)
        return ret == 0

    def run_op_test(self, op_source, tester, test_all_func, chip):
        print(f"======= test_{op_source}.py ======")
        dir = f"{op_source}_test_{chip}"
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        if op_source == "tflite" or op_source == "tpulang":
            tester = tester(chip=chip)
        else:
            tester = tester(chip=chip, simple=self.is_basic)
        error_cases = test_all_func(tester)
        self.add_result(f"test_{op_source}.py {chip}", not error_cases, error_cases)
        os.chdir(self.current_dir)
        return not error_cases

    def run_script_test(self):
        # run scripts under $REGRESSION_OUT/script_test
        print("======= script test ======")
        ret = os.system(os.path.expandvars("$REGRESSION_PATH/script_test/run.sh"))
        self.add_result("Script Test", ret == 0)
        return ret == 0

    def run_all(self):
        start_time = time.time()
        tmp_time = end_time = start_time

        # test op from different sources
        for op_source in self.op_test_types.keys():
            tester, test_func, chips = self.op_test_types[op_source]
            for chip in chips:
                success = self.run_op_test(op_source, tester, test_func, chip)
                # basic test stops once a test failed
                if not success and self.is_basic:
                    return 1
            end_time = time.time()
            self.time_cost.append(f"run_{op_source}: {int(end_time - tmp_time)} seconds")
            tmp_time = end_time

        # test script
        success = self.run_script_test()
        if not success and self.is_basic:
            return 1
        end_time = time.time()
        self.time_cost.append(f"run_script: {int(end_time - tmp_time)} seconds")
        tmp_time = end_time

        # test model regression
        model_list = basic_model_list if self.is_basic else full_model_list
        for idx, chip in enumerate(chip_support.keys()):
            cur_model_list = [
                model_name for model_name, do_test in model_list.items() if do_test[idx]
            ]
            if self.disable_thread:
                finished_list = list()
                for model in cur_model_list:
                    self.run_regression_net(model, chip, finished_list)
                self.results.extend(finished_list)
                for result in finished_list:
                    if result["status"] == "FAILED" and self.is_basic:
                        return 1
            else:
                process_number = multiprocessing.cpu_count() // 2 + 1
                processes = []
                finished_list = multiprocessing.Manager().list()
                for model in cur_model_list:
                    p = multiprocessing.Process(target=self.run_regression_net,
                                                args=(model, chip, finished_list))
                    processes.append(p)
                    if len(processes) == process_number:
                        for p in processes:
                            p.start()
                        for j in processes:
                            j.join()
                        processes = []
                if processes:
                    for p in processes:
                        p.start()
                    for j in processes:
                        j.join()

                self.results.extend(finished_list)
                for result in finished_list:
                    if result["status"] == "FAILED" and self.is_basic:
                        return 1

            end_time = time.time()
            self.time_cost.append(f"run models for {chip}: {int(end_time - tmp_time)} seconds")
            tmp_time = end_time
        self.time_cost.append(f"total time: {int(end_time - start_time)} seconds")

        return 1 if any(result.get("status") == "FAILED" for result in self.results) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=['all', 'basic'],
                        help="whether do all model test, 'all' runs all modes, 'baisc' runs basic models f16 and int8 sym only")
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    # yapf: enable
    args = parser.parse_args()

    dir = os.path.expandvars("${REGRESSION_PATH}/regression_out")
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)

    main_entry = MAIN_ENTRY(args.test_type, args.disable_thread)
    exit_status = main_entry.run_all()
    for time in main_entry.time_cost:
        print(time)
    for result in main_entry.results:
        print("{} {}".format(result["name"], result["status"]))
        if result["error_cases"]:
            print("Failed cases: ", result["error_cases"])
    print("TEST {} {}".format(args.test_type, "PASSED" if not exit_status else "FAILED"))
    exit(exit_status)
