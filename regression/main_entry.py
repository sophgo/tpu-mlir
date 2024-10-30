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
test_custom_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'customlayer', 'test'))
sys.path.append(test_custom_dir)

from utils.timer import Timer
from chip import *
from run_model import MODEL_RUN
import test_tpulang
import test_torch
import test_tflite
import test_onnx
import test_custom_tpulang
import argparse
import logging
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

    def __init__(self, test_type, disable_thread: bool):
        self.test_type = test_type
        self.disable_thread = disable_thread
        self.current_dir = os.getcwd()
        self.is_basic = test_type == "basic"
        # yapf: disable
        self.op0_test_types = {
            # op_source: (tester, test_all_func, chips)
            "onnx_int4": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_int4,  ["bm1688"]),
            "onnx_fp8": (test_onnx.ONNX_IR_TESTER,  test_onnx.test_fp8,  ["bm1690"]),
            "onnx":     (test_onnx.ONNX_IR_TESTER,       test_onnx.test_all,   ["bm1684", "bm1684x", "bm1688", "cv183x", "mars3"]),
            "tflite":   (test_tflite.TFLITE_IR_TESTER,   test_tflite.test_all, ["bm1684x", "bm1688"]),
        }
        self.op1_test_types = {
            # op_source: (tester, test_all_func, chips)
            "torch":    (test_torch.TORCH_IR_TESTER,     test_torch.test_all,  ["bm1684", "bm1684x", "bm1688", "cv183x", "mars3"]),
            "tpulang":  (test_tpulang.TPULANG_IR_TESTER, test_tpulang.test_all, ["bm1684x", "bm1688"]),
            "custom_tpulang":  (test_custom_tpulang.CUSTOM_TPULANG_TESTER, test_custom_tpulang.test_all, ["bm1684x", "bm1688"]),
        }
        if not self.is_basic:
            del self.op1_test_types["tpulang"]

        self.script_basic = ["test1", "test2","test5","test9","test11","test_llm0","test12",'test_modelzoo','test_encrypt']
        self.script_extend = ["test3","test4","test6","test7","test8","test10","test_llm1"]
        # yapf: enable
        self.test_set = {
            "op0": self.run_op0_test,
            "op1": self.run_op1_test,
            "script": self.run_script_test,
            "model": self.run_model_test,
            "multi_core_model": self.run_multi_core_test,
            "cuda": self.run_cuda_test,
            "maskrcnn": self.run_maskrcnn_test,
            "tdb": self.run_tdb_test,
        }

        self.results = []
        self.time_cost = []
        self.logger = logging.getLogger()

    def add_result(self, test_name: str, status: bool, time: int = 0, error_cases: list = []):
        self.results.append({
            "name": test_name,
            "status": Status.PASSED if status else Status.FAILED,
            "error_cases": error_cases,
            "time": time
        })

    def run_regression_net(self, model_name, chip, num_core, finished_list):
        case_name = f"{model_name}_{chip}_num_core_{num_core}"
        # set the file for saving output stream
        log_filename = case_name + ".log"

        file_handler = logging.FileHandler(filename=log_filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

        print(
            f"======= run_models.py {model_name} {chip} {self.test_type} num_core: {num_core} ====="
        )
        t = Timer()
        target_dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model_name}_{chip}")
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        regressor = MODEL_RUN(model_name,
                              chip,
                              self.test_type,
                              save_log=True,
                              num_core=num_core,
                              disable_thread=self.disable_thread)
        ret = regressor.run_full()
        finished_list.append({
            "name": case_name,
            "status": Status.PASSED if ret == 0 else Status.FAILED,
            "error_cases": [],
            "time":int(t.elapsed_time())
        })
        os.chdir(self.current_dir)
        shutil.rmtree(target_dir)
        self.logger.removeHandler(file_handler)
        file_handler.close()
        return ret == 0

    def _run_op_test(self, op_source, tester, test_all_func, chip):
        print(f"======= test_{op_source}.py ======")
        case_name = f"{op_source}_test_{chip}"
        t = Timer()
        os.makedirs(case_name, exist_ok=True)
        os.chdir(case_name)
        if op_source == "tflite":
            tester = tester(chip=chip)
        else:
            tester = tester(chip=chip, simple=self.is_basic)
        error_cases = test_all_func(tester)
        self.add_result(case_name, not error_cases, int(t.elapsed_time()), error_cases)
        os.chdir(self.current_dir)
        shutil.rmtree(case_name) # too large
        return not error_cases

    def _run_script_test(self, source):
        print(f"======= test script:{source}.sh ======")
        case_name = f"test_script_{source}"
        file_handler = logging.FileHandler(filename=case_name + ".log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        t = Timer()
        os.makedirs(case_name, exist_ok=True)
        os.chdir(case_name)

        success = True
        try:
            _os_system_log(os.path.expandvars("$REGRESSION_PATH/script_test/{}.sh".format(source)))
        except:
            success = False
        self.logger.removeHandler(file_handler)
        file_handler.close()
        self.add_result(case_name, success, int(t.elapsed_time()))
        os.chdir(self.current_dir)
        shutil.rmtree(case_name) # too large
        return success

    def run_script_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= script test ======")

        sources = self.script_basic
        if not self.is_basic:
            sources += self.script_extend
        ret = True
        for source in sources:
            ret = self._run_script_test(source)
            if not ret and self.is_basic:
                break
        self.time_cost.append(f"run_script: {int(t.elapsed_time())} seconds")
        return SUCCESS if ret else FAILURE

    def run_cuda_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= cuda test ======")
        ret = self._run_script_test("test_cuda")
        self.time_cost.append(f"run_cuda: {int(t.elapsed_time())} seconds")
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
            self.time_cost.append(f"run_{op_source}: {int(t.elapsed_time())} seconds")
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

            t = Timer()
            cur_model_list = [
                model_name for model_name, do_test in model_list.items() if do_test[idx]
            ]
            if self.disable_thread:
                finished_list = list()
                for model in cur_model_list:
                    self.run_regression_net(model, chip, num_core, finished_list)
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
                    name = f"{model}_{chip}_num_core_{num_core}"
                    p = multiprocessing.Process(target=self.run_regression_net,
                                                name=name,
                                                args=(model, chip, num_core, finished_list))
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
                            "error_cases": [],
                            "time": -1
                        })
                self.results.extend(finished_list)
                if self.is_basic:
                    for result in finished_list:
                        if result["status"] != Status.PASSED:
                            return 1
            self.time_cost.append(f"run models for {chip}: {int(t.elapsed_time())} seconds")

    def run_multi_core_test(self):
        self.run_model_test(multi_core=True)

    def run_maskrcnn_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= MaskRCNN test ======")
        ret = self._run_script_test("test_MaskRCNN")
        self.time_cost.append(f"run_MaskRCNN: {int(t.elapsed_time())} seconds")
        return SUCCESS if ret else FAILURE
    
    def run_tdb_test(self):
        # return exit status
        t = Timer()
        # run scripts under $REGRESSION_OUT/script_test
        print("======= TDB test ======")
        ret = self._run_script_test("test_tdb")
        self.time_cost.append(f"run_TDB: {int(t.elapsed_time())} seconds")
        return SUCCESS if ret else FAILURE

    def run_all(self, test_set):
        t = Timer()

        for test in test_set:
            if self.test_set[test]() and self.is_basic:
                return FAILURE

        self.time_cost.append(f"total time: {int(t.elapsed_time())} seconds")
        return FAILURE if any(result.get("status") != Status.PASSED
                              for result in self.results) else SUCCESS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--test_type", default="all", type=str.lower, choices=['all', 'basic'],
                        help="whether do all model test, 'all' runs all modes, 'baisc' runs basic models f16 and int8 sym only")
    choices = ["op0", "op1", "script", "model", "multi_core_model", "cuda", "maskrcnn", "tdb"]
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

    print("============ Time Consum ============")
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
