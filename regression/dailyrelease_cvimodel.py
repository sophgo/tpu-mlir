#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import datetime
import logging
import os
import shutil
import time

from main_entry import Status
from run_model import MODEL_RUN
from chip import *

def move_model(dir, dst_model):
    files = os.listdir(dir)
    for file in files:
        if file.split(".")[-1] == "cvimodel":
            shutil.move(os.path.join(dir, file), dst_model)
            return

def cmd_exec(cmd_str):
    print("[Running]: {}".format(cmd_str))
    ret = os.system(cmd_str)
    if ret == 0:
        print("[Success]: {}".format(cmd_str))
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))

logger = logging.getLogger()

class CviModelRegressor:
    def __init__(self, dest_dir, save_interval):
        self.disable_thread = False
        self.current_dir = os.getcwd()
        self.dest_dir = dest_dir
        os.makedirs(self.dest_dir, exist_ok=True)
        self.chips = ["cv180x", "cv181x", "cv182x", "cv183x"]
        # self.chips = ["cv183x"]
        self.model_list = full_model_list
        self.test_type = "all"
        self.results = []
        #get cv182x chip idx,since cv182x and cv181x have the same number NPU, in chip.py all cv181x models are set as "N"
        #to store cv181x regression models, cv181x model_list should use cv182x's
        self.cv182x_idx = -1
        for idx, chip in enumerate(chip_support.keys()):
            if chip == "cv182x":
                self.cv182x_idx = idx
                break
        #whether to store models
        day = int(datetime.datetime.now().strftime('%d'))
        self.save_interval = save_interval
        self.store_model = True if day % self.save_interval == 0 else False

    def run_regression_net(self, model_name, chip, finished_list):
        case_name = f"{model_name}_{chip}"
        # set the file for saving output stream
        log_filename = case_name + ".log"

        file_handler = logging.FileHandler(filename=log_filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

        print(f"======= run_regression_net {model_name} {chip} {self.test_type}=====")
        dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model_name}_{chip}")
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        regressor = MODEL_RUN(model_name,
                              chip,
                              self.test_type,
                              save_log = True,
                              disable_thread = self.disable_thread,
                              debug = True)
        ret = regressor.run_full()
        finished_list.append({
            "name": case_name,
            "status": Status.PASSED if ret == 0 else Status.FAILED,
            "error_cases": []
        })
        os.chdir(self.current_dir)

        logger.removeHandler(file_handler)
        file_handler.close()
        return ret == 0

    def run_nets(self):
        # test cv18xx model regression
        for idx, chip in enumerate(chip_support.keys()):
            if chip not in self.chips:
                continue
            if self.store_model and chip == "cv181x":
                idx = self.cv182x_idx
            cur_model_list = [
                model_name for model_name, do_test in self.model_list.items() if do_test[idx]
            ]
            if self.disable_thread:
                finished_list = list()
                for model in cur_model_list:
                    self.run_regression_net(model, chip, finished_list)
                self.results.extend(finished_list)
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
                        collect_process(processes, error_cases, 2000)
                        processes = []
                collect_process(processes, error_cases, 2000)
                processes = []
                for error in error_cases:
                    if error not in finished_list:
                        finished_list.append({
                            "name": error,
                            "status": Status.TIMEOUT,
                            "error_cases": []
                        })
                self.results.extend(finished_list)

    def pack_models(self):
        for idx, chip in enumerate(chip_support.keys()):
            if chip not in self.chips:
                continue
            if self.store_model and chip == "cv181x":
                idx = self.cv182x_idx


            temp_save_dir_bf16 = os.path.expandvars(f"$REGRESSION_PATH/regression_out/cvimodel_regression_bf16_{chip}")
            temp_save_dir_int8 = os.path.expandvars(f"$REGRESSION_PATH/regression_out/cvimodel_regression_int8_{chip}")
            os.makedirs(temp_save_dir_int8, exist_ok=True)
            store_bf16 = False
            if chip == "cv182x" or chip == "cv183x":
                os.makedirs(temp_save_dir_bf16, exist_ok=True)
                store_bf16 = True
            cur_model_list = [
                model_name for model_name, do_test in self.model_list.items() if do_test[idx]
            ]
            for model in cur_model_list:
                reg_dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model}_{chip}")
                if not os.path.exists(reg_dir):
                    raise RuntimeError("[!Error]: {}_{} not regression.".format(model, chip))
                files = os.listdir(reg_dir)
                for file in files:
                    if file.split(".")[-1] == "cvimodel":
                        quant_type = file.split(f"_{chip}_")[-1].split(".cvimodel")[0]
                        if quant_type == "int8_sym":
                            shutil.copy(os.path.join(reg_dir, file), os.path.join(temp_save_dir_int8, f"{model}_bs1.cvimodel"))
                        elif quant_type == "bf16":
                            if store_bf16:
                                shutil.copy(os.path.join(reg_dir, file), os.path.join(temp_save_dir_bf16, f"{model}_bs1.cvimodel"))
                        else:
                            assert(0 and f"unknown quant_type:{quant_type}")
                #copy input
                input_npz = f"{model}_in_f32.npz"
                if (os.path.exists(os.path.join(reg_dir, input_npz))):
                    shutil.copy(os.path.join(reg_dir, input_npz), os.path.join(temp_save_dir_int8, input_npz))
                    if store_bf16:
                        shutil.copy(os.path.join(reg_dir, input_npz), os.path.join(temp_save_dir_bf16, input_npz))
                #copy output
                output_int8_npz = f"{model}_{chip}_int8_sym_model_outputs.npz"
                if (os.path.exists(os.path.join(reg_dir, output_int8_npz))):
                    shutil.copy(os.path.join(reg_dir, output_int8_npz), os.path.join(temp_save_dir_int8, f"{model}_bs1_out_all.npz"))
                if store_bf16:
                    output_bf16_npz = f"{model}_{chip}_bf16_model_outputs.npz"
                    if (os.path.exists(os.path.join(reg_dir, output_bf16_npz))):
                        shutil.copy(os.path.join(reg_dir, output_bf16_npz), os.path.join(temp_save_dir_bf16, f"{model}_bs1_out_all.npz"))
            des_gz_int8 = os.path.join(self.dest_dir, f"cvimodel_regression_int8_{chip}.tar.gz")
            tar_cmd_int8 = f"tar zcvf {des_gz_int8} cvimodel_regression_int8_{chip}"
            cmd_exec(tar_cmd_int8)
            shutil.rmtree(temp_save_dir_int8)
            if store_bf16:
                des_gz_bf16 = os.path.join(self.dest_dir, f"cvimodel_regression_bf16_{chip}.tar.gz")
                tar_cmd_bf16 = f"tar zcvf {des_gz_bf16} cvimodel_regression_bf16_{chip}"
                cmd_exec(tar_cmd_bf16)
                shutil.rmtree(temp_save_dir_bf16)

    def run(self):
        self.run_nets()
        if self.store_model:
            self.pack_models()

class CviSampleGenerator:

    def __init__(self, dest_dir):
        self.chips = ["cv180x", "cv181x", "cv182x", "cv183x"]
        # self.chips = ["cv182x"]
        self.current_dir = os.getcwd()
        self.dest_dir = dest_dir
        os.makedirs(self.dest_dir, exist_ok=True)

    def run_sample_net(self,
                       model,
                       dst_model,
                       chip,
                       quant_type,
                       customization_format="",
                       fuse_pre=False,
                       aligned_input=False,
                       merge_weight=False):
        info = f"{chip}_{model}_{quant_type}_fuse_preprocess:{fuse_pre}_{customization_format}_aligned_input:{aligned_input}_merge_weight:{merge_weight}"
        print(f"run_sample_net {info}")
        #set the file for saving output stream
        log_filename = f"sample_{chip}_{model}.log"
        file_handler = logging.FileHandler(filename = log_filename, mode = "w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/{model}_{chip}")
        os.makedirs(dir, exist_ok=True)
        os.chdir(dir)
        regressor = MODEL_RUN(model,
                              chip,
                              quant_type,
                              dyn_mode=False,
                              merge_weight=merge_weight,
                              fuse_preprocess=fuse_pre,
                              customization_format=customization_format,
                              aligned_input=aligned_input,
                              save_log=True)
        ret = regressor.run_full()
        logger.removeHandler(file_handler)
        file_handler.close()
        if ret == 0:
            move_model(dir, dst_model)
            os.chdir(self.current_dir)
            shutil.rmtree(dir)
        else:
            raise RuntimeError("[!Error]: {}".format(f"{info} Failed."))


    def generate_chip_models(self, chip):
        tmp_dir = os.path.expandvars(f"$REGRESSION_PATH/regression_out/cvimodel_samples")
        os.makedirs(tmp_dir, exist_ok=True)
        # yapf: disable
        self.run_sample_net("mobilenet_v2_cf", os.path.join(tmp_dir, "mobilenet_v2.cvimodel"), chip, "int8_sym")
        self.run_sample_net("mobilenet_v2_cf", os.path.join(tmp_dir, "mobilenet_v2_bf16.cvimodel"), chip, "bf16")
        self.run_sample_net("mobilenet_v2_cf", os.path.join(tmp_dir, "mobilenet_v2_fused_preprocess.cvimodel"), chip, "int8_sym", "BGR_PLANAR", True)
        self.run_sample_net("mobilenet_v2_cf", os.path.join(tmp_dir, "mobilenet_v2_int8_yuv420.cvimodel"), chip, "int8_sym", "BGR_PLANAR", True, True)
        self.run_sample_net("retinaface_mnet_with_det", os.path.join(tmp_dir, "retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
        self.run_sample_net("retinaface_mnet_with_det", os.path.join(tmp_dir, "retinaface_mnet25_600_fused_preprocess_aligned_input.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True, True)
        self.run_sample_net("yolov8n", os.path.join(tmp_dir, "yolov8n_int8.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)

        if chip != "cv181x" and chip != "cv180x":
            self.run_sample_net("yolov3_416_with_det", os.path.join(tmp_dir, "yolo_v3_416_fused_preprocess_with_detection.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)

        if chip != "cv180x":
            self.run_sample_net("yolox_s", os.path.join(tmp_dir, "yolox_s.cvimodel"), chip, "int8_sym")
            self.run_sample_net("yolov5s", os.path.join(tmp_dir, "yolov5s_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("yolov5s", os.path.join(tmp_dir, "yolov5s_fused_preprocess_aligned_input.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True, True)
            self.run_sample_net("alphapose_res50", os.path.join(tmp_dir, "alphapose_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("arcface_res50", os.path.join(tmp_dir, "arcface_res50_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("arcface_res50", os.path.join(tmp_dir, "arcface_res50_fused_preprocess_aligned_input.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True, True)
            self.run_sample_net("pp_yoloe_m", os.path.join(tmp_dir, "pp_yoloe_m_int8.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
        # yapf: enable
        #gen merged cvimodel
        full_bs1_model = os.path.join(tmp_dir, "mobilenet_v2_bs1.cvimodel")
        full_bs4_model = os.path.join(tmp_dir, "mobilenet_v2_bs4.cvimodel")
        self.run_sample_net("mobilenet_v2_cf", full_bs1_model, chip, "int8_sym", "", False, False,
                            True)
        self.run_sample_net("mobilenet_v2_cf_bs4", full_bs4_model, chip, "int8_sym", "", False,
                            False, True)
        merge_model = os.path.join(tmp_dir, "mobilenet_v2_bs1_bs4.cvimodel")
        merge_cmd = f"model_tool --combine {full_bs1_model} {full_bs4_model} -o {merge_model}"
        cmd_exec(merge_cmd)
        os.remove(full_bs1_model)
        os.remove(full_bs4_model)
        os.chdir(os.path.expandvars("$REGRESSION_PATH/regression_out"))
        des_gz = os.path.join(self.dest_dir, f"cvimodel_samples_{chip}.tar.gz")
        tar_cmd = f"tar zcvf {des_gz} cvimodel_samples"
        cmd_exec(tar_cmd)
        os.chdir(self.current_dir)
        shutil.rmtree(tmp_dir)

    def generate_models(self):
        for chip in self.chips:
            self.generate_chip_models(chip)
        shutil.rmtree(os.path.expandvars(f"$REGRESSION_PATH/regression_out"))


if __name__ == "__main__":
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir', metavar='dest_dir', help='Generate cvimodel samples path')
    parser.add_argument('save_interval', type=int, help='how many days to save regression models')
    args = parser.parse_args()
    dir = os.path.expandvars("${REGRESSION_PATH}/regression_out")
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    logging.basicConfig(filename="main.log",
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(message)s')
    regressor = CviModelRegressor(args.dest_dir, args.save_interval)
    regressor.run()
    print("============ Regression Passed Cases ============")
    for result in regressor.results:
        if result["status"] == Status.PASSED:
            print("{} {}".format(result["name"], result["status"]))
    print("============ Regression Failed Cases ============")
    count_err = 0
    for result in regressor.results:
        if result["status"] != Status.PASSED:
            count_err += 1
            print("{} {}".format(result["name"], result["status"]))
            if result["error_cases"]:
                print("Failed cases: ", result["error_cases"])
    t2 = time.time()
    print("Regression time cost(s):", t2 - t1)
    if count_err != 0:
        assert(0 and "Regression Failed")
    #clear
    shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    generator = CviSampleGenerator(args.dest_dir)
    generator.generate_models()
    t3 = time.time()
    print("Generate cvimodel samples cost(s):", t3 - t2)
    print("Start time:", time.asctime(time.localtime(t1)))
    print("Finish time:", time.asctime(time.localtime(t3)))
