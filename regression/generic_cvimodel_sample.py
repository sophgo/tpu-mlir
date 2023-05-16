#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import os
import shutil
import time
from run_model import MODEL_RUN
from utils.mlir_shell import _os_system


class CviSampleGenerator:

    def __init__(self, dest_dir):
        self.chips = ["cv180x", "cv181x", "cv182x", "cv183x"]
        #self.chips = ["cv183x"]
        self.current_dir = os.getcwd()
        self.dest_dir = dest_dir
        if os.path.exists(self.dest_dir):
            shutil.rmtree(self.dest_dir)
        os.makedirs(self.dest_dir, exist_ok=True)

    def move_model(self, dir, dst_model):
        files = os.listdir(dir)
        for file in files:
            if file.split(".")[-1] == "cvimodel":
                shutil.move(os.path.join(dir, file), dst_model)
                return

    def cmd_exec(self, cmd_str):
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))

    def run_sample_net(self,
                       model,
                       dst_model,
                       chip,
                       quant_type,
                       customization_format="",
                       fuse_pre=False,
                       aligned_input=False,
                       merge_weight=False):
        info = f"{chip} {model} {quant_type} fuse_preprocess:{fuse_pre} {customization_format} aligned_input:{aligned_input} merge_weight:{merge_weight}"
        print(f"run_model.py {info}")
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
                              aligned_input=aligned_input)
        ret = regressor.run_full()
        if ret == 0:
            print(f"{info} Success.")
            self.move_model(dir, dst_model)
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

        if chip != "cv181x" and chip != "cv180x":
            self.run_sample_net("yolov3_416_with_det", os.path.join(tmp_dir, "yolo_v3_416_fused_preprocess_with_detection.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)

        if chip != "cv180x":
            self.run_sample_net("yolox_s", os.path.join(tmp_dir, "yolox_s.cvimodel"), chip, "int8_sym")
            self.run_sample_net("yolov5s_with_trans", os.path.join(tmp_dir, "yolov5s_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("yolov5s_with_trans", os.path.join(tmp_dir, "yolov5s_fused_preprocess_aligned_input.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True, True)
            self.run_sample_net("alphapose_res50", os.path.join(tmp_dir, "alphapose_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("arcface_res50", os.path.join(tmp_dir, "arcface_res50_fused_preprocess.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True)
            self.run_sample_net("arcface_res50", os.path.join(tmp_dir, "arcface_res50_fused_preprocess_aligned_input.cvimodel"), chip, "int8_sym", "RGB_PLANAR", True, True)
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
        self.cmd_exec(merge_cmd)
        os.remove(full_bs1_model)
        os.remove(full_bs4_model)
        os.chdir(os.path.expandvars("$REGRESSION_PATH/regression_out"))
        des_gz = os.path.join(self.dest_dir, f"cvimodel_samples_{chip}.tar.gz")
        tar_cmd = f"tar zcvf {des_gz} cvimodel_samples"
        self.cmd_exec(tar_cmd)
        os.chdir(self.current_dir)
        shutil.rmtree(tmp_dir)

    def generate_models(self):
        for chip in self.chips:
            self.generate_chip_models(chip)


if __name__ == "__main__":
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir', metavar='dest_dir', help='Generate cvimodel samples path')
    args = parser.parse_args()
    generator = CviSampleGenerator(args.dest_dir)
    generator.generate_models()
    t2 = time.time()
    print("Start time:", time.asctime(time.localtime(t1)))
    print("Finish time:", time.asctime(time.localtime(t2)))
    print("Time cost(s):", t2 - t1)
