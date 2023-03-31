from ast import arg
from inspect import getmodule
from readline import get_completer_delims
import subprocess
import os
import shutil

import argparse
import torch
from torch import Tensor
from unet_model import UNet
import re


# https://tpumlir.org/docs/quick_start/03_onnx.html
def get_model(height, width) -> None:
    if check_file(f"../model/float_scale_0.5_{height}_{width}.onnx"):
        return
    inputs = torch.randn(1, 3, height, width)
    net = UNet(3, 2)
    net.load_state_dict(torch.load("../model/unet_carvana_scale0.5_epoch2.pth", map_location="cpu"))
    torch.onnx.export(
        net,
        inputs,
        f"../model/float_scale_0.5_{height}_{width}.onnx",
        verbose=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )


def transform(height, width):
    if check_file(f"../model/float_scale_0.5_{height}_{width}.mlir"):
        return
    splog = subprocess.check_output(
        [
            f"model_transform.py \
                --model_name unet_{height}_{width} \
                --model_def ../model/float_scale_0.5_{height}_{width}.onnx \
                --input_shapes [[1,3,{height},{width}]] \
                --mean 0.0,0.0,0.0 \
                --scale 0.0039216,0.0039216,0.0039216 \
                --keep_aspect_ratio \
                --pixel_format rgb \
                --test_input ../test_hq/0cdf5b5d0ce1_01.jpg \
                --test_result ./top_result_{height}_{width}.npz \
                --mlir ../model/float_scale_0.5_{height}_{width}.mlir"
        ],
        shell=True,
    )


def calib(height, width):
    if check_file(f"./float_scale_0.5_{height}_{width}_calib.table"):
        return
    splog = subprocess.check_output(
        [
            f"run_calibration.py ../model/float_scale_0.5_{height}_{width}.mlir \
                --dataset ../test_hq \
                --input_num  0 \
                --tune_num 200 \
                --histogram_bin_num 2048 \
                -o ./unet_scale0.5_cali_table"
        ],
        shell=True,
    )


def quantize_symmetric(height, width, tolerance=[0.84, 0.45]):
    if check_file(f"../model/unet_0.5_{height}_{width}_sym.bmodel") and check_file(
            f"../model/unet_0.5_{height}_{width}_sym.bmodel"):
        return
    splog = subprocess.check_output(
        [
            f"model_deploy.py \
                --mlir ../model/float_scale_0.5_{height}_{width}.mlir \
                --quantize INT8 \
                --calibration_table ./unet_scale0.5_cali_table \
                --chip bm1684x \
                --tolerance {tolerance[0]},{tolerance[1]} \
                --model ../model/unet_0.5_{height}_{width}_sym.bmodel"
        ],
        shell=True,
    )
    # --test_input ./unet_{height}_{width}_in_f32.npz \
    # --test_reference  ./top_result_{height}_{width}.npz \


def run(height, width, sym_type="sym"):
    check_dir(f"../result")
    # if already run, skip
    if 200 == len(os.listdir(f"../result")):
        return
    splog = subprocess.check_output(
        [
            f"python3 \
                mlir_tester.py \
                --model ./unet_{height}_{width}_bm1684x_int8_{sym_type}_tpu.mlir \
                --img_dir ../test_hq \
                --out_dir ../result  \
                --height {height}  \
                --width {width}  "
        ],
        shell=True,
    )
    shutil.copy(
        f"../model/unet_0.5_200_400_sym.bmodel.compiler_profile_0.txt",
        f"./profile.txt",
    )


def parse(height, width):
    with open(f"./profile.txt") as f1:
        text1 = f1.read()
        usage = re.findall(r"DDR BW USAGE : (.+?)\n", text1)
        flops = re.findall(r"flops: (.+?), ", text1)
        runtime = re.findall(r"runtime: (.+?), ", text1)
        ca = re.findall(r"ComputationAbility: (.+?)\n", text1)
        rec1 = [usage[0], int(flops[0]) / 10**12, runtime[0], ca[0]]

        return [[rec1[index], "-"] for index in range(len(rec1))]


def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def check_file(path):
    return os.path.exists(path)


if __name__ == "__main__":
    DATA_PATH = "../test_hq"
    tolerance = [0.99, 0.85]  # for quant
    RESOLUTION_PAIR = [  # resolution for resize, 64x
        (200, 400),
    ]
    for height, width in iter(RESOLUTION_PAIR):
        print(f">>>>>>>>>>>>>>>>>>>[  {height}   {width}  ]>>>>>>>>>>>>>>>>>>>")

        # step 1, get model
        get_model(height, width)
        print(f"::::::[  {height}   {width}  ] get_model")

        # step 2, model transform
        transform(height, width)
        print(f"::::::[  {height}   {width}  ] transform")

        # skip calib
        # calib(height, width)
        print(f"::::::[  {height}   {width}  ] calib")

        quantize_symmetric(height, width, tolerance)
        print(f"::::::[  {height}   {width}  ] quantize_symmetric")

        # quantize_asymmetric(height, width, tolerance)
        # log_print(f"::::::[  {height}   {width}  ] quantize_asymmetric")

        run(height, width, sym_type="sym")
        # run(height, width, sym_type="asym")

        print(f"<<<<<<<<<<<<<<<<<<<[  {height}   {width}  ]<<<<<<<<<<<<<<<<<<<")
        usage, tflops, runtime, ca = parse(height, width)
        print(runtime)
