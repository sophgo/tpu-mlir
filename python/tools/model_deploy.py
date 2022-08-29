#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import abc
import numpy as np
import argparse
from utils.mlir_shell import *
from utils.mlir_parser import *
from tools.model_runner import mlir_inference, model_inference
import pymlir


def str2list(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    return files


def show_fake_cmd(in_npz: str, model: str, out_npz: str):
    print("[CMD]: model_runner.py --input {} --model {} --output {}".format(in_npz, model, out_npz))


class DeployTool:

    def __init__(self, args):
        self.mlir_file = args.mlir
        self.chip = args.chip.lower()
        self.excepts = args.excepts
        self.tolerance = args.tolerance
        self.correctness = args.correctness
        self.test_input = args.test_input
        self.quantize = args.quantize.lower()
        self.asymmetric = args.asymmetric
        self.cali_table = args.calibration_table
        self.quant_input = args.quant_input
        self.quant_output = args.quant_output
        self.quantize_table = args.quantize_table
        self.model = args.model
        self.test_input = args.test_input
        self.ref_npz = args.test_reference
        self.module = MlirParser(args.mlir)
        self.module_name = eval(self.module.attrs['module.name'])
        self.state = eval(self.module.attrs['module.state'])
        self.in_f32_npz = self.module_name + "_in_f32.npz"
        self.prefix = "{}_{}_{}".format(self.module_name, self.chip, self.quantize)
        if self.quantize == "int8":
            if self.asymmetric:
                self.prefix += "_asym"
            else:
                self.prefix += "_sym"
        self._prepare_input_npz()

    def lowering(self):
        self.tpu_mlir = "{}_tpu.mlir".format(self.prefix)
        self.final_mlir = "{}_final.mlir".format(self.prefix)
        mlir_lowering(self.mlir_file, self.tpu_mlir, self.quantize, self.chip, self.cali_table,
                      self.asymmetric, self.quantize_table)
        if self.do_validate:
            tool.validate_tpu_mlir()

    def _prepare_input_npz(self):
        num_inputs = len(self.test_input)
        self.do_validate = (0 < num_inputs)
        if not self.do_validate:
            return
        self.inputs = {}
        if num_inputs == 1 and self.test_input[0].endswith(".npz"):
            x = np.load(self.test_input[0])
            for name in x.files:
                self.inputs[name] = x[name]
        else:
            assert (len(self.test_input) == len(self.module.inputs))
            for infile, op in zip(self.test_input, self.module.inputs):
                assert (infile.endswith(".npy"))
                data = np.load(infile)
                self.inputs[op.name] = data
        np.savez(self.in_f32_npz, **self.inputs)
        if len(self.ref_npz) == 0:
            self.ref_npz = self.module_name + "_top_outputs.npz"
            show_fake_cmd(self.in_f32_npz, self.mlir_file, self.ref_npz)
            top_outputs = mlir_inference(self.inputs, self.mlir_file)
            np.savez(self.ref_npz, **top_outputs)
        self.tpu_npz = "{}_tpu_outputs.npz".format(self.prefix)

    def validate_tpu_mlir(self):
        show_fake_cmd(self.in_f32_npz, self.tpu_mlir, self.tpu_npz)
        tpu_outputs = mlir_inference(self.inputs, self.tpu_mlir)
        np.savez(self.tpu_npz, **tpu_outputs)
        # compare fp32 blobs and quantized tensors with tolerance similarity
        f32_blobs_compare(self.tpu_npz, self.ref_npz, self.tolerance, self.excepts)

    def build_model(self):
        mlir_to_model(
            self.tpu_mlir,
            self.model,
            self.final_mlir,
            self.quant_input,
            self.quant_output,
        )
        if self.do_validate:
            tool.validate_model()

    def validate_model(self):
        self.model_npz = "{}_model_outputs.npz".format(self.prefix)
        show_fake_cmd(self.in_f32_npz, self.model, self.model_npz)
        model_outputs = model_inference(self.inputs, self.model)
        np.savez(self.model_npz, **model_outputs)
        if self.state == "TOP_QUANTIZED":
            f32_blobs_compare(self.model_npz, self.ref_npz, self.correctness)
        else:
            f32_blobs_compare(self.model_npz, self.tpu_npz, self.correctness)


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True, help="optimized mlir fp32 model")
    parser.add_argument("--calibration_table", help="calibration table for int8 quantization")
    parser.add_argument("--quantize_table", help="table of OPs that quantized to specific mode")
    parser.add_argument("--quantize",
                        default="F32",
                        type=str,
                        choices=['F32', 'BF16', 'F16', 'INT8'],
                        help="set default qauntization type: F32/BF16/F16/INT8")
    parser.add_argument("--asymmetric",
                        action='store_true',
                        default=False,
                        help="for INT8 quantization")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--tolerance", default='0.8,0.5', help="tolerance")
    parser.add_argument("--correctness", default='0.99,0.90', help="correctness")
    parser.add_argument("--chip",
                        required=True,
                        type=str,
                        choices=['bm1684x', 'bm1684', 'cv183x', 'cv182x', 'mars'],
                        help="chip platform name")
    parser.add_argument("--test_input",
                        default="",
                        type=str2list,
                        help="input npy/npz file for inference, "
                        "if has more than one input, join npy with semicolon")
    parser.add_argument("--test_reference",
                        default="",
                        help="reference npz file; if none, will run inner")
    parser.add_argument("--model", required=True, help='output model')
    parser.add_argument("--quant_input",
                        action="store_true",
                        help="strip input type cast in bmodel, need outside type conversion")
    parser.add_argument("--quant_output",
                        action="store_true",
                        help="strip output type cast in bmodel, need outside type conversion")
    args = parser.parse_args()
    tool = DeployTool(args)
    # lowering to tpu
    tool.lowering()
    # generate model
    tool.build_model()
