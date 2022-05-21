#!/usr/bin/env python3
import abc
import numpy as np
import argparse
from utils.mlir_shell import *
from utils.mlir_parser import *
from tools.model_runner import mlir_inference, onnx_inference, tflite_inference
import pymlir


def str2list(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    return files


def check_return_value(cond, msg):
    if not cond:
        raise RuntimeError(msg)


class DeployTool:

    def __init__(self, args):
        self.mlir_file = args.mlir
        self.chip = args.chip.lower()
        self.excepts = args.excepts
        self.tolerance = args.tolerance
        self.correctness = args.correctness
        self.test_input = args.test_input
        self.test_ref = args.test_reference
        self.quantize = args.quantize.lower()
        self.asymmetric = args.asymmetric
        self.cali_table = args.calibration_table
        self.quantize_table = args.quantize_table
        self.model = args.model
        self.inpus_type = args.inputs_type
        self.outputs_type = args.outputs_type
        self.test_input = args.test_input
        self.ref_npz = args.test_reference
        self.module = MlirParser(args.mlir)
        self.module_name = eval(self.module.attrs['module.name'])
        self.state = eval(self.module.attrs['module.state'])
        self.in_fp32_npz = self.module_name + "_in_fp32.npz"
        self._prepare_input_npz()

    def lowering(self):
        self.tpu_mlir = "{}_{}_tpu_{}.mlir".format(self.module_name, self.chip, self.quantize)
        self.final_mlir = "{}_{}_{}_final.mlir".format(self.module_name, self.chip, self.quantize)
        ret = mlir_lowering(self.mlir_file, self.tpu_mlir, self.quantize, self.chip,
                            self.cali_table, self.asymmetric, self.quantize_table)
        check_return_value(ret == 0, 'lowering failed')
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
        np.savez(str(self.in_fp32_npz), **self.inputs)
        if len(self.ref_npz) == 0:
            self.ref_npz = self.module_name + "_top_outputs.npz"
            top_outputs = mlir_inference(self.inputs, self.mlir_file)
            np.savez(self.ref_npz, **top_outputs)
        self.tpu_npz = self.module_name + "_{}_{}_tpu_outputs.npz".format(self.chip, self.quantize)

    def validate_tpu_mlir(self):
        tpu_outputs = mlir_inference(self.inputs, self.tpu_mlir)
        np.savez(self.tpu_npz, **tpu_outputs)
        # compare fp32 blobs and quantized tensors with tolerance similarity
        ret = f32_blobs_compare(self.tpu_npz, self.ref_npz, self.tolerance, self.excepts)
        check_return_value(ret == 0, "accuracy validation of tpu mlir failed")

    def build_model(self):
        ret = mlir_to_model(self.tpu_mlir, self.model, self.final_mlir)
        check_return_value(ret == 0, "failed to generate model")
        if self.do_validate:
            tool.validate_model()

    def validate_model(self):
        print("TODO: run model by runner")

if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True, help="optimized mlir fp32 model")
    parser.add_argument("--calibration_table", help="calibration table for int8 quantization")
    parser.add_argument("--quantize_table", help="table of OPs that quantized to specific mode")
    parser.add_argument("--quantize", default='', help="set qauntization type: F32/BF16/F16/INT8")
    parser.add_argument("--asymmetric",
                        action='store_true',
                        default=False,
                        help="for INT8 quantization")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--tolerance", default='0.8,0.3', help="tolerance")
    parser.add_argument("--correctness", default='0.99,0.98', help="correctness")
    parser.add_argument("--chip",
                        required=True,
                        choices=['bm1686', 'bm1684', 'cv183x', 'cv182x', 'mars'],
                        help="chip platform name")
    parser.add_argument("--inputs_type",
                        default="F32",
                        choices=['AUTO', 'F32'],
                        help="set inputs type. if AUTO, use input layer quantize type")
    parser.add_argument("--outputs_type",
                        default="F32",
                        choices=['AUTO', 'F32'],
                        help="set outputs type. if AUTO, use output layer quantize type")
    parser.add_argument("--test_input",
                        default=None,
                        type=str2list,
                        help="input npy/npz file for inference, "
                        "if has more than one input, join npy with semicolon")
    parser.add_argument("--test_reference",
                        default=None,
                        help="reference npz file; if none, will run inner")
    parser.add_argument("--model", required=True, help='output model')
    args = parser.parse_args()
    tool = DeployTool(args)
    # lowering to tpu
    tool.lowering()
    # generate model
    tool.build_model()
