#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import numpy as np
import argparse
from utils.mlir_shell import *
from utils.mlir_parser import *
from utils.preprocess import preprocess, supported_customization_format
from utils.auto_remove import file_mark, file_clean
from tools.model_runner import mlir_inference, model_inference, show_fake_cmd
import pymlir
from utils.misc import str2bool


def str2list(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    return files


def getCustomFormat(pixel_format, channel_format):
    custom_format = ""
    if pixel_format == "rgb":
        if channel_format == "nchw":
            custom_format = "RGB_PLANAR"
        else:
            custom_format = "RGB_PACKED"
    elif pixel_format == "bgr":
        if channel_format == "nchw":
            custom_format = "BGR_PLANAR"
        else:
            custom_format = "BGR_PACKED"
    elif pixel_format == "gray":
        custom_format = "GRAYSCALE"
    elif pixel_format == "rgba":
        custom_format == "RGBA_PLANAR"
    else:
        print("pixel_format of {} no supported!".format(pixel_format))
        assert (0)
    return custom_format


class DeployTool:

    def __init__(self, args):
        self.mlir_file = args.mlir
        self.chip = args.chip
        self.includeWeight = args.includeWeight
        self.excepts = args.excepts
        self.tolerance = args.tolerance
        self.test_input = args.test_input
        self.linear_quant_mode = args.linear_quant_mode
        self.quantize = args.quantize.lower()
        self.asymmetric = args.asymmetric
        self.cali_table = args.calibration_table
        self.quant_input = args.quant_input
        self.quant_output = args.quant_output
        self.quantize_table = args.quantize_table
        self.embed_debug_info = args.debug
        self.model = args.model
        self.ref_npz = args.test_reference
        self.customization_format = args.customization_format
        self.fuse_preprocess = args.fuse_preprocess
        self.aligned_input = args.aligned_input
        self.do_winograd = args.do_winograd
        self.module = MlirParser(args.mlir)
        self.module_name = self.module.module_name
        self.state = self.module.module_state
        self.disable_layer_group = args.disable_layer_group
        self.merge_weight = args.merge_weight
        self.op_divide = args.op_divide
        self.ignore_f16_overflow = args.ignore_f16_overflow
        self.num_device = args.num_device
        self.num_core = args.num_core
        self.correctness = "0.99,0.90"
        if self.quantize_table:
            self.correctness = "0.99,0.85"
        self.in_f32_npz = self.module_name + "_in_f32.npz"
        self.prefix = "{}_{}_{}".format(self.module_name, self.chip, self.quantize)
        self.dynamic = args.dynamic
        self.compare_all = args.compare_all
        self.skip_validation = args.skip_validation
        self.model_version = args.model_version
        if self.quantize == "int8" or self.quantize == "int4":
            if self.asymmetric:
                self.prefix += "_asym"
            else:
                self.prefix += "_sym"
        self._prepare_input_npz()

    def cleanup(self):
        file_clean()

    def lowering(self):
        if self.chip == 'cpu':
            top_to_tosa(self.mlir_file, "tmp_tosa.mlir", self.includeWeight)
            # replace func name from "main" to "model"
            self.tosa_mlir = "{}_tosa.mlir".format(self.prefix)
            with open("tmp_tosa.mlir", "r", encoding="utf-8") as file:
                content = file.read()
            content = content.replace("main", "model")
            with open(self.tosa_mlir, "w", encoding="utf-8") as file:
                file.write(content)
            delete_file("tmp_tosa.mlir")
        else:
            self.tpu_mlir = "{}_tpu.mlir".format(self.prefix)
            file_mark(self.tpu_mlir)
            self.final_mlir = "{}_final.mlir".format(self.prefix)
            mlir_lowering(self.mlir_file, self.tpu_mlir, self.quantize, self.chip, self.cali_table,
                          self.asymmetric, self.quantize_table, self.customization_format,
                          self.fuse_preprocess, self.aligned_input, self.ignore_f16_overflow,
                          self.linear_quant_mode, self.do_winograd)
            if self.do_validate:
                tool.validate_tpu_mlir()

    def _prepare_input_npz(self):
        num_inputs = len(self.test_input)
        self.do_validate = (0 < num_inputs)
        if not self.do_validate:
            if self.customization_format == '':
                ppa = preprocess()
                input_op = self.module.inputs[0].op
                ppa.load_config(input_op)
                if ppa.has_pre:
                    self.customization_format = getCustomFormat(ppa.pixel_format,
                                                                ppa.channel_format)
            return
        self.inputs = {}
        gen_input_f32 = {}
        gen_ref = True if len(self.ref_npz) == 0 else False

        if self.fuse_preprocess:
            assert (self.test_input[0].endswith(('.jpg', '.jpeg', '.png')))
        if num_inputs == 1 and self.test_input[0].endswith(".npz"):
            x = np.load(self.test_input[0])
            for name in x.files:
                self.inputs[name] = x[name]
            if gen_ref:
                gen_input_f32 = self.inputs
        else:
            assert (len(self.test_input) == len(self.module.inputs))
            for infile, op in zip(self.test_input, self.module.inputs):
                if infile.endswith(('.jpg', '.jpeg', '.png')):
                    ppa = preprocess()
                    input_op = op.op
                    input_shape = [Operation.shape(input_op)]
                    ppa.load_config(input_op)
                    if self.fuse_preprocess:
                        #fuse_preprocess should input origin image format
                        if self.customization_format == '':
                            self.customization_format = getCustomFormat(
                                ppa.pixel_format, ppa.channel_format)
                        config = {
                            'input_shapes': input_shape,
                            'resize_dims': ppa.resize_dims,
                            'fuse_pre': True,
                            'keep_aspect_ratio': ppa.keep_aspect_ratio,
                            'keep_ratio_mode': ppa.keep_ratio_mode,
                            "pixel_format": ppa.pixel_format,
                            'customization_format': self.customization_format,
                            'aligned': self.aligned_input,
                            'pad_type': ppa.pad_type,
                            'pad_value': ppa.pad_value,
                            'chip': self.chip,
                        }
                        print("Add preprocess, set the following params:")
                        ppb = preprocess()
                        ppb.config(**config)
                        self.inputs[op.name + "_raw"] = ppb.run(infile)
                        self.in_f32_npz = self.module_name + "_in_ori.npz"
                    else:
                        self.inputs[ppa.input_name] = ppa.run(infile)
                    if gen_ref:
                        gen_input_f32[ppa.input_name] = ppa.run(infile)

                elif infile.endswith(".npy"):
                    data = np.load(infile)
                    self.inputs[op.name] = data
                    if gen_ref:
                        gen_input_f32[op.name] = self.inputs
                else:
                    raise TypeError("Unsupport input type *{}".format(os.path.splitext(infile)))
        if self.aligned_input and not self.fuse_preprocess:
            raise RuntimeError(
                "Not support now, aligned_input requires fuse_preprocess to be set to True.")
        np.savez(self.in_f32_npz, **self.inputs)
        if gen_ref:
            gen_in_f32_npz = self.module_name + '_in_f32.npz'
            file_mark(gen_in_f32_npz)
            np.savez(gen_in_f32_npz, **gen_input_f32)
            self.ref_npz = self.module_name + "_top_outputs.npz"
            show_fake_cmd(gen_in_f32_npz, self.mlir_file, self.ref_npz)
            top_outputs = mlir_inference(gen_input_f32, self.mlir_file)
            np.savez(self.ref_npz, **top_outputs)
        self.tpu_npz = "{}_tpu_outputs.npz".format(self.prefix)
        file_mark(self.tpu_npz)

    def validate_tpu_mlir(self):
        show_fake_cmd(self.in_f32_npz, self.tpu_mlir, self.tpu_npz)
        tpu_outputs = mlir_inference(self.inputs, self.tpu_mlir, self.compare_all)
        np.savez(self.tpu_npz, **tpu_outputs)
        # compare fp32 blobs and quantized tensors with tolerance similarity
        f32_blobs_compare(self.tpu_npz, self.ref_npz, self.tolerance, self.excepts)

    def build_model(self):
        if self.chip == 'cpu':
            tosa_to_llvm(self.tosa_mlir, self.model)
        else:
            mlir_to_model(self.tpu_mlir, self.model, self.final_mlir, self.dynamic,
                          self.quant_input, self.quant_output, self.disable_layer_group,
                          self.merge_weight, self.op_divide, self.num_device, self.num_core,
                          self.embed_debug_info, self.model_version)
            if not self.skip_validation and self.do_validate:
                tool.validate_model()

    def validate_model(self):
        size = os.path.getsize(self.model)
        self.model_npz = "{}_model_outputs.npz".format(self.prefix)
        file_mark(self.model_npz)
        show_fake_cmd(self.in_f32_npz, self.model, self.model_npz)
        model_outputs = model_inference(self.inputs, self.model)
        np.savez(self.model_npz, **model_outputs)
        if self.state == "TOP_QUANTIZED":
            f32_blobs_compare(self.model_npz, self.ref_npz, self.correctness, self.excepts, True)
        else:
            f32_blobs_compare(self.model_npz, self.tpu_npz, self.correctness, self.excepts, True)


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--mlir", required=True,
                        help="optimized mlir fp32 model")
    parser.add_argument("--calibration_table",
                        help="calibration table for int8 quantization")
    parser.add_argument("--quantize_table",
                        help="table of OPs that quantized to specific mode")
    parser.add_argument("--quantize", default="F32", type=str.upper, choices=['F32', 'BF16', 'F16', 'INT8', 'INT4', 'QDQ'],
                        help="set default qauntization type: F32/BF16/F16/INT8")
    parser.add_argument("--asymmetric", action='store_true',
                        help="do INT8 asymmetric quantization")
    parser.add_argument("--ignore_f16_overflow", action='store_true',
                        help="some ops convert from f16 to f32, to avoid f16 overflow. These Ops are: LayerNorm, RMSNorm, AvgPool")
    parser.add_argument("--chip", required=True, type=str.lower,
                        choices=['bm1686', 'bm1684x', 'bm1684', 'sg2260', 'mars3',
                                 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'cpu'],
                        help="chip platform name")
    parser.add_argument("--model", required=True, help='output model')
    parser.add_argument("--dynamic", action='store_true',
                        help="do compile dynamic")
    # tosa includeWeight
    parser.add_argument("--includeWeight", action='store_true',
                        help="include weight in tosa.mlir")
    # fuse preprocess
    parser.add_argument("--fuse_preprocess", action='store_true',
                        help="add tpu preprocesses (mean/scale/channel_swap) in the front of model")
    parser.add_argument("--customization_format", default='', type=str.upper,
                        choices=supported_customization_format,
                        help="pixel format of input frame to the model")
    parser.add_argument("--aligned_input", action='store_true',
                        help='if the input frame is width/channel aligned')
    # check result
    parser.add_argument("--test_input", default="", type=str2list,
                        help="input npy/npz/image file for inference; image if fuse preprocess"
                        "if has more than one input, join npy with semicolon")
    parser.add_argument("--test_reference", default="",
                        help="reference npz file; if none, will run inner")
    parser.add_argument("--compare_all", action="store_true",
                        help="Decide if compare all tensors when lowering")
    parser.add_argument("--excepts", default='-', help="excepts tensors no compare")
    parser.add_argument("--tolerance", default='0.8,0.5', help="tolerance for compare")
    # other functions
    parser.add_argument("--quant_input", action="store_true",
                        help="strip input type cast in bmodel, need outside type conversion")
    parser.add_argument("--quant_output", action="store_true",
                        help="strip output type cast in bmodel, need outside type conversion")
    parser.add_argument("--disable_layer_group", action="store_true",
                        help="Decide whether to enable layer group pass")
    parser.add_argument("--op_divide", action="store_true",
                        help="if do large global op divide.")
    parser.add_argument("--num_device", default=1, type=int,
                        help="The number of devices to run for distributed computation.")
    parser.add_argument("--num_core", default=1, type=int,
                        help="The number of TPU cores used for parallel computation.")
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    parser.add_argument("--skip_validation", action='store_true', help='skip checking the correctness of bmodel.')
    parser.add_argument("--merge_weight", action="store_true", default=False,
                        help="merge weights into one weight binary with previous generated cvimodel")
    parser.add_argument("--linear_quant_mode", default="NORMAL", type=str.upper, choices=["W8A16", "W4A16", "NORMAL"],
                    help="quantize linear with W8A16 / W4A16 strategy")
    parser.add_argument("--do_winograd", action="store_true", default=False,
                        help="do_winograd")
    parser.add_argument("--model_version", default="latest",
                        help="if need old version cvimodel, set the verion, such as 1.2")

    # yapf: enable
    args = parser.parse_args()
    if args.linear_quant_mode != "NORMAL":
        assert((args.quantize == "F16" or args.quantize == "BF16")
               and "W8A16 and W4A16 can only be used in F16/ BF16 quantize mode")
    if args.customization_format.startswith("YUV"):
        args.aligned_input = True
    if not args.fuse_preprocess and args.customization_format:
        assert (0 and "Error! If not fuse_preprocess, customization_format shouldn't be set.")
    tool = DeployTool(args)
    # lowering to tpu/tosa
    tool.lowering()
    # generate model
    tool.build_model()
    if not args.debug:
        tool.cleanup()
