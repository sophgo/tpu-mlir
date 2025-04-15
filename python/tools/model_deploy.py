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
from utils.misc import *
from utils.preprocess import preprocess, supported_customization_format
from utils.auto_remove import file_mark, file_clean
from tools.model_runner import mlir_inference, model_inference, show_fake_cmd
import pymlir
from utils.misc import str2bool
from utils.log_setting import setup_logger
from utils.cache_tool import CommandRecorder

logger = setup_logger("deploy")


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
    elif pixel_format == "gbrg":
        custom_format == "GBRG_RAW"
    elif pixel_format == "grbg":
        custom_format == "GRBG_RAW"
    elif pixel_format == "bggr":
        custom_format == "BGGR_RAW"
    elif pixel_format == "rggb":
        custom_format == "RGGB_RAW"
    else:
        logger.info("pixel_format of {} no supported!".format(pixel_format))
        assert 0
    return custom_format


class DeployTool:

    def __init__(self, args):
        self.mlir_file = args.mlir
        self.chip = args.chip
        self.includeWeight = args.includeWeight
        self.excepts = args.excepts
        self.quantize = args.quantize.lower()
        self.tolerance = args.tolerance
        if not self.tolerance:
            if self.quantize in ["int8", "w8f16", "w8bf16", "w4f16", "w4bf16"]:
                self.tolerance = "0.90,0.80"
            elif self.quantize in ["f16", "bf16"]:
                self.tolerance = "0.99,0.90"
            elif self.quantize in ["f32"]:
                self.tolerance = "0.99,0.99"
            elif self.quantize in ["f8e4m3", "f8e5m2"]:
                self.tolerance = "0.80,0.50"
            else:
                raise RuntimeError("Please set the tolerance for quantization type: {}".format(
                    self.quantize))
        self.test_input = args.test_input
        self.asymmetric = args.asymmetric
        self.cali_table = args.calibration_table
        self.quant_input = args.quant_input
        self.quant_output = args.quant_output
        self.quant_input_list = args.quant_input_list
        self.quant_output_list = args.quant_output_list
        self.quant_output_bf16 = args.quant_output_bf16
        self.quantize_table = args.quantize_table
        self.embed_debug_info = args.debug
        self.lg_debugger = args.debug
        self.debug_cmd = args.debug_cmd
        self.bmodel_path = args.model
        self.ref_npz = args.test_reference
        self.fazzy_match = args.fazzy_match
        self.customization_format = args.customization_format
        self.fuse_preprocess = args.fuse_preprocess
        self.aligned_input = args.aligned_input
        self.do_winograd = args.do_winograd
        self.module = MlirParser(args.mlir)
        self.module_name = self.module.module_name
        self.state = self.module.module_state
        self.disable_layer_group = args.disable_layer_group
        self.gdma_check = not args.disable_gdma_check
        self.opt = args.opt
        self.merge_weight = args.merge_weight
        self.op_divide = args.op_divide
        self.high_precision = args.high_precision
        self.num_device = args.num_device
        self.num_core = args.num_core
        self.group_by_cores = args.group_by_cores
        self.correctness = "0.99,0.90"
        if self.quantize_table:
            self.correctness = "0.99,0.85"
        self.in_f32_npz = self.module_name + "_in_f32.npz"
        self.prefix = "{}_{}_{}".format(self.module_name, self.chip, self.quantize)
        self.dynamic = args.dynamic
        self.compare_all = args.compare_all

        self.skip_validation = args.skip_validation
        self.model_version = args.model_version
        self.addr_mode = args.addr_mode
        self.cuda = args.cuda
        self.q_group_size = args.q_group_size if self.quantize in [
            "w4f16", "w4bf16", "w8f16", "w8bf16"
        ] else 0
        self.q_symmetric = args.q_symmetric
        if self.quantize == "int8" or self.quantize == "int4":
            if self.asymmetric:
                self.prefix += "_asym"
            else:
                self.prefix += "_sym"
        self.cache_skip = args.cache_skip

        self.tpu_npz = "{}_tpu_outputs.npz".format(self.prefix)
        self.model_npz = "{}_model_outputs.npz".format(self.prefix)
        self._prepare_input_npz()
        self.patterns_count = args.patterns_count
        self.compress_mode = args.compress_mode if self.chip == "bm1688" else "none"
        self.mute = args.not_gen_bmodel
        self.matmul_perchannel = args.matmul_perchannel
        self.enable_maskrcnn = args.enable_maskrcnn
        self.future_update_rank = args.future_update_rank
        self.future_update_list = args.future_update_list
        self.gelu_mode = args.gelu_mode
        self.time_fixed_subnet = args.time_fixed_subnet
        self.subnet_params = args.subnet_params

        self.tosa_mlir = "{}_tosa.mlir".format(self.prefix)
        self.tpu_mlir = "{}_tpu.mlir".format(self.prefix)
        self.tpu_opt_mlir = "{}_tpu_opt.mlir".format(self.prefix)
        self.final_mlir = "{}_final.mlir".format(self.prefix)

        self.trunc_final = args.trunc_final
        if self.trunc_final:
            self.compare_all = True
        self.opt_post_processor = args.opt_post_processor

        self.context_dir = os.path.splitext(self.bmodel_path)[0]
        os.makedirs(self.context_dir, exist_ok=True)
        self.file_recorder_cache_path = os.path.join(self.context_dir, f"ref_files.json")

        self.file_recorder = CommandRecorder(self.file_recorder_cache_path)
        self.file_recorder.clear()
        self.file_recorder.update_file(f"{self.mlir_file.replace('.mlir','')}.ref_files.json")
        self.file_recorder.add_file(tpuc_opt=shutil.which("tpuc-opt"), )
        self.file_recorder.add_property(prefix=self.prefix)
        self.file_recorder.add_command(deploy_cmd=" ".join(sys.argv))
        self.file_recorder.add_property(chip=self.chip, compare_all=self.compare_all)
        self.file_recorder.dump()

    def cleanup(self):
        file_clean()

    def pack_profile(self):
        import shutil
        import datetime
        t = datetime.datetime.now()
        date = t.strftime("%y%m%d")
        profile_path = f'./{self.prefix}_profile_{date}/'
        os.makedirs(profile_path, exist_ok=True)

        shutil.copy2(self.mlir_file, os.path.join(profile_path, f'{self.prefix}_top.mlir'))
        mlir2onnx(self.mlir_file, os.path.join(profile_path, f'{self.prefix}_top.onnx'))

        shutil.copy2(self.tpu_mlir, os.path.join(profile_path, f'{self.prefix}_tpu.mlir'))
        mlir2onnx(self.tpu_mlir, os.path.join(profile_path, f'{self.prefix}_tpu.onnx'))

        shutil.copy2(self.tpu_opt_mlir, os.path.join(profile_path, f'{self.prefix}_tpu_opt.mlir'))
        mlir2onnx(self.tpu_opt_mlir, os.path.join(profile_path, f'{self.prefix}_tpu_opt.onnx'))

        shutil.copy2(self.final_mlir, os.path.join(profile_path, f'{self.prefix}_final.mlir'))

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
            return {}
        else:
            file_mark(self.tpu_mlir)
            patterns = mlir_lowering(
                self.mlir_file,
                self.tpu_mlir,
                self.quantize,
                self.chip,
                self.num_device,
                self.num_core,
                self.cali_table,
                self.asymmetric,
                self.quantize_table,
                self.customization_format,
                self.fuse_preprocess,
                self.aligned_input,
                self.high_precision,
                self.do_winograd,
                self.q_group_size,
                self.q_symmetric,
                True if self.patterns_count else False,
                addr_mode=self.addr_mode,
                mute=self.mute,
                matmul_perchannel=self.matmul_perchannel,
                gelu_mode=self.gelu_mode,
            )
            if self.do_validate and not self.enable_maskrcnn:
                self.validate_tpu_mlir()
            return patterns

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
        self.tpu_inputs = {}
        gen_input_f32 = {}
        gen_ref = True if len(self.ref_npz) == 0 else False

        if self.fuse_preprocess:
            assert (self.test_input[0].endswith(('.jpg', '.jpeg', '.png', '.yuv')))
        if num_inputs == 1 and self.test_input[0].endswith(".npz"):
            x = np.load(self.test_input[0])
            for name in x.files:
                self.tpu_inputs[name] = x[name]
            if gen_ref:
                gen_input_f32 = self.tpu_inputs
        else:
            assert (len(self.test_input) == len(self.module.inputs))
            for infile, op in zip(self.test_input, self.module.inputs):

                if infile.endswith(('.jpg', '.jpeg', '.png', '.yuv')):
                    ppa = preprocess()
                    input_op = op.op
                    input_shape = [Operation.shape(input_op)]
                    ppa.load_config(input_op)
                    if self.fuse_preprocess:
                        # fuse_preprocess should input origin image format
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
                        logger.info("Add preprocess, set the following params:")
                        ppb = preprocess()
                        ppb.config(**config)
                        self.tpu_inputs[op.name + "_raw"] = ppb.run(infile)
                        self.in_f32_npz = self.module_name + "_in_ori.npz"
                    else:
                        self.tpu_inputs[ppa.input_name] = ppa.run(infile)
                    if gen_ref:
                        gen_input_f32[ppa.input_name] = ppa.run(infile)

                elif infile.endswith(".npy"):
                    data = np.load(infile)
                    self.tpu_inputs[op.name] = data
                    if gen_ref:
                        gen_input_f32[op.name] = self.tpu_inputs[op.name]
                else:
                    raise TypeError("Unsupport input type *{}".format(os.path.splitext(infile)))
        if self.aligned_input and not self.fuse_preprocess:
            raise RuntimeError(
                "Not support now, aligned_input requires fuse_preprocess to be set to True.")
        np.savez(self.in_f32_npz, **self.tpu_inputs)
        if gen_ref:
            gen_in_f32_npz = self.module_name + '_in_f32.npz'
            file_mark(gen_in_f32_npz)
            np.savez(gen_in_f32_npz, **gen_input_f32)
            self.ref_npz = self.module_name + "_top_outputs.npz"
            show_fake_cmd(gen_in_f32_npz, self.mlir_file, self.ref_npz)
            top_outputs = mlir_inference(gen_input_f32, self.mlir_file)
            np.savez(self.ref_npz, **top_outputs)

        if not self.cache_skip:
            file_mark(self.tpu_npz)

        if not self.cache_skip:
            file_mark(self.model_npz)
        # dynamic layer output data dump
        if "NEED_DUMP_DYNAMIC_LAYER_OUTPUT_DATA" in os.environ and os.environ[
                "NEED_DUMP_DYNAMIC_LAYER_OUTPUT_DATA"] == "1":
            if self.chip in ("bm1684x", "bm1688") and self.state != "TOP_QUANTIZED":
                dyn_layer_out_data_path = "./.tmp"
                os.system(f"export DYNAMIC_LAYER_OUTPUT_DATA_PATH={dyn_layer_out_data_path}")
                os.system("export DYNAMIC_LAYER_OUTPUT_ID_DICT_PATH={}".format(
                    os.path.join(dyn_layer_out_data_path, "id_dict")))
                if not os.path.exists(dyn_layer_out_data_path):
                    os.makedirs(dyn_layer_out_data_path)
                else:
                    os.system(f"rm -rf {dyn_layer_out_data_path}/*")

    def validate_tpu_mlir(self):
        show_fake_cmd(self.in_f32_npz, self.tpu_mlir, self.tpu_npz, self.compare_all)
        tpu_outputs = mlir_inference(self.tpu_inputs, self.tpu_mlir, self.compare_all)
        np.savez(self.tpu_npz, **tpu_outputs)
        if self.trunc_final:
            self.tpu_inputs.update(tpu_outputs)
        # compare fp32 blobs and quantized tensors with tolerance similarity
        f32_blobs_compare(self.tpu_npz,
                          self.ref_npz,
                          self.tolerance,
                          self.excepts,
                          fuzzy_match=self.fazzy_match)
        if self.cuda:
            show_fake_cmd(self.in_f32_npz, self.tpu_mlir, self.tpu_npz, self.compare_all, True)
            cuda_outputs = mlir_inference(self.tpu_inputs,
                                          self.tpu_mlir,
                                          self.compare_all,
                                          use_cuda=True)
            cuda_npz = self.tpu_npz.replace("_tpu_", "_cuda_")
            np.savez(cuda_npz, **cuda_outputs)
            file_mark(cuda_npz)
            f32_blobs_compare(cuda_npz, self.tpu_npz, "0.9999,0.9999", self.excepts)

    def build_model(self):
        try:
            if self.chip == "cpu":
                tosa_to_llvm(self.tosa_mlir, self.bmodel_path)
                return {}
            else:
                command_mem = {}
                patterns = mlir_to_model(
                    tpu_mlir=self.tpu_mlir,
                    bmodel_path=self.bmodel_path,
                    final_mlir=self.final_mlir,
                    dynamic=self.dynamic,
                    quant_input=self.quant_input,
                    quant_output=self.quant_output,
                    quant_input_list=self.quant_input_list,
                    quant_output_list=self.quant_output_list,
                    disable_layer_group=self.disable_layer_group,
                    opt=self.opt,
                    merge_weight=self.merge_weight,
                    op_divide=self.op_divide,
                    embed_debug_info=self.embed_debug_info,
                    group_by_cores=self.group_by_cores,
                    model_version=self.model_version,
                    count_patterns=True if self.patterns_count else False,
                    compress_mode=self.compress_mode,
                    future_update_rank=self.future_update_rank,
                    future_update_list=self.future_update_list,
                    debug_info=self.debug_cmd,
                    trunc_final=self.trunc_final,
                    command_mem=command_mem,
                    quant_output_bf16=self.quant_output_bf16,
                    opt_post_processor=self.opt_post_processor,
                    gdma_check=self.gdma_check,
                    lg_debugger=self.lg_debugger,
                    time_fixed_subnet = self.time_fixed_subnet,
                    subnet_params = self.subnet_params,
                    layer_group_cache = (
                        f"{self.prefix}.layer_group_cache.json"
                        if self.quantize != "int8"
                        else f"{self.prefix.removesuffix('_sym')}.layer_group_cache.json")
                )
                if not self.skip_validation and self.do_validate:
                    self.validate_model()

            return patterns
        finally:
            if self.chip != "cpu":
                self.file_recorder.add_file(
                    bmodel=self.bmodel_path,
                    tensor_location=f"{self.bmodel_path}.json",
                    final_mlir=self.final_mlir,
                    tpu_mlir=self.tpu_mlir,
                    tpu_opt_mlir=self.tpu_opt_mlir,
                    tpu_output=self.tpu_npz,
                    bmodel_output=self.model_npz,
                    context_dir=self.context_dir,
                    layer_group_cache=f"{self.prefix}.layer_group_cache.json",
                )
                self.file_recorder.add_command(**command_mem)
                self.file_recorder.dump()

    def revise_MaskRCNN_tpu_ref(self):
        if self.enable_maskrcnn:
            dict_ref_npz = np.load(self.ref_npz)
            dict_model_npz = np.load(self.model_npz)
            keys_list = list(dict_model_npz.keys())
            temp_ref_ = dict()
            for i, per_key in enumerate(dict_ref_npz.keys()):
                temp_ref_[keys_list[i]] = dict_ref_npz[per_key]
            np.savez(self.tpu_npz, **temp_ref_)

    def validate_model(self):
        show_fake_cmd(self.in_f32_npz, self.bmodel_path, self.model_npz, self.compare_all)
        model_outputs = model_inference(self.tpu_inputs, self.bmodel_path, self.compare_all)
        np.savez(self.model_npz, **model_outputs)
        if self.enable_maskrcnn:
            self.revise_MaskRCNN_tpu_ref()
        if self.state == "TOP_QUANTIZED":
            f32_blobs_compare(self.model_npz, self.ref_npz, self.correctness, self.excepts, True,
                              self.fazzy_match)
        else:
            f32_blobs_compare(self.model_npz, self.tpu_npz, self.correctness, self.excepts, True)


def deprecated_option(cond, msg):
    if cond:
        raise RuntimeError(msg)


if __name__ == '__main__':
    logger.info("TPU-MLIR {}".format(pymlir.__version__))
    parser = argparse.ArgumentParser()
    # yapf: disable
    # ========== Basic Options ===========
    parser.add_argument("--mlir", required=True, help="top mlir from model_transform.py")
    parser.add_argument("--chip", "--processor", required=True, type=str.lower,
                        choices=['bm1688', 'bm1684x', 'bm1684', 'bm1690', 'mars3', 'sgtpuv8', 'sg2380',
                                 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'sg2262', 'cpu'],
                        help="chip platform name")
    parser.add_argument("--quantize", default="F32", type=str.upper,
                        choices=['F32', 'BF16', 'F16', 'INT8', 'INT4', 'W8F16', 'W8BF16',
                                 'W4F16', 'W4BF16', "F8E4M3", "F8E5M2", 'QDQ'],
                        help="set default qauntization type")
    parser.add_argument("--model", required=True, help='output model')
    # ========== Quantization Options ==============
    parser.add_argument("--calibration_table",
                        help="calibration table for int8 quantization")
    parser.add_argument("--quantize_table",
                        help="table of OPs that quantized to specific mode")
    parser.add_argument("--asymmetric", action='store_true',
                        help="do INT8 asymmetric quantization")
    parser.add_argument("--q_group_size", default=64, type=int,
                        help="group size for per-group quant, only used in W4A16/W8A16 quant mode")
    parser.add_argument("--q_symmetric", action='store_true',
                        help="do symmetric W4A16/W8A16 quant, only works for per-group quant")
    parser.add_argument("--quant_input", action="store_true",
                        help="strip input type cast in bmodel, need outside type conversion")
    parser.add_argument("--quant_output", action="store_true",
                        help="strip output type cast in bmodel, need outside type conversion")
    parser.add_argument("--quant_output_bf16", action="store_true",
                        help="force output to be bf16 type")
    parser.add_argument("--quant_input_list", default="", type=str,
                        help="choose index to strip cast, such as 1,3 means first & third input`s cast")
    parser.add_argument("--quant_output_list", default="", type=str,
                        help="choose index to strip cast, such as 1,3 means first & third output`s cast")
    parser.add_argument("--high_precision", action='store_true',
                        help="some ops will force to be fp32")
    parser.add_argument("--ignore_f16_overflow", action='store_true',
                        help="some ops convert from f16 to f32, to avoid f16 overflow. These Ops are: LayerNorm, RMSNorm, AvgPool")
    # ========== Validation Options ==============
    parser.add_argument("--test_input", default="", type=str2list,
                        help="input npy/npz/image file for inference; image if fuse preprocess"
                        "if has more than one input, join npy with semicolon")
    parser.add_argument("--test_reference", default="",
                        help="reference npz file; if none, will run inner")
    parser.add_argument("--compare_all", action="store_true",
                        help="Decide if compare all tensors when lowering")
    parser.add_argument("--tolerance", default='', help="tolerance for compare")
    parser.add_argument("--excepts", default='-', help="excepts tensors no compare")
    parser.add_argument("--skip_validation", action='store_true', help='skip checking the correctness of bmodel.')
    parser.add_argument("--cache_skip", action='store_true', help='skip checking the correctness when generate same mlir and bmodel.')
    parser.add_argument("--fazzy_match", action="store_true",
                        help="do fazzy match bettwen target and ref data")
    parser.add_argument("--cuda", action="store_true", help="do inference by cuda")
    # ========== Fuse Preprocess Options ==============
    parser.add_argument("--fuse_preprocess", action='store_true',
                        help="add tpu preprocesses (mean/scale/channel_swap) in the front of model")
    parser.add_argument("--customization_format", default='', type=str.upper,
                        choices=supported_customization_format,
                        help="pixel format of input frame to the model")
    parser.add_argument("--aligned_input", action='store_true',
                        help='if the input frame is width/channel aligned')
    # ========== Parallel Options ==============
    parser.add_argument("--num_device", default=1, type=int,
                        help="The number of devices to run for distributed computation.")
    parser.add_argument("--num_core", default=1, type=int,
                        help="The number of TPU cores used for parallel computation.")
    parser.add_argument("--group_by_cores", default="auto", type=str.lower,
                        choices=['auto', 'true', 'false'],
                        help="whether layer groups force group by cores")
    # ========== Compiler Options ==============
    parser.add_argument("--dynamic", action='store_true', help="do compile dynamic")
    parser.add_argument("--opt", default=2, type=int, choices=[1, 2, 3], help="Optimization level")
    parser.add_argument("--addr_mode", default="auto", type=str.lower,
                        choices=['auto', 'basic', 'io_alone', 'io_tag', 'io_tag_fuse', 'io_reloc'],
                        help="set address assign mode, if not set, auto as default")
    parser.add_argument("--not_gen_bmodel", action="store_true",
                        help="for qat intergation, only gen tpu.mlir")
    # ========== Debug Options ==============
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    parser.add_argument("--disable_layer_group", action="store_true", help="Whether to enable layer group pass")
    parser.add_argument("--disable_gdma_check", action='store_true', help='disable gdma addr check')
    parser.add_argument("--trunc_final", nargs="*", help="assign op to be trunced in final mlir.")
    parser.add_argument("-V", "--version", action='version', version='%(prog)s ' + pymlir.__version__)
    # ========== Other Options ==============
    # for cv18xx
    parser.add_argument("--op_divide", action="store_true", help="if do large global op divide.")
    parser.add_argument("--merge_weight", action="store_true", default=False,
                        help="merge weights into one weight binary with previous generated cvimodel")
    parser.add_argument("--model_version", default="latest",
                        help="if need old version cvimodel, set the verion, such as 1.2")
    # for bm1684
    parser.add_argument("--do_winograd", action="store_true", default=False,
                        help="do_winograd")
    # for tosa includeWeight
    parser.add_argument("--includeWeight", action='store_true', help="include weight in tosa.mlir")
    # for bm1688
    parser.add_argument("--compress_mode", default="none", type=str.lower,
                        choices=["none", "weight", "activation", "all"],
                        help="set compress mode")
    # for bm1684x and bm1688
    parser.add_argument("--matmul_perchannel", action="store_true", default=False,
                        help="if quantize matmul in per-channel mode for BM1684x and BM1688")
    # for mars3
    parser.add_argument("--opt_post_processor", action="store_true", default=False,
                        help="opt_post_processor")
    # regression test only, not for users
    parser.add_argument("--patterns_count", type=str2dict, default=dict(),
                    help='used for regression test, check if patterns are successfully applied a specific number of times')
    parser.add_argument("--gelu_mode", default="normal", type=str.lower,
                        help="how to approximate gelu, possible options: normal/tanh/sigm")
    # Customized requirements: Segmenting the model based on fixed time intervals.
    parser.add_argument('--time_fixed_subnet', default=None, type=str.lower, choices=['normal', 'limit', 'custom'],
                    help='Split the model by fixed duration intervals')
    parser.add_argument('--subnet_params', default=None,
                    help='When time_fixed_subnet is custom, it is used to set the frequency(MHZ) and duration(ms) of the subnet')
    # ========== DEPRECATED Options ==============
    parser.add_argument("--io_alone", action="store_true", default=False,
                        help="DEPRECATED, please use --addr_mode io_alone")
    # ========== MaskRCNN Options ==============
    parser.add_argument("--enable_maskrcnn", action="store_true", default=False,
                        help="enable maskrcnn")
    # ========== Future Update Options ==============
    parser.add_argument("--future_update_rank", default=0, type=int,
                        help="the rank of matmul, when use the pass of future-update")
    parser.add_argument("--future_update_list", default="", type=str,
                        help="the idx list of weight, when use the pass of future-update, suck as 1,2,3")
    parser.add_argument("--debug_cmd", default="", type=str,
                        help="debug cmd")

    # yapf: enable
    args = parser.parse_args()
    deprecated_option(args.io_alone, "DEPRECATED, please use --addr_mode io_alone")
    deprecated_option(args.ignore_f16_overflow, "DEPRECATED, please use --high_precision")
    if args.quant_output_bf16:
        if args.quantize == "BF16":
            RuntimeError("quantize is BF16, please use --quant_output instead")
        if args.quant_output:
            RuntimeError("quant_output and quant_output_bf16 can't both be true")

    if args.customization_format.startswith("YUV"):
        args.aligned_input = True
    if not args.fuse_preprocess and args.customization_format:
        assert (0 and "Error! If not fuse_preprocess, customization_format shouldn't be set.")
    tool = DeployTool(args)
    # lowering to tpu/tosa
    if args.not_gen_bmodel:
        tool.do_validate = False
    lowering_patterns = tool.lowering()
    # generate model
    if args.time_fixed_subnet == 'custom' and not args.subnet_params:
        parser.error("time_fixed_subnet is custom, please use --subnet_params to set the frequency(MHZ) and duration(ms) of the subnet.")
    if args.not_gen_bmodel:
        exit(0)
    tpu_patterns = tool.build_model()
    if not args.debug:
        tool.cleanup()
    total_patterns = {**lowering_patterns, **tpu_patterns}
    if args.patterns_count:
        for k, v in args.patterns_count.items():
            assert k in total_patterns and v == total_patterns[k], \
            "The number of times {} was applied does not meet the requirements. Expected {}, got {}" \
            .format(k, v, total_patterns.get(k))
