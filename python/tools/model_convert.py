#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import pymlir

from model_transform import OnnxTransformer, TorchTransformer
from utils.log_setting import setup_logger
from utils.misc import *
from utils.mlir_shell import origin_mlir_to_bmodel

logger = setup_logger("convert")


def get_model_transform(args):
    tool = None
    if args.model_def.endswith('.onnx'):
        tool = OnnxTransformer(model_name=args.model_name,
                               model_def=args.model_def,
                               input_shapes=args.input_shapes,
                               do_onnx_sim=args.do_onnx_sim,
                               dump_final_opt=args.debug)
    elif args.model_def.endswith('.pt'):
        tool = TorchTransformer(model_name=args.model_name,
                                model_def=args.model_def,
                                input_shapes=args.input_shapes,
                                input_types=args.input_types)
    else:
        # TODO: support more deep learning model types
        raise RuntimeError("unsupport model:{}".format(args.model_def))
    return tool


if __name__ == '__main__':
    logger.info("TPU-MLIR {}".format(pymlir.__version__))
    parser = argparse.ArgumentParser()
    # yapf: disable
    # ========== Basic Options ===========
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_def", required=True, help="model definition file.")
    parser.add_argument("--chip", "--processor", required=True, type=str.lower,
                        choices=['bm1688', 'bm1684x', 'bm1684', 'bm1690', 'mars3', 'sgtpuv8', 'sg2380',
                                 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'cpu'],
                        help="chip platform name")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                        help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    parser.add_argument("--input_types", type=str2list, default=list(),
                        help="list of input types, like:float32,int32. if not set, float32 as default")
    parser.add_argument("--model", required=True, help='output model')

    # ========== Quantization Options ==============
    parser.add_argument("--quantize", default="F32", type=str.upper,
                        choices=['F32', 'BF16', 'F16', 'INT8', 'INT4', 'W8F16', 'W8BF16',
                                 'W4F16', 'W4BF16', "F8E4M3", "F8E5M2", 'QDQ'],
                        help="set default qauntization type")
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
    # ========== Parallel Options ==============
    parser.add_argument("--num_device", default=1, type=int,
                        help="The number of devices to run for distributed computation.")
    parser.add_argument("--num_core", default=1, type=int,
                        help="The number of TPU cores used for parallel computation.")
    # ========== Compiler Options ==============
    parser.add_argument("--do_onnx_sim", default=False, type=bool, help="whether do onnx sim for onnx")
    parser.add_argument("--dynamic", action='store_true', help="do compile dynamic")
    parser.add_argument("--addr_mode", default="auto", type=str.lower,
                        choices=['auto', 'basic', 'io_alone', 'io_tag', 'io_tag_fuse'],
                        help="set address assign mode, if not set, auto as default")
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    # yapf: enable
    args = parser.parse_args()

    tool = get_model_transform(args)
    origin_mlir = tool.model_mlir()
    origin_mlir_to_bmodel(origin_mlir=origin_mlir,
                          bmodel_name=args.model,
                          mode=args.quantize,
                          chip=args.chip,
                          num_device=args.num_device,
                          num_core=args.num_core,
                          high_precision=args.high_precision,
                          q_group_size=args.q_group_size,
                          q_symmetric=args.q_symmetric,
                          addr_mode=args.addr_mode,
                          quant_input=args.quant_input,
                          quant_output=args.quant_output,
                          quant_input_list=args.quant_input_list,
                          quant_output_list=args.quant_output_list,
                          quant_output_bf16=args.quant_output_bf16,
                          dynamic=args.dynamic,
                          debug=args.debug)
    if not args.debug:
        tool.cleanup()
