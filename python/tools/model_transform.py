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

from transform.BaseConverter import BaseConverter
from utils.mlir_shell import *
from utils.mlir_parser import *
from utils.misc import *
from utils.auto_remove import file_mark, file_clean
from utils.preprocess import get_preprocess_parser, preprocess
from utils.cache_tool import CacheTool
from utils.log_setting import setup_logger
import pymlir
import warnings

logger = setup_logger("transform")

class ModelTransformer(object):

    def __init__(self, model_name, model_def):
        self.model_name = model_name
        self.model_def = model_def
        self.do_mlir_infer = True
        self.converter = BaseConverter()
        self.in_f32_npz = self.model_name + '_in_f32.npz'
        self.ref_npz = self.model_name + '_ref_outputs.npz'

    def cleanup(self):
        file_clean()

    @staticmethod
    def ensure_batch_size(arr: np.ndarray, batch_size):
        """arr: [old_batch_size, ...]"""
        old_batch_size = arr.shape[0]
        if old_batch_size > 1:
            return arr
        repeat_factor = int(np.ceil(batch_size / old_batch_size))
        repeated_arr = np.repeat(arr, repeat_factor, axis=0)
        trimmed_arr = repeated_arr[:batch_size]
        return trimmed_arr

    def model_transform(self, mlir_file: str, add_postprocess: str="", patterns_count: dict={}):
        self.mlir_file = mlir_file
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        if self.converter:
            file_mark(mlir_origin)
            self.converter.generate_mlir(mlir_origin)
        else:
            mlir_origin = self.model_def # skip frontend conversion if model_def is origin.mlir
        patterns = mlir_opt_for_top(mlir_origin, self.mlir_file, add_postprocess, True if patterns_count else False)
        if patterns_count:
            for k, v in patterns_count.items():
                assert k in patterns and v == patterns[k], \
                "The number of times {} was applied does not meet the requirements. Expected {}, got {}" \
                .format(k, v, patterns.get(k))

        logger.info("Mlir file generated:{}".format(mlir_file))

        self.module_parsered = MlirParser(self.mlir_file)
        self.input_num = self.module_parsered.get_input_num()

    def model_validate(self, file_list: str, tolerance, excepts, test_result):
        from tools.model_runner import mlir_inference, free_mlir_module, show_fake_cmd
        import gc
        inputs = dict()
        if len(file_list) == 1 and file_list[0].endswith('.npz'):
            npz_in = np.load(file_list[0])
            only_one = len(npz_in.files) == 1
            if only_one:
                assert (len(self.converter.input_names) == 1)
                name = self.converter.input_names[0]
                # for GRU, batch_second, ensure_batch_size will have no effects.
                batch_size = self.module_parsered.get_batch_size()
                inputs[name] = self.ensure_batch_size(npz_in[npz_in.files[0]], batch_size)
            else:
                for name in self.converter.input_names:
                    assert (name in npz_in.files)
                    batch_size = self.converter.getShape(name)[0]
                    inputs[name] = self.ensure_batch_size(npz_in[name], batch_size)
        elif file_list[0].endswith(('.jpg', '.jpeg', '.png', '.JPEG')):  #todo add isPicture in util
            ppa = preprocess()
            for i in range(self.input_num):
                pic_path = file_list[i] if i < len(file_list) else file_list[-1]
                file = os.path.expanduser(pic_path)
                ppa.load_config(self.module_parsered.get_input_op_by_idx(i))
                inputs[ppa.input_name] = ppa.run(file)
        else:
            assert (len(file_list) == len(self.converter.input_names))
            for name, file in zip(self.converter.input_names, file_list):
                assert (file.endswith('.npy'))
                inputs[name] = np.load(file)
        np.savez(self.in_f32_npz, **inputs)
        # original model inference to get blobs of all layer
        show_fake_cmd(self.in_f32_npz, self.model_def, self.ref_npz)
        ref_outputs = self.origin_inference(inputs)
        if not self.do_mlir_infer:
            logger.info("Saving {}".format(test_result))
            np.savez(test_result, **ref_outputs)
            return
        logger.info("Saving {}".format(self.ref_npz))
        np.savez(self.ref_npz, **ref_outputs)
        del self.converter  #save memory
        del ref_outputs
        gc.collect()

        # inference of mlir model
        show_fake_cmd(self.in_f32_npz, self.mlir_file, test_result)
        f32_outputs = mlir_inference(inputs, self.mlir_file)
        logger.info("Saving {}".format(test_result))
        np.savez(test_result, **f32_outputs)
        del f32_outputs
        free_mlir_module()
        gc.collect()
        # compare all blobs layer by layers
        f32_blobs_compare(test_result, self.ref_npz, tolerance, excepts=excepts)
        file_mark(self.ref_npz)
        cache_tool.mark_top_success()

    @abc.abstractmethod
    def origin_inference(self, inputs: dict) -> dict:
        pass


class OnnxTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_def,
                 input_shapes: list = [],
                 output_names: list = [],
                 test_input='',
                 preprocessor: dict = {},
                 static_shape=True,
                 onnx_sim='',
                 dynamic_shape_input_names: list = [],
                 dynamic=False,
                 shape_influencing_input_names: list = [],
                 dump_final_opt=True):
        super().__init__(model_name, model_def)
        from transform.OnnxConverter import OnnxConverter
        self.converter = OnnxConverter(self.model_name,
                                       self.model_def,
                                       input_shapes,
                                       output_names,
                                       test_input,
                                       preprocessor,
                                       static_shape,
                                       onnx_sim=onnx_sim,
                                       dynamic_shape_input_names=dynamic_shape_input_names,
                                       dynamic=dynamic,
                                       shape_influencing_input_names=shape_influencing_input_names,
                                       dump_final_opt=dump_final_opt)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import onnx_inference
        return onnx_inference(inputs, self.converter.onnx_file)


class CaffeTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_def,
                 model_data,
                 input_shapes: list = [],
                 output_names: list = [],
                 preprocessor: dict = {},
                 shape_influencing_input_names: list = []):
        super().__init__(model_name, model_def)
        self.model_data = model_data
        from transform.CaffeConverter import CaffeConverter
        self.converter = CaffeConverter(self.model_name, self.model_def, model_data, input_shapes,
                                        output_names, preprocessor, shape_influencing_input_names=shape_influencing_input_names)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import caffe_inference
        return caffe_inference(inputs, self.model_def, self.model_data)


class TFLiteTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_def,
                 input_shapes: list = [],
                 output_names: list = [],
                 preprocessor: dict = {},
                 shape_influencing_input_names: list = []):
        super().__init__(model_name, model_def)
        self.do_mlir_infer = False
        from transform.TFLiteConverter import TFLiteConverter
        self.converter = TFLiteConverter(self.model_name, self.model_def, input_shapes,
                                         output_names, preprocessor, shape_influencing_input_names=shape_influencing_input_names)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import tflite_inference
        is_nchw = True
        tf_layout = 1
        if 'channel_format' in self.converter.preprocess_args:
            cf = self.converter.preprocess_args['channel_format']
            is_nchw = cf == 'nchw'
            tf_layout = 0
        return tflite_inference(inputs,
                                self.converter.tflite_file,
                                input_is_nchw=is_nchw,
                                tf_layout=tf_layout)


class TorchTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_def,
                 input_shapes: list = [],
                 input_types: list = [],
                 output_names: list = [],
                 preprocessor: dict = {},
                 dynamic=False,
                 shape_influencing_input_names: list = []):
        super().__init__(model_name, model_def)
        from transform.TorchConverter import TorchConverter
        self.converter = TorchConverter(self.model_name, self.model_def, input_shapes, input_types,
                                        output_names, preprocessor, dynamic=dynamic, shape_influencing_input_names=shape_influencing_input_names)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import torch_inference
        return torch_inference(inputs, self.model_def)

class MlirTransformer(ModelTransformer):

    def __init__(self,
                 model_name, model_def):
        super().__init__(model_name, model_def)
        parser = MlirParser(model_def)
        state = parser.module_state
        assert state == "TOP_F32" or state == "TOP_QUANTIZED", f"unsupport model: {model_def}"
        assert model_name == parser.module_name, f"model_name is inconsistent with module_name in {model_def}"
        weight_file = parser.module_weight_file
        assert os.path.exists(weight_file), f"{weight_file} not exist, please check!"
        self.converter = None # origin.mlir model no need converter

class MaskRCNNTransformer(ModelTransformer):

    def __init__(self,
                 model_name:   str  = None,
                 model_def:    list = [],
                 model_extern: list = [],
                 input_shapes: list = [],
                 input_types:  list = [],
                 output_names: list = [],
                 preprocessor: dict = {},
                 path_yaml:    list = []):
        super().__init__(model_name, model_def)
        from transform.MaskRCNNConverter import MaskRCNNConverter
        if not self.model_def.endswith('.pt'):
            raise RuntimeError("maskrcnn model only support torch.pt! unsupport model:{}".format(self.model_def))
        if model_extern is not None:
            for model in model_extern:
                if(not model.endswith('.pt')):
                    raise RuntimeError("maskrcnn model only support torch.pt! unsupport model:{}".format(model))

        self.converter = MaskRCNNConverter(self.model_name,
                                           self.model_def,
                                           model_extern,
                                           input_shapes,
                                           input_types,
                                           output_names,
                                           preprocessor,
                                           path_yaml)
        self.is_able_infer = model_extern is not None

    def origin_inference(self, inputs: dict):
        if (self.is_able_infer):
            from tools.model_runner import torch_inference
            return torch_inference(inputs, self.model_def)
        assert 0,"[MaskRCNN] only support single torch inference!"

def get_model_transform(args):
    preprocessor = preprocess()
    preprocessor.config(**vars(args))
    if not args.mlir.endswith('.mlir'):
        raise RuntimeError("your mlir file should endswith .mlir, not:{}".format(args.mlir))
    tool = None
    if args.enable_maskrcnn:
        tool = MaskRCNNTransformer(args.model_name,
                                   args.model_def,
                                   args.model_extern,
                                   args.input_shapes,
                                   args.input_types,
                                   args.output_names,
                                   preprocessor.to_dict(),
                                   path_yaml  =  args.path_yaml)
    elif args.model_def.endswith('.onnx'):
        tool = OnnxTransformer(args.model_name,
                               args.model_def,
                               args.input_shapes,
                               args.output_names,
                               args.test_input,
                               preprocessor.to_dict(),
                               onnx_sim=args.onnx_sim,
                               dynamic_shape_input_names=args.dynamic_shape_input_names,
                               dynamic=args.dynamic,
                               shape_influencing_input_names=args.shape_influencing_input_names,
                               dump_final_opt=args.dump_final_opt)
    elif args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
        tool = CaffeTransformer(args.model_name, args.model_def, args.model_data, args.input_shapes,
                                args.output_names, preprocessor.to_dict(),
                                shape_influencing_input_names=args.shape_influencing_input_names)
    elif args.model_def.endswith('.tflite'):
        tool = TFLiteTransformer(args.model_name, args.model_def, args.input_shapes,
                                 args.output_names, preprocessor.to_dict(),
                                 shape_influencing_input_names=args.shape_influencing_input_names)
    elif args.model_def.endswith('.pt'):
        tool = TorchTransformer(args.model_name, args.model_def, args.input_shapes,
                                args.input_types, args.output_names, preprocessor.to_dict(),
                                dynamic=args.dynamic, shape_influencing_input_names=args.shape_influencing_input_names)
    elif args.model_def.endswith('.mlir'):
        tool = MlirTransformer(args.model_name, args.model_def)
    else:
        # TODO: support more deep learning model types
        raise RuntimeError("unsupport model:{}".format(args.model_def))
    return tool

def model_transform_func(model_name, model_def, input_shapes, mlir_scale, mlir_mean, output_mlir):
    preprocessor = preprocess()
    preprocessor.config(mean=mlir_mean , scale=mlir_scale, pixel_format='rgb', keep_aspect_ratio=True)
    if not output_mlir.endswith('.mlir'):
        raise RuntimeError("your mlir file should endswith .mlir, not:{}".format(args.mlir))
    tool = OnnxTransformer(model_name,
                            model_def,
                            input_shapes,
                            [], '',
                            preprocessor.to_dict())
    tool.model_transform(output_mlir)

if __name__ == '__main__':
    logger.info("TPU-MLIR {}".format(pymlir.__version__))
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_def", required=True, help="model definition file.")
    parser.add_argument("--model_extern", default=None, type=str2list, help="multiple model definition files, all after the first model_def.")
    parser.add_argument("--model_data", help="caffemodel, only for caffe model")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                        help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    parser.add_argument("--input_types", type=str2list, default=list(),
                        help="list of input types, like:float32,int32. if not set, float32 as default")
    parser.add_argument("--output_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")
    parser.add_argument("--test_input", default="", type=str2list,
                        help="input jpg/npy/npz file for inference, "
                        "if has more than one input, join jpg or npy with semicolon")
    parser.add_argument("--test_result", default="", type=str,
                        help="if input is set, result is mlir inference result")
    parser.add_argument("--cache_skip", action='store_true', help='skip checking the correctness when generate same mlir and bmodel.')
    parser.add_argument("--tolerance", default='0.99,0.99',
                        help="minimum similarity tolerance to model transform")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--add_postprocess", default="", type=str.lower,
                        choices=['','yolov3','yolov5','yolov8','ssd','bnr'], help="add postprocess for model")
    parser.add_argument("--onnx_sim", default="", type=str, choices=['', 'skip_fuse_bn'],
                        help="pass options of onnx-sim, sep by quote without space")
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    parser.add_argument("--dump_final_opt", default=True, help='save final_opt onnx file')
    parser.add_argument("--mlir", type=str, required=True, help="output mlir model file")
    # regression test only, not for users
    parser.add_argument("--patterns_count", type=str2dict, default=dict(),
                        help='used for regression test, check if patterns are successfully applied a specific number of times')
    parser.add_argument("--dynamic_shape_input_names", type=str2list, default=list(),
                        help="name list of inputs with dynamic shape, like:input1,input2")
    parser.add_argument("--shape_influencing_input_names", type=str2list, default=list(),
                        help="name list of inputs which influencing other tensors\' shape during inference, like:input1,input2. \
                            if set, test_input is required")
    parser.add_argument("--dynamic", action='store_true',
                        help='only valid for onnx model. if set, will automatically set inputs with dyanmic axis \
                            as dynamic_shape_input_names and set 1-d inputs as shape_influencing_input_names')
    parser.add_argument("--path_yaml", type=str, default=None, help="path of the yaml file")
    # ========== MaskRCNN Options ==============
    parser.add_argument("--enable_maskrcnn", action='store_true', help="if enable maskrcnn")


    # yapf: enable
    parser = get_preprocess_parser(existed_parser=parser)
    args, unknown_args = parser.parse_known_args()
    if (args.enable_maskrcnn):
        assert ((not args.test_input) and (not args.test_result)), "[Error] Please don't give input/output when enable_MaskRCNN!"

    if unknown_args:
        args.unknown_params += unknown_args
    if args.test_input:
        for input in args.test_input:
            assert os.path.exists(input), f"test_input {input} not exist!"
    if args.shape_influencing_input_names:
        assert args.test_input, "if shape_influencing_input_names is set, test_input is required!"
    cache_tool = CacheTool(args.cache_skip)
    tool = get_model_transform(args)
    tool.model_transform(args.mlir, args.add_postprocess, args.patterns_count)
    if args.test_input and cache_tool.do_top_validate(tool.mlir_file, tool.in_f32_npz, args.tolerance, args.debug):
        assert (args.test_result)
        tool.model_validate(args.test_input, args.tolerance, args.excepts, args.test_result)
    if not args.debug:
        tool.cleanup()
