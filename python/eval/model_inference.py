#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import importlib
import numpy as np
import argparse
import pymlir
pymlir.set_mem_mode("value_mem")
import onnx
import ast
import onnxruntime
import onnxsim.onnx_simplifier as onnxsim
from utils.preprocess import get_preprocess_parser, preprocess
from utils.mlir_parser import *
from utils.misc import *
from tools.model_runner import get_chip_from_model, round_away_from_zero



class common_inference():
    def __init__(self, args):
        self.idx = 0
        self.postprocess_type = args.postprocess_type
        if not args.model_file.endswith('.onnx'):
            model_file = args.model_file
            if args.model_file.endswith(".bmodel") or args.model_file.endswith(".cvimodel"):
                bmodel_mlir_file = ''.join(args.model_file.split('.')[:-1])
                model_file = f'{bmodel_mlir_file}_tpu.mlir'
                if not os.path.exists(model_file):
                    print(f'the mlir file:{model_file} of {args.model_file} is not exist, can not extract preprocess para')
                    exit(0)
            self.module = pymlir.module()
            self.module.load(model_file)
            self.module_parsered = MlirParser(model_file)
            self.batch_size = self.module_parsered.get_batch_size()
            args.batch_size = self.batch_size
            self.input_num = self.module_parsered.get_input_num()
            self.img_proc = preprocess(args.debug_cmd)
            self.img_proc.load_config(self.module_parsered.get_input_op_by_idx(0))
            args.net_input_dims = self.img_proc.net_input_dims
        self.batched_labels = []
        self.batched_imgs = ''
        exec('from eval.postprocess_and_score_calc.{name} import {name}'.format(name = args.postprocess_type))
        self.score = eval('{}(args)'.format(args.postprocess_type))
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        print('batch_size:', self.batch_size)

    def run(self, idx, img_path, target = None):
        self.idx = idx
        self.batched_imgs += '{},'.format(img_path)
        if target is not None:
            self.batched_labels.append(target)
        if (idx+1) % self.batch_size == 0:
            self.batched_imgs = self.batched_imgs[:-1]
            self.model_invoke()

    def get_result(self):
        if self.batched_imgs != '':
            print('get_result do the remained imgs')
            tmp = self.batched_imgs[:-1].split(',')
            n = self.batch_size - len(tmp)
            if n > 0:
                for i in range(n):
                    tmp.append(tmp[-1])
                    if len(self.batched_labels) > 0:
                        self.batched_labels.append(self.batched_labels[-1])
            self.batched_imgs = ','.join(tmp)
            self.idx += n
            self.model_invoke()
        return self.score.get_result()

    def model_invoke(self):
        ratio_list = None
        if 'not_use_preprocess' in self.debug_cmd:
            self.x = self.score.preproc(self.batched_imgs)
        else:
            self.x= self.img_proc.run(self.batched_imgs)
            ratio_list = self.img_proc.get_config('ratio')
        outputs = self.invoke()
        if len(self.batched_labels) > 0:
            self.score.update(self.idx, outputs, labels = self.batched_labels, ratios = ratio_list)
        else:
            self.score.update(self.idx, outputs, img_paths = self.batched_imgs, ratios = ratio_list)
        self.batched_labels.clear()
        self.batched_imgs = ''
        if (self.idx + 1) % 5 == 0:
            self.score.print_info()

    def invoke(self):
        pass

class bmodel_inference(common_inference):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        pyruntime = "pyruntime_"
        self.first = False
        self.is_cv18xx = False
        if self.args.model_file.endswith(".bmodel"):
            pyruntime = pyruntime + "bm"
            chip = get_chip_from_model(self.args.model_file)
            # trick for runtime link chip cmodel
            lib_so = 'libcmodel_1684x.so'
            if chip == 'BM1688' or chip == 'CV186X':
                lib_so = 'libcmodel_1688.so'
            elif chip == 'BM1684':
                lib_so = 'libcmodel_1684.so'
            elif chip == "BM1690":
                lib_so = 'libcmodel_bm1690.so'
            elif chip == "MARS3":
                lib_so = 'libcmodel_mars3.so'
            elif chip == "SG2380":
                lib_so = 'libcmodel_sg2380.so'
            cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel.so'.format(lib_so)
            os.system(cmd)
        elif self.args.model_file.endswith(".cvimodel"):
            pyruntime = pyruntime + "cvi"
            self.is_cv18xx = True
        else:
            raise RuntimeError("not support modle file:{}".format(self.args.model_file))
        pyruntime = importlib.import_module(pyruntime)
        self.model = pyruntime.Model(self.args.model_file)
        if not self.is_cv18xx:
            self.net = self.model.Net(self.model.networks[0])
        else:
            self.net = self.model
        self.net_input_num = len(self.net.inputs)

    def invoke(self):
        inputs = {}
        inputs[self.net.inputs[0]] = self.x
        outputs = []
        dyn_input_shapes = []
        only_one = len(inputs) == 1
        if only_one and len(self.net.inputs) != 1:
            raise RuntimeError("Input num not the same")

        for i in self.net.inputs:
            if not only_one:
                assert i.name in inputs
                input = inputs[i.name]
            else:
                input = list(inputs.values())[0]
            overflow = np.prod(i.data.shape) - np.prod(input.shape)
            if self.is_cv18xx and i.aligned:
                overflow = i.size - np.prod(input.shape)
            assert (len(i.data.shape) == len(input.shape))
            for max, dim in zip(i.data.shape, input.shape):
                if dim > max:
                    raise RuntimeError("Error shape: form {} to {}".format(
                        i.data.shape, input.shape))
            dyn_input_shapes.append(input.shape)
            input = np.concatenate([input.flatten(),
                                        np.zeros([overflow]).astype(input.dtype)]).reshape(i.data.shape)
            zp = i.qzero_point
            if i.data.dtype == input.dtype:
                i.data[:] = input.reshape(i.data.shape)
            elif i.dtype == "i8" and input.dtype == np.float32:
                data = round_away_from_zero(input * i.qscale + zp)
                i.data[:] = np.clip(data, -128, 127).astype(np.int8).reshape(i.data.shape)
            elif i.dtype == "u8" and input.dtype == np.float32:
                data = round_away_from_zero(input * i.qscale + zp)
                i.data[:] = np.clip(data, 0, 255).astype(np.uint8).reshape(i.data.shape)
            elif i.dtype == "u16" and (input.dtype == np.float32 or input.dtype == np.int32):
                i.data[:] = input.astype(np.uint16).reshape(i.data.shape)
            elif i.dtype == "f16" and input.dtype == np.float32:
                i.data[:] = input.astype(np.float16)
            elif i.dtype == "bf16" and input.dtype == np.float32:
                i.data[:] = fp32_to_bf16(input).reshape(i.data.shape)
            elif i.dtype == "i32" and (input.dtype == np.float32 or input.dtype == np.int64):
                i.data[:] = input.astype(np.int32).reshape(i.data.shape)
            elif i.dtype == "i4" and input.dtype == np.float32:
                data = round_away_from_zero(input * i.qscale + zp)
                i.data[:] = np.clip(data, -8, 7).astype(np.int8).reshape(i.data.shape)
            elif i.dtype == "u4" and input.dtype == np.float32:
                data = round_away_from_zero(input * i.qscale + zp)
                i.data[:] = np.clip(data, 0, 15).astype(np.uint8).reshape(i.data.shape)
            elif i.dtype == "f32":
                i.data[:] = input.astype(np.float32)
            else:
                raise ValueError(f"unknown type: form {input.dtype} to {i.data.dtype}")
        dyn_output_shapes = self.net.forward_dynamic(dyn_input_shapes)
        dyn_idx = 0

        for i in self.net.outputs:
            if (i.data.dtype == np.int8 or i.data.dtype == np.uint8) and i.qscale != 0:
                if self.is_cv18xx and i.name in inputs:
                    output = np.array(i.data.astype(np.float32) / np.float32(i.qscale))
                else:
                    zp = i.qzero_point
                    output = np.array((i.data.astype(np.float32) - zp) * np.float32(i.qscale),
                                            dtype=np.float32)
            elif (i.dtype == 'u16'):
                output = np.array(i.data.astype(np.float32))
            elif (i.dtype == "f16"):
                output = np.array(i.data.astype(np.float32))
            elif (i.dtype == "bf16"):
                output = bf16_to_fp32(i.data)
            else:
                output = np.array(i.data)
            if output.shape != dyn_output_shapes[dyn_idx]:
                dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                output = output.flatten()[:dyn_len].reshape(
                    *dyn_output_shapes[dyn_idx])
                dyn_idx += 1
            outputs.append(output)
        return outputs

class mlir_inference(common_inference):
    def __init__(self, args):
        super().__init__(args)

    def invoke(self):
        self.module.set_tensor(self.img_proc.input_name, self.x)
        self.module.invoke()
        all_tensors = self.module.get_all_tensor()
        outputs = []
        for i in self.module.output_names:
            outputs.append(all_tensors[i])
        return outputs

class onnx_inference(object):
    def __init__(self, args):
        self.img_proc = preprocess()
        self.img_proc.config(**vars(args))
        self.batched_labels = []
        args.batch_size = self.img_proc.batch_size
        args.net_input_dims = self.img_proc.net_input_dims
        self.batched_imgs = ''
        self.net = onnxruntime.InferenceSession(args.model_file,providers=['CPUExecutionProvider'])
        exec('from eval.postprocess_and_score_calc.{name} import {name}'.format(name = args.postprocess_type))
        self.score = eval('{}(args)'.format(args.postprocess_type))
        self.batch_size = args.batch_size
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        print('onnx batch size:', self.batch_size)

    def invoke(self):
        input_name = self.net.get_inputs()[0].name
        outs = self.net.run(None, {input_name:self.x})
        output = outs[0:1] if type(outs) == list else [outs]
        return output

class model_inference(object):
    def __init__(self, parser):
        args, _ = parser.parse_known_args()
        if args.postprocess_type == 'topx':
            from eval.postprocess_and_score_calc.topx import score_Parser
        elif args.postprocess_type == 'coco_mAP':
            from eval.postprocess_and_score_calc.coco_mAP import score_Parser
        else:
            print('postprocess_type error')
            exit(1)
        new_parser = argparse.ArgumentParser(
            parents=[parser, score_Parser().parser],
            conflict_handler='resolve')
        args = new_parser.parse_args()
        if args.model_file.endswith('.mlir'):
            engine = mlir_inference(args)
        elif args.model_file.endswith(".bmodel") or args.model_file.endswith(".cvimodel"):
            engine = bmodel_inference(args)
        elif args.model_file.endswith('.onnx'):
            parser = get_preprocess_parser(existed_parser=new_parser)
            parser.add_argument("--input_shapes", type=str, help="input_shapes")
            args = parser.parse_args()
            engine = onnx_inference(args)
        else:
            print('model_file:{}, ext_name error'.format(args.model_file))
            exit(1)
        self.engine = engine

    def run(self, idx, img_path, target = None):
        self.engine.run(idx, img_path, target)

    def get_result(self):
        self.engine.get_result()
