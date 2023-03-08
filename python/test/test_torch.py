#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from copy import deepcopy
from re import T
import numpy as np
from typing import List, Union

from tools.model_runner import mlir_inference, model_inference, torch_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

Failed_Cases = ["PRelu", "Gather", "Conv2d", "LayerNorm", "Elu"]


class TORCH_IR_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self, chip: str = "bm1684x", mode: str = "all"):
        self.test_function = {
            #############################
            # Torch Test Case, Alphabetically
            #############################
            "Add": self.test_Add,
            "Concat": self.test_Concat,
            "ConstantPad1d": self.test_Pad1d,
            "ConstantPad2d": self.test_Pad2d,
            "Conv2d": self.test_Conv2d,
            "Div": self.test_Div,
            "Elu": self.test_Elu,
            # "Gather": self.test_Gather,
            "LayerNorm": self.test_LayerNorm,
            "Mul": self.test_Mul,
            "PRelu": self.test_PRelu,
            "Permute": self.test_Permute,
            "ReflectionPad1d": self.test_Pad1d,
            "ReflectionPad2d": self.test_Pad2d,
            "Relu": self.test_Relu,
            "ReplicationPad1d": self.test_Pad1d,
            "ReplicationPad2d": self.test_Pad2d,
            "Sub": self.test_Sub,
            "T": self.test_T,
            "Transpose": self.test_Transpose,
            "ZeroPad2d": self.test_Pad2d,
        }
        self.support_quant_modes = ["f32", "f16", "bf16"]
        #self.support_quant_modes = ["f32", "f16", "bf16", "int8", "int4"]
        self.support_asym = [True, False]
        self.model_file = ".bmodel"
        self.is_cv18xx = False
        self.chip = chip.lower()
        # self.dynamic = dynamic
        if self.chip.startswith("cv18"):
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
            self.model_file = ".cvimodel"
            self.is_cv18xx = True
        elif self.chip == "bm1684":
            self.support_quant_modes = ["f32", "int8"]
            self.support_asym = [False]
        self.mode = mode.lower()
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    def test_single(self, case: str):
        np.random.seed(0)
        if case in self.test_function:
            print("Test: {}".format(case))
            self.test_function[case](case)
            print("====== TEST {} Success ======".format(case))
        else:
            self.list()

    def check_support(self, case):
        if case in Failed_Cases:
            return False
        return True

    def list(self):
        print("====== All Support Ops ======")
        for case in self.test_function:
            if case not in Failed_Cases:
                print(case)
        print("====== Error Ops ======")
        for case in self.test_function:
            if case in Failed_Cases:
                print(case)

    def square_rooted(self, x):
        return np.sqrt(sum([a * a for a in x]))

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def compare(self, ref_out, targe_out):
        if ref_out.dtype in [np.int64, np.int32, np.int16, np.int8]:
            cos = self.cosine_similarity(ref_out, targe_out)
            assert (cos > 0.999)
        else:
            np.testing.assert_allclose(ref_out, targe_out, rtol=1e-5, atol=1e-01)

    def make_test_calibration_table(self, tensors, table_name):
        # simple calibration table
        with open(table_name, 'w') as f:
            for name in tensors:
                flatten_tensor = tensors[name].flatten()
                max_val = max(flatten_tensor)
                min_val = min(flatten_tensor)
                if max_val == min_val:
                    max_val = max_val + 0.01
                t = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
                f.write("{} {} {} {}\n".format(name, t, min_val, max_val))

    def create_random_input(self, shapes):
        inputs = [np.clip(np.random.randn(*s).astype(np.float32), -10, 10) for s in shapes]
        return [torch.from_numpy(inp) for inp in inputs]

    def torch_convert(self, in_shapes, torch_model, model_name: str):
        # torch --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)

        tool = TorchTransformer(model_name, torch_model, input_shapes=in_shapes)
        tool.model_transform(fp32_mlir)

        input_npz = "{}_ref_in_fp32.npz".format(model_name)
        ref_npz = model_name + '_ref_outputs.npz'
        input_data = {}
        for idx, name in enumerate(tool.converter.input_names):
            input_data[name] = np.random.random(size=in_shapes[idx]).astype(np.float32)
        np.savez(input_npz, **input_data)
        # # top mlir outputs will be inferenced first in case the quant mode is int8
        show_fake_cmd(input_npz, torch_model, ref_npz)
        torch_outs = torch_inference(input_data, torch_model, True)
        np.savez(ref_npz, **torch_outs)
        show_fake_cmd(input_npz, fp32_mlir, "top_out.npz")
        top_mlir_outs = mlir_inference(input_data, fp32_mlir, True)
        return (torch_outs, top_mlir_outs, input_npz)

    def bmodel_generate(self,
                        model_name: str,
                        top_mlir_outs: dict,
                        quant_mode: str,
                        isAsym: bool = False):
        table_name = None
        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        if quant_mode == "int8":
            tpu_mlir += "_asym" if isAsym else "_sym"
            table_name = "{}_cali_table".format(model_name)
            self.make_test_calibration_table(top_mlir_outs, table_name)

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      cali_table=table_name,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        bmodel = tpu_mlir + ".bmodel"
        mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final)

        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              torch_output: dict,
                              tpu_mlir: str,
                              bmodel: str,
                              input_npz: str,
                              quant_mode: str,
                              model_name: str,
                              isAsym: bool = False):
        ref_tpu_tolerance = "0.9,0.9"
        input_data = np.load(input_npz)
        # save ref
        ref_npz = "{}_ref_outputs.npz".format(model_name)
        # tpu mlir inference and compare
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        show_fake_cmd(input_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(ref_npz, **torch_output)
        np.savez(tpu_npz, **tpu_mlir_outs)
        npz_compare([ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel inference and compare
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)
        print("[Success] test {} {}".format(model_name, msg))

    def convert_torch_and_compare(
        self,
        in_shapes,
        model_name,
        torch_model,
    ):
        """Generic function to generate and compare torch and Tpu-Mlir output"""
        model_def = model_name + ".pt"
        inputs = self.create_random_input(in_shapes)
        jit.trace(torch_model, inputs).save(model_def)
        torch_outs, top_mlir_outs, input_npz = self.torch_convert(in_shapes, model_def, model_name)
        # test onnx and mlir outputs
        counter = 0
        for name in torch_outs:
            if name in top_mlir_outs:
                print("Compare mlir and torch:{}\n".format(name))
                top_mlir_output = top_mlir_outs[name].flatten()
                onnx_output = torch_outs[name].flatten()
                self.compare(onnx_output, top_mlir_output)
                counter += 1
        if counter == 0:
            raise RuntimeError("No compare between torch outs and mlir outts")
        print("Success: Torch outs and Mlir outs are equal\n")
        for quant_mode in self.quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode,
                                                            isAsym)
                    self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz,
                                               quant_mode, model_name, isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode)
                self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz, quant_mode,
                                           model_name)

    #######################################################################
    # Convolution
    # ------------
    def test_Conv2d(self, case_name):
        """Conv 2D"""

        def case1(suffix: str = ""):

            class Net(torch.nn.Module):

                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = nn.Conv2d(8, 8, 3, 1, 1)
                    self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)

                def forward(self, x):
                    y = self.conv1(x)
                    z = self.conv2(x)
                    return y + z

            self.convert_torch_and_compare([(4, 8, 28, 28)], case_name + suffix, Net().eval())

        def case2(input_shape,
                  kernel_shape,
                  oc,
                  has_bias=False,
                  padding: Union[int, str, List[int]] = 0,
                  stride: Union[int, List[int]] = 1,
                  dilation: Union[int, List[int]] = 1,
                  group=1,
                  suffix: str = ""):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    filter_shape = (oc, input_shape[1] // group, kernel_shape[0], kernel_shape[1])
                    self.filter = torch.randn(filter_shape)
                    self.bias = torch.randn(oc) if has_bias else None

                def forward(self, x):
                    y = F.conv2d(x,
                                 self.filter,
                                 bias=self.bias,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 groups=group)
                    return y

            self.convert_torch_and_compare([input_shape], case_name + suffix, Model().eval())

        case1("_1")
        case2((1, 3, 32, 32), (3, 3), 12, has_bias=True, group=1, padding="same", suffix="_2")
        case2((2, 32, 16, 16), (5, 5), 64, padding=2, stride=2, dilation=1, suffix="_3")
        case2((1, 3, 32, 32), (3, 3), 12, group=3, padding=(1, 1), stride=(2, 1), suffix="_4")

    #######################################################################
    # Binary Base
    # ------------
    def _test_binary(self, case_name, op_type, in0_shape, in1_shape, alpha=None):

        _alpha = {}
        if alpha:
            _alpha = dict(alpha=alpha)

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.randn(in1_shape)

            def forward(self, x):
                y0 = x + 1
                y1 = op_type(self.weight, y0, **_alpha)
                y2 = op_type(y0, y1, **_alpha)
                return y2

        self.convert_torch_and_compare([in0_shape], case_name, Model().eval())

    #######################################################################
    # Add
    # ------------
    def test_Add(self, case_name):
        """Add"""

        self._test_binary(case_name, torch.add, (1, 3, 32, 32), (1, 3, 32, 32), 3)
        self._test_binary(case_name, torch.add, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(case_name, torch.add, (32, 32), (32))

    #######################################################################
    # Sub
    # ------------
    def test_Sub(self, case_name):
        """Sub"""

        self._test_binary(case_name, torch.sub, (1, 3, 32, 31), (1, 3, 32, 1), 3)
        self._test_binary(case_name, torch.sub, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(case_name, torch.sub, (32, 32), (32))

    #######################################################################
    # Mul
    # ------------
    def test_Mul(self, case_name):
        """Mul"""

        self._test_binary(case_name, torch.multiply, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(case_name, torch.multiply, (2, 32, 16), (2, 1, 16))
        self._test_binary(case_name, torch.multiply, (32, 32), (32))

    #######################################################################
    # Div
    # ------------
    def test_Div(self, case_name):
        """Div"""

        self._test_binary(case_name, torch.div, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(case_name, torch.div, (2, 32, 16), (2, 1, 16))
        self._test_binary(case_name, torch.div, (32, 32), (32))

    #######################################################################
    # LayerNorm
    # ------------
    def test_LayerNorm(self, case_name):
        normalize_shape = [40, 80]

        class Net(torch.nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.layer_norm = nn.LayerNorm(normalize_shape, elementwise_affine=True)
                self.prelu = nn.PReLU()

            def forward(self, x):
                x = self.layer_norm(x)
                y = self.prelu(x)
                return y

        input_shape = [14, 25] + normalize_shape
        self.convert_torch_and_compare([input_shape], case_name, Net().eval())

    #######################################################################
    # PRelu
    # ------------
    def test_PRelu(self, case_name):
        """PRelu"""

        def _test_prelu(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    # self.weight0 = torch.randn(1)
                    self.weight1 = torch.randn(input_shape[1])

                def forward(self, x):
                    # y0 = F.prelu(x, self.weight0)
                    y1 = F.prelu(x, self.weight1)
                    return y1

            self.convert_torch_and_compare([input_shape], case_name, Model().eval())

        _test_prelu((1, 3, 32, 32))
        _test_prelu((2, 32, 16))
        _test_prelu((32, 32))

    #######################################################################
    # Gather
    # ------------
    # def test_Gather(self, case_name):
    #   """Gather"""
    #   def _test_gather(in0_shape, in1_shape, dim=None):
    #     class Model(nn.Module):
    #       def __init__(self):
    #           super(Model, self).__init__()
    #       def forward(self, x):
    #           # if dim is None:
    #           #   y1 = torch.concat((x, self.weight))
    #           # else:
    #           #   y1 = torch.concat((x, self.weight), dim=dim)
    #           return
    #     self.convert_torch_and_compare([in0_shape], case_name, Model().eval())

    #   _test_gather((1, 3, 32, 32), (1,6,32,32), 1)
    #   _test_gather((2, 32, 16), (3, 32, 16))
    #   _test_gather((32, 32), (1, 32), 0)

    #######################################################################
    # Permute
    # ------------
    def test_Permute(self, case_name):
        """Permute"""

        def _test_permute(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.permute(x, dims=dims)
                    return y1

            self.convert_torch_and_compare([in_shape], case_name, Model().eval())

        _test_permute((1, 3, 32, 32), (0, 3, 1, 2))
        _test_permute((2, 32, 16), (2, 0, 1))
        _test_permute((32, 32), (1, 0))

    #######################################################################
    # T
    # ------------
    def test_T(self, case_name):
        """T"""

        def _test_t(in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    x = torch.concat((x, x))
                    x = torch.t(x)
                    y1 = torch.concat((x, x))
                    return y1

            self.convert_torch_and_compare([in_shape], case_name, Model().eval())

        _test_t((32, 32))
        _test_t((32, ))

    #######################################################################
    # Transpose
    # ------------
    def test_Transpose(self, case_name):
        """Transpose"""

        def _test_transpose(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.transpose(x, dim0=dims[0], dim1=dims[1])
                    return y1

            self.convert_torch_and_compare([in_shape], case_name, Model().eval())

        _test_transpose((1, 3, 32, 32), (0, 3))
        _test_transpose((2, 32, 16), (2, 0))
        _test_transpose((32, 32), (1, 0))

    #######################################################################
    # Concat
    # ------------
    def test_Concat(self, case_name):
        """Concat"""

        def _test_concat(in0_shape, in1_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    if dim is None:
                        y1 = torch.concat((x, self.weight))
                    else:
                        y1 = torch.concat((x, self.weight), dim=dim)
                    return y1

            self.convert_torch_and_compare([in0_shape], case_name, Model().eval())

        _test_concat((1, 3, 32, 32), (1, 6, 32, 32), 1)
        _test_concat((2, 32, 16), (3, 32, 16))
        _test_concat((32, 32), (1, 32), 0)

    #######################################################################
    # Elu
    # ------------
    def test_Elu(self, case_name):
        """Elu"""

        def _test_elu(input_shape, alpha=1.0):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.elu(x)

            self.convert_torch_and_compare([input_shape], case_name, Model().eval())

        _test_elu((1, 3, 32, 32), 2.0)
        _test_elu((3, 16, 32), 3.5)
        _test_elu((64, 32))

    #######################################################################
    # Relu
    # ------------
    def test_Relu(self, case_name):
        """Relu"""

        def _test_relu(in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return torch.relu(x)

            self.convert_torch_and_compare([in_shape], case_name, Model().eval())

        _test_relu((1, 3, 32, 32))
        _test_relu((3, 16, 32))
        _test_relu((64, 32))

    #######################################################################
    # Pad1D
    # ------------
    def test_Pad1d(self, case_name):
        """Pad 1D (ReflectionPad1d, ReplicationPad1d, ConstantPad1d)
        Input: (N,C,W) or (C, W)
        Padding: int (pad_left == pad_right) or tuple (pad_left, pad_right)
      """

        def _test_pad1d(input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pad1d = nn.ConstantPad1d(padding, value)
                    if case_name == "ReflectionPad1d":
                        self.pad1d = nn.ReflectionPad1d(padding)
                    elif case_name == "ReplicationPad1d":
                        self.pad1d = nn.ReplicationPad1d(padding)

                def forward(self, x):
                    return self.pad1d(x)

            self.convert_torch_and_compare([input_shape], case_name, Model().eval())

        _test_pad1d((1, 3, 32), 5, 3.0)
        _test_pad1d((2, 3, 32), (5, 3))
        _test_pad1d((64, 32), (3, 6), 5.0)
        _test_pad1d((64, 32), 7, 6.0)

    #######################################################################
    # Pad2D
    # ------------
    def test_Pad2d(self, case_name):
        """Pad 2D (ReflectionPad2d, ReplicationPad2d, ConstantPad2d, ZeroPad2d)
        Input: (C, H, W) or (N, C, H, W)
        Padding: int (pad_left == pad_right == pad_top == pad_bottom)
                  or tuple (pad_left, pad_right, pad_top, pad_bottom)
      """

        def _test_pad2d(input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pad2d = nn.ConstantPad2d(padding, value)
                    if case_name == "ReflectionPad2d":
                        self.pad2d = nn.ReflectionPad2d(padding)
                    elif case_name == "ReplicationPad2d":
                        self.pad2d = nn.ReplicationPad2d(padding)
                    elif case_name == "ZeroPad2d":
                        self.pad2d = nn.ZeroPad2d(padding)

                def forward(self, x):
                    return self.pad2d(x)

            self.convert_torch_and_compare([input_shape], case_name, Model().eval())

        _test_pad2d((1, 16, 32), 7, 3.0)
        _test_pad2d((3, 16, 32), (4, 6, 7, 8))
        _test_pad2d((1, 3, 16, 32), 3, 4.0)
        _test_pad2d((2, 4, 16, 32), (3, 4, 5, 6), 6.0)


def test_one_case_in_all(tester: TORCH_IR_TESTER, case, error_cases, success_cases):
    try:
        tester.test_single(case)
    except:
        error_cases.append(case)
        return
    success_cases.append(case)


def test_all(tester: TORCH_IR_TESTER):
    import multiprocessing
    process_number = multiprocessing.cpu_count() // 2 + 1
    processes = []
    error_cases = multiprocessing.Manager().list()
    success_cases = multiprocessing.Manager().list()
    for case in tester.test_function:
        if tester.check_support(case):
            p = multiprocessing.Process(target=test_one_case_in_all,
                                        args=(tester, case, error_cases, success_cases))
            processes.append(p)
        if len(processes) == process_number:
            for p in processes:
                p.start()
            for j in processes:
                j.join()
            processes = []
    if processes:
        for p in processes:
            p.start()
        for j in processes:
            j.join()
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_torch.py --chip {} TEST Failed ======".format(tester.chip))
        exit(1)
    else:
        print("====== test_torch.py --chip {} TEST Success ======".format(tester.chip))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'int8'],
                        help="chip platform name")
    # yapf: enable
    args = parser.parse_args()
    tester = TORCH_IR_TESTER(args.chip, args.mode)
    dir = "torch_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
