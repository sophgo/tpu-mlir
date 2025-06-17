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
from tools.model_runner import mlir_inference, model_inference, tflite_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
from utils.auto_remove import file_mark, file_clean, clean_kmp_files
from utils.timer import Timer
from utils.regression_logger import run_in_log_wrapper
import os

# from tflite.BuiltinOperator import BuiltinOperator
try:
    import tensorflow.compat.v1 as tf

    # tensorflow.python.framework.ops module itself is not part of
    # TensorFlow's public API: the precise contents of that module
    # may vary from one version to the next
    import tensorflow.compat.v1 as ops
except ImportError:
    import tensorflow as tf
    import tensorflow as ops
from tensorflow.python.framework import constant_op

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variables

try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

Failed_Cases = ["Add", "Cast", "Gather", "ReduceMin", "Deconv2d", "ReduceMax", "Sum", "Matmul"]


class TFLITE_IR_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self, chip: str = "bm1684x", mode: str = "all", concise_log: bool = False):
        self.test_function = {
            #######################################
            # TfLite Test Case, Alphabetically
            #######################################
            "Add": self.test_Add,
            "AveragePool2d": self.test_Average_Pool2d,
            "Cast": self.test_Cast,
            "Concat": self.test_Concat,
            "Conv2d": self.test_Conv2d,
            "Deconv2d": self.test_Deconv2d,
            "DepthwiseConv2d": self.test_Depthwise_Conv2d,
            "Dequant": self.test_Max_Pool2d,
            "Gather": self.test_Gather,
            "Matmul": self.test_Matmul,
            "MaxPool2d": self.test_Max_Pool2d,
            "Mean": self.test_Mean,
            "Mul": self.test_Mul,
            "Pack": self.test_Pack,
            "Pad": self.test_Pad,
            "ReduceMax": self.test_Redcue_Max,
            "ReduceMin": self.test_Redcue_Min,
            "Requant": self.test_Max_Pool2d,
            "Reshape": self.test_Reshape,
            "Sigmoid": self.test_Sigmoid,
            "Softmax": self.test_Softmax,
            "Split": self.test_Split,
            "StridedSlice": self.test_Strided_slice,
            "StridedSliceMerge": self.test_Strided_slice_merge,
            "Sub": self.test_Sub,
            "Sum": self.test_Sum,
            "Transpose": self.test_Transpose,
            "Unpack": self.test_Unpack,
        }
        # no quantization when quant_mode == "f32"
        self.support_quant_modes = ["int8"]
        self.support_chip = ["bm1684x", 'bm1688']
        self.concise_log = concise_log  # use when run regression/main_entry.py
        if chip not in self.support_chip:
            raise RuntimeError("{} not support tflite now.".format(self.chip))
        self.chip = chip.lower()
        if mode == "" or mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [mode]

    @run_in_log_wrapper
    def test_single(self, case: str):
        np.random.seed(0)
        if case in self.test_function:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
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

    def create_placeholder(self, shape, dtype=np.float32, min_value=1.0, max_value=10.0, name=None):
        data = np.random.uniform(min_value, max_value, shape).astype(dtype)
        tensor = tf.placeholder(shape=data.shape, dtype=data.dtype, name=name)
        return tensor, data

    def create_constant(self, shape, dtype=np.float32, min_value=1.0, max_value=10.0):
        data = np.random.uniform(min_value, max_value, shape).astype(dtype)
        tensor = tf.constant(data, dtype)
        return tensor

    def gen_input(self, shapes, range):
        datas = []
        inputs = []
        input_range = {}
        index = 0
        for s in shapes:
            name = "in_" + str(index)
            input, data = self.create_placeholder(s,
                                                  min_value=range[0],
                                                  max_value=range[1],
                                                  name=name)
            datas.append(data)
            inputs.append(input)
            input_range[name] = (range[0], range[1])
            index += 1
        return datas, inputs, input_range

    def tflite_convert(self, input_data: dict, tflite_model, model_name: str, need_transpose=False):
        # tflite --> mlir conversion (origin and optimized mlir models will be generated and saved)
        def nhwc2nchw(x):
            if x.ndim == 4:
                return x.transpose([0, 3, 1, 2])
            return x

        fp32_mlir = "{}.mlir".format(model_name)
        preprocessor = {}
        preprocessor['channel_format'] = 'nhwc' if need_transpose else "none"
        input_is_nchw = not need_transpose
        layout = not need_transpose
        tool = TFLiteTransformer(model_name, tflite_model, preprocessor=preprocessor)
        tool.model_transform(fp32_mlir)

        input_npz = "{}_ref_in_fp32.npz".format(model_name)
        file_mark(input_npz)
        ref_npz = model_name + '_ref_outputs.npz'
        for name in input_data:
            if input_data[name].dtype in [np.int64, np.int32]:
                input_data[name] = input_data[name].astype(np.int32)
            else:
                input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        # top mlir outputs will be inferenced first in case the quant mode is int8
        show_fake_cmd(input_npz, tflite_model, ref_npz)
        tflite_outs = tflite_inference(input_data,
                                       tflite_model,
                                       True,
                                       input_is_nchw=False,
                                       tf_layout=layout)
        if not input_is_nchw:
            input_data = {k: nhwc2nchw(v) for k, v in input_data.items()}
            input_npz = "{}_in_fp32.npz".format(model_name)
            file_mark(input_npz)
            np.savez(input_npz, **input_data)
        return (tflite_outs, input_npz)

    def bmodel_generate(self, model_name: str, quant_mode: str, isAsym: bool = False):

        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        if quant_mode == "int8":
            tpu_mlir += "_asym" if isAsym else "_sym"

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        bmodel = tpu_mlir + ".bmodel"
        mlir_to_model(tpu_mlir=tpu_mlir + ".mlir", bmodel_path=bmodel, final_mlir=tpu_final)

        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              tflite_output: dict,
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
        file_mark(ref_npz)
        # tpu mlir inference and compare
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        file_mark(tpu_npz)
        show_fake_cmd(input_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(ref_npz, **tflite_output)
        np.savez(tpu_npz, **tpu_mlir_outs)
        npz_compare([ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel inference and compare
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        file_mark(model_npz)
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)
        print("[Success] test {} {}".format(model_name, msg))

    def convert_to_list(self, x):
        if not isinstance(x, list):
            x = [x]
        return x

    def convert_tflite_and_compare(self,
                                   in_data,
                                   model_name,
                                   tflite_model_quant,
                                   need_transpose=False):
        """Generic function to generate and compare TFLite and Tpu-Mlir output"""
        from transform.tflite.Model import Model
        tflite_model = Model.GetRootAsModel(tflite_model_quant, 0)
        subgraph = tflite_model.Subgraphs(0)
        model_input = subgraph.InputsAsNumpy()
        in_node = [subgraph.Tensors(input).Name().decode("utf-8") for input in model_input]
        model_def = model_name + ".tflite"
        open(model_def, "wb").write(tflite_model_quant)

        input_data = dict()
        for i in range(len(in_node)):
            input_data[in_node[i]] = in_data[i]
        tflite_outs, input_npz = self.tflite_convert(input_data, model_def, model_name,
                                                     need_transpose)
        for quant_mode in self.quant_modes:
            tpu_mlir, bmodel = self.bmodel_generate(model_name, quant_mode, True)
            self.inference_and_compare(tflite_outs, tpu_mlir, bmodel, input_npz, quant_mode,
                                       model_name, True)

    def _quantize_sess_model(
        self,
        input_tensors,
        output_tensors,
        input_data=None,
        init_global_variables=False,
        quantized=False,
        input_range=None,
        experimental_new_converter=False,
        fp16_quantized=False,
        int_quant_dtype=tf.int8,
    ):
        if input_data is not None:
            # To create quantized values with dynamic range of activations, needs representative dataset
            def representative_data_gen():
                for _ in range(1):
                    yield [input_data]

        with tf.Session() as sess:
            if init_global_variables:
                sess.run(variables.global_variables_initializer())
            # convert to tflite model
            converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)
            converter.experimental_new_converter = experimental_new_converter
            if quantized:
                if int_quant_dtype == tf.int16:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                    ]
                else:
                    # default to uint8 quantization
                    converter.inference_type = tf.lite.constants.QUANTIZED_UINT8

                input_arrays = converter.get_input_arrays()
                input_stats = {}
                # calculate the mean and quantization scale for every input tensor,
                # with respect to its fp32 input range, defined in fake_quant.
                # scale = 255 / (fmax - fmin);  mean = -fmin * scale (the zero point)
                for i in input_arrays:
                    try:
                        quant_scale = 255 / (input_range[i][1] - input_range[i][0])
                    except ZeroDivisionError:
                        print("Min and max of the input range for tensor " + i + " can't be equal")
                    mean = -input_range[i][0] * quant_scale
                    input_stats[i] = (mean, quant_scale)
                converter.quantized_input_stats = input_stats
            elif fp16_quantized:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            tflite_model_buffer = converter.convert()
        return tflite_model_buffer

    def gen_keras_model(self, nn_func, input_shape, *args, **kwargs):
        data = np.random.uniform(0, 1, input_shape).astype("float32")
        data_in = tf.keras.layers.Input(shape=data.shape[1:])
        pool = nn_func(*args, **kwargs)(data_in)
        keras_model = tf.keras.models.Model(data_in, pool)
        return keras_model, data

    def _quantize_keras_model(
        self,
        keras_model,
        input_data,
        is_float_input=False,
        is_float_output=False,
        int_quant_dtype=tf.int8,
    ):
        """Utility function to quantize a Keras model using TFLite converter."""

        # import tflite
        # To create quantized values with dynamic range of activations, needs representative dataset
        def representative_data_gen():
            for _ in range(1):
                yield [input_data]

        converter = interpreter_wrapper.TFLiteConverter.from_keras_model(keras_model)
        if int_quant_dtype == tf.int8:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            inference_dtype = tf.uint8
        elif int_quant_dtype == tf.int16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            inference_dtype = tf.uint16
        else:
            raise RuntimeError(
                f"Invalid quantized dtype {int_quant_dtype}. Supported types: int8, int16.")

        # NOTE: If representative dataset is provided, and inference input type is not set,
        #       then converter will self add quant & dequant Op accordingly.
        if not is_float_input:
            converter.inference_input_type = inference_dtype
        if not is_float_output:
            converter.inference_output_type = inference_dtype

        return converter.convert()

    ##################################
    # adding operators from here
    ##################################
    def with_fused_activation_function(self, input_tensor, fn_name):
        """Fused activation function"""
        if fn_name is None or fn_name == "NONE":
            return input_tensor
        if fn_name == "RELU":
            return nn_ops.relu(input_tensor)
        if fn_name == "RELU6":
            return nn_ops.relu6(input_tensor)
        if fn_name == "RELU_N1_TO_1":
            return math_ops.maximum(-1, math_ops.minimum(input_tensor, 1))
        if fn_name == "TANH":
            return math_ops.tanh(input_tensor)
        raise AssertionError(f"Unknown fused_activation_function {fn_name}")

    #######################################################################
    # Element-wise
    # ------------
    def _test_elemwise(
        self,
        math_op,
        data,
        fused_activation_function=None,
        op_name="",
        same_params=False,
    ):
        """One iteration of elemwise"""

        assert len(data) == 2

        def _test_elemwise_out_range(op):
            # set the fake_quant output range with respect to the input tensors float32 range
            out_range = {
                "Add": (-150, 150),
                "Sub": (-150, 150),
                "Mul": (-5e3, 5e3),
                "Maximum": (-112, 111),
                "Minimum": (-128, 127),
                "Equal": (-150, 150),
                "Greater": (-150, 150),
            }
            return out_range[op]

        def __test_elemwise(in_data):
            assert len(in_data) == 2
            # set the fp32 output range with respect to the operation
            out_min, out_max = _test_elemwise_out_range(op_name)
            inq0_min, inq0_max = (-100, 100)
            inq1_min, inq1_max = (-50, 50)

            # if requested use same quantization parameters provided by _test_elemwise_qnn_out_range
            if same_params:
                inq0_min, inq0_max = (out_min, out_max)
                inq1_min, inq1_max = (out_min, out_max)

            # fake_quant will keep the tensors in float32 until the conversion in the session
            inq_data = [
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[0], min=out_min, max=out_max, name="inq_0")
                if in_data[0] is not None else tf.quantization.fake_quant_with_min_max_args(
                    data[0], min=out_min, max=out_max, name="const_tensor0"),
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[1], min=out_min, max=out_max, name="inq_1")
                if in_data[1] is not None else tf.quantization.fake_quant_with_min_max_args(
                    data[1], min=out_min, max=out_max, name="const_tensor1"),
            ]

            input_range = {
                x[1][0]: x[1][1]
                for x in zip(in_data, (("inq_0", (inq0_min, inq0_max)),
                                       ("inq_1", (inq1_min, inq1_max)))) if x[0] is not None
            }

            out = math_op(inq_data[0], inq_data[1])
            out = self.with_fused_activation_function(out, fused_activation_function)
            out = tf.quantization.fake_quant_with_min_max_args(out,
                                                               min=out_min,
                                                               max=out_max,
                                                               name="out")

            # Note same_params uses experimental_new_converter as toco failed
            model_def = self._quantize_sess_model(
                [x[1] for x in zip(in_data, inq_data) if x[0] is not None],
                [out],
                quantized=True,
                input_range=input_range,
                experimental_new_converter=same_params,
            )
            self.convert_tflite_and_compare([x[1] for x in zip(in_data, data) if x[0] is not None],
                                            model_name,
                                            model_def,
                                            need_transpose=True)

        # Test with two tensors
        with tf.Graph().as_default():
            model_name = op_name + "_test0"
            __test_elemwise(in_data=[
                array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in_0"),
                array_ops.placeholder(shape=data[1].shape, dtype="float32", name="in_1"),
            ])
        # Test with tensor and constant
        with tf.Graph().as_default():
            model_name = op_name + "_test1"
            __test_elemwise(in_data=[
                array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in_0"), None
            ])
        # Test with constant and tensor
        with tf.Graph().as_default():
            model_name = op_name + "_test2"
            __test_elemwise(in_data=[
                None,
                array_ops.placeholder(shape=data[1].shape, dtype="float32", name="in_1")
            ])

    def _test_quant_elemwise(self, math_op, case_name):
        data = [
            np.array(np.random.uniform(0, 255, (2, 8, 3, 1)), dtype=np.uint8),
            np.array(np.random.uniform(0, 255, (2, 8, 1, 3)), dtype=np.uint8),
        ]
        self._test_elemwise(
            math_op,
            data,
            op_name=case_name,
        )
        data = [
            np.array(np.random.uniform(0, 255, (2, 4, 3)), dtype=np.uint8),
            np.array(np.random.uniform(0, 255, (2, 4, 3)), dtype=np.uint8),
        ]
        self._test_elemwise(math_op, data, op_name=case_name, fused_activation_function="RELU")
        data = [
            np.array(np.random.uniform(0, 255, (2, 4)), dtype=np.uint8),
            np.array(np.random.uniform(0, 255, (1, 4)), dtype=np.uint8),
        ]
        self._test_elemwise(math_op, data, op_name=case_name, fused_activation_function="RELU")
        data = [
            np.array(np.random.uniform(0, 255, (1, 1, 1)), dtype=np.uint8),
            np.array(np.random.uniform(0, 255, (2, 4, 3)), dtype=np.uint8),
        ]
        self._test_elemwise(math_op, data, op_name=case_name)

    def test_Add(self, case_name):
        """Add"""
        self._test_quant_elemwise(math_ops.add, case_name)

    def test_Sub(self, case_name):
        """Sub"""
        self._test_quant_elemwise(math_ops.sub, case_name)

    def test_Mul(self, case_name):
        """Mul"""
        self._test_quant_elemwise(math_ops.mul, case_name)

    #######################################################################
    # Convolution
    # ------------
    def test_Conv2d(self, case_name):
        """Conv 2D"""

        def _test_convolution(input_shape,
                              kernel_shape,
                              filters,
                              padding="valid",
                              data_format="NHWC",
                              int_quant_dtype=tf.int8):
            data_format = "channels_last" if data_format == "NHWC" else "channels_first"
            keras_model, data = self.gen_keras_model(
                tf.keras.layers.Conv2D,
                input_shape,
                filters=filters,
                kernel_size=(kernel_shape[0], kernel_shape[1]),
                activation=tf.nn.relu,
                padding=padding,
                data_format=data_format,
            )

            model_def = self._quantize_keras_model(
                keras_model,
                data,
                is_float_input=True,
                is_float_output=True,
                int_quant_dtype=int_quant_dtype,
            )
            self.convert_tflite_and_compare([data], case_name, model_def, need_transpose=True)

        _test_convolution((1, 28, 28, 3), (1, 1), 12)
        _test_convolution((1, 3, 32, 32), (3, 3), 12, data_format="NCHW")

    #######################################################################
    # Ddconvolution
    # ------------
    def test_Deconv2d(self, case_name):
        """Deconv 2D"""

        def _test_deconvolution(input_shape,
                                kernel_shape,
                                filters,
                                padding="valid",
                                data_format="NHWC",
                                int_quant_dtype=tf.int8):
            data_format = "channels_last" if data_format == "NHWC" else "channels_first"
            keras_model, data = self.gen_keras_model(
                tf.keras.layers.Conv2DTranspose,
                input_shape,
                filters=filters,
                kernel_size=(kernel_shape[0], kernel_shape[1]),
                activation=tf.nn.relu,
                padding=padding,
                data_format=data_format,
            )

            model_def = self._quantize_keras_model(
                keras_model,
                data,
                is_float_input=True,
                is_float_output=True,
                int_quant_dtype=int_quant_dtype,
            )
            self.convert_tflite_and_compare([data], case_name, model_def, need_transpose=True)

        _test_deconvolution((1, 28, 28, 3), (1, 1), 12)
        _test_deconvolution((1, 3, 32, 32), (3, 3), 12, data_format="NCHW")

    #######################################################################
    # Depthwise Convolution
    # ------------
    def test_Depthwise_Conv2d(self, case_name):
        """Depthwise 2D"""

        def _test_depthwise_conv2d(input_shape,
                                   kernel_shape,
                                   filters=1,
                                   padding="valid",
                                   data_format="NHWC",
                                   int_quant_dtype=tf.int8):
            data_format = "channels_last" if data_format == "NHWC" else "channels_first"
            keras_model, data = self.gen_keras_model(
                tf.keras.layers.DepthwiseConv2D,
                input_shape,
                depth_multiplier=filters,
                kernel_size=(kernel_shape[0], kernel_shape[1]),
                activation=tf.nn.relu,
                padding=padding,
                data_format=data_format,
            )

            model_def = self._quantize_keras_model(
                keras_model,
                data,
                is_float_input=True,
                is_float_output=True,
                int_quant_dtype=int_quant_dtype,
            )
            self.convert_tflite_and_compare([data], case_name, model_def, need_transpose=True)

        _test_depthwise_conv2d((1, 28, 28, 3), (1, 1), 3)
        _test_depthwise_conv2d((1, 3, 32, 32), (3, 3), data_format="NCHW")

    #######################################################################
    # Average Pooling 2D
    # ------------
    def test_Average_Pool2d(self, case_name):
        """Average Pooling 2D"""

        def _test_average_pool2d(input_shape,
                                 pool_size,
                                 strides=None,
                                 padding="valid",
                                 data_format="NHWC",
                                 int_quant_dtype=tf.int8):
            data_format = "channels_last" if data_format == "NHWC" else "channels_first"
            keras_model, data = self.gen_keras_model(
                tf.keras.layers.AveragePooling2D,
                input_shape,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

            model_def = self._quantize_keras_model(
                keras_model,
                data,
                is_float_input=True,
                is_float_output=True,
                int_quant_dtype=int_quant_dtype,
            )
            self.convert_tflite_and_compare([data], case_name, model_def, need_transpose=True)

        _test_average_pool2d(
            (1, 28, 28, 3),
            (2, 2),
        )
        _test_average_pool2d((1, 32, 32, 3), (3, 3), strides=(2, 2))

    #######################################################################
    # Max Pooling 2D
    # ------------
    def test_Max_Pool2d(self, case_name):
        """Max Pooling 2D"""

        def _test_max_pool2d(input_shape,
                             pool_size,
                             strides=None,
                             padding="valid",
                             data_format="NHWC",
                             int_quant_dtype=tf.int8):
            data_format = "channels_last" if data_format == "NHWC" else "channels_first"
            keras_model, data = self.gen_keras_model(
                tf.keras.layers.MaxPool2D,
                input_shape,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )

            model_def = self._quantize_keras_model(
                keras_model,
                data,
                is_float_input=True,
                is_float_output=True,
                int_quant_dtype=int_quant_dtype,
            )
            self.convert_tflite_and_compare([data], case_name, model_def, need_transpose=True)

        _test_max_pool2d(
            (1, 28, 28, 3),
            (2, 2),
        )
        _test_max_pool2d((1, 32, 32, 3), (3, 3), strides=(2, 2))

    #######################################################################
    # transpose
    # ---------
    def test_Transpose(self, case_name):
        """Transpose"""

        def _test_forward_transpose(case_name, ishape, axes=()):
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=ishape, range=(-32, 32))
                if not axes:
                    out = array_ops.transpose(inputs[0])
                else:
                    out = array_ops.transpose(inputs[0], axes)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def, need_transpose=True)

        _test_forward_transpose(case_name, ((2, 2), ))
        _test_forward_transpose(case_name, ((2, 3, 4), ))
        _test_forward_transpose(case_name, ((7, 8, 9, 10), ))
        _test_forward_transpose(case_name, ((2, 3, 4), ), (1, 2, 0))
        _test_forward_transpose(case_name, ((2, 3, 4, 5), ), (3, 0, 1, 2))
        _test_forward_transpose(case_name, ((2, 3, 4, 5), ), ())

    #######################################################################
    # Cast
    # ----
    def test_Cast(self, case_name):
        """CAST"""

        def _test_cast(data, cast_dtype):
            with tf.Graph().as_default():
                in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
                out = math_ops.cast(in_data, cast_dtype)
                model_def = self._quantize_sess_model(
                    [in_data],
                    [out],
                )
                self.convert_tflite_and_compare([data], case_name, model_def)

        _test_cast(np.random.uniform(-32, 32, size=(1, 6)).astype(np.float32), cast_dtype=tf.int32)
        _test_cast(np.random.uniform(-32, 32, size=(5, 6)).astype(np.float32), cast_dtype=tf.uint8)
        _test_cast(np.random.uniform(-32, 32, size=(1, 6, 4, 3)).astype(np.int32),
                   cast_dtype=tf.int16)

    #######################################################################
    # Softmax
    # -------
    def test_Softmax(self, case_name):
        """Softmax"""

        def _test_softmax(shapes, range=(-32, 32), need_transpose=True):
            """One iteration of softmax"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = nn_ops.softmax(inputs[0])
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas,
                                                case_name,
                                                model_def,
                                                need_transpose=need_transpose)

        _test_softmax(((1, 6), ), (-32, 32))
        _test_softmax(((1, 3, 5, 4), ), (-32, 32))
        _test_softmax(((1, 3, 5, 4), ), (-32, 32), False)

    #######################################################################
    # Sigmoid
    # -------
    def test_Sigmoid(self, case_name):
        """Sigmoid"""

        def _test_sigmoid(shapes, range=(-32, 32)):
            """One iteration of sigmoid"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = math_ops.sigmoid(inputs[0])
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def, need_transpose=True)

        _test_sigmoid(((1, 6), ), (-32, 32))
        _test_sigmoid(((1, 3, 5, 4), ), (-32, 32))

    #######################################################################
    # Concat
    # -------
    def test_Concat(self, case_name):
        """Concat"""

        def _test_concat(shapes, axis, range=(-5, 32)):
            """One iteration of concat"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = array_ops.concat(inputs, axis)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_concat(((1, 6), (5, 6)), 0, range=(-5, 5))
        _test_concat(((1, 3, 5, 4), (1, 3, 3, 4)), 2)

    #######################################################################
    # Pack
    # -------
    def test_Pack(self, case_name):
        """Pack"""

        def _test_pack(shapes, axis, range=(-5, 32)):
            """One iteration of pack"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = array_ops.pack(inputs, axis)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_pack(((5, 6), (5, 6)), 0, range=(-5, 5))
        _test_pack(((1, 3, 5, 4), (1, 3, 5, 4)), 2)

    #######################################################################
    # Unpack
    # -------
    def test_Unpack(self, case_name):
        """Unpack"""

        def _test_unpack(shapes, axis, range=(-5, 32)):
            """One iteration of unpack"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                outs = gen_array_ops.unpack(inputs[0], shapes[0][axis], axis=axis)
                model_def = self._quantize_sess_model(inputs,
                                                      outs,
                                                      quantized=True,
                                                      input_range=in_range,
                                                      experimental_new_converter=True)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_unpack(((32, 6), ), 1, range=(-5, 5))
        _test_unpack(((32, 6, 16), ), 1, range=(-5, 5))

    #######################################################################
    # Split
    # -------
    def test_Split(self, case_name):
        """Split"""

        def _test_split(shapes, num, axis, range=(-5, 32)):
            """One iteration of split"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                outs = array_ops.split(inputs[0], num, axis=axis)
                model_def = self._quantize_sess_model(inputs,
                                                      outs,
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_split(((6, 5), ), 3, 0, range=(-5, 5))
        _test_split(((1, 3, 6, 4), ), 2, 2)

    #######################################################################
    # Gather
    # -------
    def test_Gather(self, case_name):
        """Gather"""

        def _test_gather(shapes, indices, axis=None, range=(-5, 32)):
            """One iteration of gather"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = array_ops.gather(inputs[0], indices, axis=axis)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_gather(((6, 5), ), [1, 4, 5], range=(-5, 5))
        _test_gather(((1, 3, 6, 4), ), [4, 1, 3, 2, 2, 5], 2)

    #######################################################################
    # Reshape
    # -------
    def test_Reshape(self, case_name):
        """Reshape"""

        def _test_reshape(shapes, size, range=(-5, 32)):
            """One iteration of reshape"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = tf.reshape(inputs[0], size)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_reshape(((6, 5), ), [1, 6, 5])
        _test_reshape(((1, 3, 6, 4), ), [4, 2, 9])

    #######################################################################
    # Pad
    # -------
    def test_Pad(self, case_name):
        """Pad"""

        def _test_pad(shapes, paddings, mode='CONSTANT', const_val=0, range=(-5, 32)):
            """One iteration of pad"""
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = array_ops.pad(inputs[0], paddings, mode=mode, constant_values=const_val)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range,
                                                      experimental_new_converter=True)
                self.convert_tflite_and_compare(
                    datas,
                    case_name,
                    model_def,
                )

        # pad
        _test_pad(((6, 5), ), [[1, 0], [1, 0]])
        #pad_v2
        _test_pad(((6, 5), ), [[1, 0], [1, 0]], const_val=1)
        #mirror_pad
        # _test_pad(((1,3,6,4),), [[0,0],[1,1],[0,1],[1,0]], mode="REFLECT")
        # _test_pad(((10,3),), [[0,1],[1,2]], mode="SYMMETRIC",)

    #######################################################################
    # Reduce
    # -------
    def _test_reduce(self, nn_op, case_name, shapes, axis, keep_dims=True, range=(-5, 32)):
        with tf.Graph().as_default():
            datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
            out = nn_op(inputs[0], axis=axis, keepdims=keep_dims)
            model_def = self._quantize_sess_model(inputs, [out],
                                                  quantized=True,
                                                  input_range=in_range)
            self.convert_tflite_and_compare(datas, case_name, model_def)

    def test_Mean(self, case_name):
        """Mean"""
        self._test_reduce(math_ops.reduce_mean, case_name, ((1, 3, 6, 5), ), (1, 2))

    def test_Sum(self, case_name):
        """Sum"""
        self._test_reduce(math_ops.reduce_sum, case_name, ((1, 3, 6, 5), ), (1, 2))

    def test_Redcue_Max(self, case_name):
        """Redcue_max"""
        self._test_reduce(math_ops.reduce_max, case_name, ((1, 3, 6, 5), ), (1, 2))

    def test_Redcue_Min(self, case_name):
        """Redcue_min"""
        self._test_reduce(math_ops.reduce_min, case_name, ((1, 3, 6, 5), ), (1, 2))

    #######################################################################
    # Matmul
    # -------
    def test_Matmul(self, case_name):
        """Matmul"""

        def _test_matmul(shapes, trans_a=False, trans_b=False, range=(-5, 32)):
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = math_ops.matmul(inputs[0],
                                      inputs[1],
                                      transpose_a=trans_a,
                                      transpose_b=trans_b)
                out = self.with_fused_activation_function(out, "RELU")
                out = tf.quantization.fake_quant_with_min_max_args(out,
                                                                   min=-range[1] * range[1] * 10,
                                                                   max=range[1] * range[1] * 10,
                                                                   name="out")
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_matmul(((6, 5), (4, 5)), trans_b=True)
        _test_matmul(((6, 5), (5, 10)), range=(-5, 32))

    #######################################################################
    # StridedSlice
    # -------
    def test_Strided_slice(self, case_name):
        """StridedSlice"""

        def _test_strided_slice(shapes,
                                begin,
                                end,
                                strides=None,
                                bmask=0,
                                emask=0,
                                elmask=0,
                                range=(-5, 32)):
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                out = array_ops.strided_slice(inputs[0],
                                              begin,
                                              end,
                                              strides=strides,
                                              begin_mask=bmask,
                                              end_mask=emask,
                                              ellipsis_mask=elmask)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_strided_slice(((3, 6, 5), ), [0, 1, 0], [2, 6, 4], [1, 2, 1])
        _test_strided_slice(((6, 5), ), [0, 1], [2, 0], emask=2)

    #######################################################################
    # StridedSliceMerge
    # -------
    def test_Strided_slice_merge(self, case_name):
        """StridedSliceMerge"""

        def _test_strided_slice_merge(shapes,
                                      a_begin,
                                      b_begin,
                                      a_end,
                                      b_end,
                                      a_strides=None,
                                      b_strides=None,
                                      a_bmask=0,
                                      b_bmask=0,
                                      a_emask=0,
                                      b_emask=0,
                                      range=(-5, 32)):
            with tf.Graph().as_default():
                datas, inputs, in_range = self.gen_input(shapes=shapes, range=range)
                middle = array_ops.strided_slice(inputs[0],
                                                 a_begin,
                                                 a_end,
                                                 strides=a_strides,
                                                 begin_mask=a_bmask,
                                                 end_mask=a_emask)
                out = array_ops.strided_slice(middle,
                                              b_begin,
                                              b_end,
                                              strides=b_strides,
                                              begin_mask=b_bmask,
                                              end_mask=b_emask)
                model_def = self._quantize_sess_model(inputs, [out],
                                                      quantized=True,
                                                      input_range=in_range)
                self.convert_tflite_and_compare(datas, case_name, model_def)

        _test_strided_slice_merge(((3, 6, 5), ), [0, 0, 0], [0, 0, 1], [3, 4, 5], [3, 4, 3],
                                  [1, 1, 1], [1, 1, 1])


def test_one_case_in_all(tester: TFLITE_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))


def test_all(tester: TFLITE_IR_TESTER):
    import multiprocessing
    from utils.misc import collect_process
    process_number = multiprocessing.cpu_count() // 2 + 1
    processes = []
    error_cases = multiprocessing.Manager().list()
    success_cases = multiprocessing.Manager().list()
    for case in tester.test_function:
        if tester.check_support(case):
            p = multiprocessing.Process(target=test_one_case_in_all,
                                        name=case,
                                        args=(tester, case, error_cases, success_cases))
            processes.append(p)
        if len(processes) == process_number:
            collect_process(processes, error_cases)
            processes = []
    collect_process(processes, error_cases)
    processes = []
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_tflite.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_tflite.py --chip {} TEST Success ======".format(tester.chip))
    clean_kmp_files()
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'bm1688'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'int8'],
                        help="chip platform name")
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--concise_log", action="store_true", help="use concise log")
    # yapf: enable
    args = parser.parse_args()
    tester = TFLITE_IR_TESTER(args.chip, args.mode, args.concise_log)
    dir = "tflite_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
