# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Union, Iterable, List
from .MLIRImporter import MLIRImporter, Top, State
from .BaseConverter import BaseConverter
from mlir.ir import *
import mlir.dialects.quant as quant
import numpy as np
from .tflite.TensorType import TensorType


def _indent(sOrIt_: Union[str, Iterable], numSpaces: int) -> str:
    """Indent string"""
    if sOrIt_ is None:
        return "None"
    if isinstance(sOrIt_, str):
        s = sOrIt_.split("\n")
        if len(s) == 1:
            return sOrIt_
        s = [(numSpaces * " ") + line for line in s]
        return "\n".join(s)
    s = "\n".join([line.__repr__() for line in sOrIt_])
    return _indent(s, numSpaces)


def _compute_pad(stride, dilation, input_size, filter, padding):
    # Matching tensorflow/lite/kernels/padding.h:ComputePaddingHeightWidth()
    from .tflite.Padding import Padding
    effective_filter_size = (filter - 1) * dilation + 1
    if padding == Padding.SAME:
        output_size = (input_size + stride - 1) // stride
    elif padding == Padding.VALID:
        output_size = (input_size + stride - effective_filter_size) // stride
    padding_needed = np.int64((output_size - 1) * stride + effective_filter_size - input_size)
    padding_needed = padding_needed.clip(min=0)
    # For odd values of total padding, add more padding at the 'right'
    # side of the given dimension.
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before
    pad = [i for i in padding_before] + [i for i in padding_after]
    return pad


class TFLiteReader:
    """
    Provide a TensorFlow lite model reader.
    This class can convert the TensorFlow lite model to a graph structure
    that is easy to travel and inspect. TensorFlow lite uses index mapping
    to preserve the graph information, which is compact and lightweight.
    It is hard to traverse the graph because you need to keep the context
    and do some cross-reference. With TFLiteReader, you can walk the graph
    without keeping the whole context; all the information is represented in
    a structured and self-content way.
    """

    def __init__(self, tflite_file):
        from .tflite.Model import Model
        from .tflite.BuiltinOperator import BuiltinOperator

        self.opType = {
            v: k
            for k, v in vars(BuiltinOperator).items()
            if isinstance(k, str) and isinstance(v, int)
        }

        self.tensorTypeStr = {
            v: k
            for k, v in vars(TensorType).items()
            if isinstance(k, str) and isinstance(v, int)
        }

        self.file = tflite_file
        self.TFL_fp = open(tflite_file, "rb")
        self.model = Model.GetRootAsModel(self.TFL_fp.read(), 0)
        self.operator = self.model.OperatorCodes
        self.buffer = self.model.Buffers
        self.version = self.model.Version()
        if self.model.Description():
            self.description = self.model.Description().decode()  # type: ignore
        self.description = ""

    def __del__(self):
        self.TFL_fp.close()

    def __repr__(self):
        s = "TensorFlowLite Version: {version}\nDescription: {description} \n{modstr}\n"
        modstr = "subgraph (\n{graph}\n)".format(graph=_indent(self.subgraph, 2))
        return s.format(
            version=self.version,
            description=self.description,
            modstr=modstr,
        )

    @property
    def subgraph(self):
        ctx = self

        class Tensor:

            TFLType2Np = {
                TensorType.FLOAT32: np.float32,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.UINT8: np.uint8,
                TensorType.UINT16: np.uint16,
                TensorType.UINT32: np.uint32,
                TensorType.UINT64: np.uint64,
            }

            def __init__(self, T, id):
                self.id = id
                self.T = T
                # TensorFlow using "ShapeSignature" as shape. If we can not do shape inference,
                # "shape" is a good start.
                if not self.T.ShapeIsNone():
                    self.shape = tuple(self.T.ShapeAsNumpy())
                else:
                    self.shape = None
                self.type = self.T.Type()
                self.type_str = ctx.tensorTypeStr[self.T.Type()]
                self.name = self.T.Name().decode()
                self.is_variable = self.T.IsVariable()
                self.quantization = self.T.Quantization()
                self.is_quantized = bool(
                    self.quantization and self.quantization.ZeroPointLength()
                )
                self.shape_signature = self.T.ShapeSignatureAsNumpy()
                self.buffer = self._buffer()

            def _buffer(self):
                bf = ctx.buffer(self.T.Buffer())
                if bf is None or bf.DataIsNone():
                    return None
                return (
                    bf.DataAsNumpy()
                    .view(self.TFLType2Np[self.type])  # type: ignore
                    .reshape(self.shape)
                )

            def __repr__(self):
                s = "tensor (\n{modstr}\n)"
                modstr = [self.name, self.shape, self.type_str, self.buffer]
                return s.format(modstr=_indent(modstr, 2))

        class Operator:
            def __init__(self, graph, Op):
                self.Op = Op
                self.G = graph
                opcode = ctx.operator(self.Op.OpcodeIndex())
                assert opcode is not None
                self.type = ctx.opType[
                    max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
                ]
                self.builtin_options = self.Op.BuiltinOptions()
                self.inputs = list(self._inputs())
                self.outputs = list(self._outputs())

            def __repr__(self):
                s = "{type} (\n{modstr}\n)"
                modstr = "\n".join(
                    [
                        "inputs (\n{}\n)".format(_indent(self.inputs, 2)),
                        "outputs (\n{}\n)".format(_indent(self.outputs, 2)),
                    ]
                )
                return s.format(type=self.type, modstr=_indent(modstr, 2))

            def _inputs(self):
                for i in self.Op.InputsAsNumpy():
                    if i == -1:
                        yield None
                    yield Tensor(self.G.Tensors(i), i)

            def _outputs(self):
                for i in self.Op.OutputsAsNumpy():
                    yield Tensor(self.G.Tensors(i), i)

        class Graph:
            def __init__(self, G):
                self.G = G
                self.name = "main" if G.Name() is None else self.G.Name().decode()
                self.inputs = [
                    Tensor(self.G.Tensors(i), i) for i in self.G.InputsAsNumpy()
                ]
                self.outputs = [
                    Tensor(self.G.Tensors(i), i) for i in self.G.OutputsAsNumpy()
                ]

            @property
            def operators(self):
                for i in range(self.G.OperatorsLength()):
                    yield Operator(self.G, self.G.Operators(i))

            def __repr__(self) -> str:
                s = "{name} (\n{modstr}\n)"
                modstr = "\n".join(
                    ["inputs (\n{}\n)".format(_indent(self.inputs, 2))]
                    + ["outputs (\n{}\n)".format(_indent(self.outputs, 2))]
                    + ["body (\n{}\n)".format(_indent(self.operators, 2))]
                )
                return s.format(name=self.name, modstr=_indent(modstr, 2))

        for i in range(self.model.SubgraphsLength()):
            yield Graph(self.model.Subgraphs(i))


class TFLiteConverter(BaseConverter):
    TFLType2MLIRImporterTypeStr = {
        TensorType.FLOAT32: "F32",
        TensorType.INT8: "INT8",
        TensorType.INT16: "INT16",
        TensorType.INT32: "INT32",
        TensorType.INT64: "INT64",
        TensorType.UINT8: "UINT8",
        TensorType.UINT16: None,
        TensorType.UINT32: None,
        TensorType.UINT64: None,
    }

    def __init__(self, model_name: str, tflite_file: str, input_shapes=None, preprocess_args = None):
        super().__init__()
        self.model_name = model_name
        self.tflite_file = tflite_file
        self.tflie = TFLiteReader(tflite_file)
        self.graph = next(self.tflie.subgraph)
        self.shape_infer = self.__shape_infer(input_shapes)
        self.preprocess_args = preprocess_args

        for x in self.graph.inputs:
            self.__nhwc2nchw(x)
        for x in self.graph.outputs:
            self.__nhwc2nchw(x)
        self.input_shapes = [x.shape for x in self.graph.inputs]
        self.output_shapes = [x.shape for x in self.graph.outputs]
        self.mlir = MLIRImporter(
            self.input_shapes,
            self.output_shapes,
            model_name=self.model_name,
            state=State.TOP_QUANTIZED,
            do_declare=False,
        )
        self.weight_file = self.mlir.weight_file
        self.constant = {}
        self.type_to_mlir = self.__type2mlir(self.mlir.ctx)
        self.BuiltinOptionsToAttributes = {
            "ADD": self.add_op,
            "PAD": self.pad_op,
            "SOFTMAX": self.softmax_op,
            "MEAN": self.mean_op,
            "CONV_2D": self.conv_2d_op,
            "DEPTHWISE_CONV_2D": self.depthwise_2d_op,
            "FULLY_CONNECTED": self.fully_connected_op,
            "MAX_POOL_2D": self.maxpool_op,
            "AVERAGE_POOL_2D": self.avgpool_op,
            "DEQUANTIZE": lambda _: ("top.Cast", {}),
            "QUANTIZE": lambda _: ("top.Cast", {}),
            "RESHAPE": lambda _: ("top.Reshape", {}),
        }
        input_types=[]
        for x in self.graph.inputs:
            if x.is_quantized:
                input_types.append(self.get_quantized_type(x))
            else:
                input_types.append(self.TFLType2MLIRImporterTypeStr[x.type])
        output_types=[]
        for x in self.graph.outputs:
            if x.is_quantized:
                output_types.append(self.get_quantized_type(x))
            else:
                output_types.append(self.TFLType2MLIRImporterTypeStr[x.type])
        self.mlir.declare_func(input_types, output_types)

    def __del__(self):
        if self.mlir != None:
            del self.mlir

    def __type2mlir(self, mlir_ctx):
        return {
            TensorType.FLOAT32: F32Type.get(mlir_ctx),
            TensorType.FLOAT16: F16Type.get(mlir_ctx),
            TensorType.INT32: IntegerType.get_signless(32, mlir_ctx),
            # tensorflow/tensorflow/compiler/mlir/lite/flatbuffer_import.cc::155
            TensorType.UINT8: IntegerType.get_unsigned(8, mlir_ctx),
            TensorType.INT64: IntegerType.get_signless(64, mlir_ctx),
            TensorType.STRING: None,
            TensorType.BOOL: IntegerType.get_signless(1, mlir_ctx),
            TensorType.INT16: IntegerType.get_signless(16, mlir_ctx),
            TensorType.COMPLEX64: ComplexType.get(F32Type.get(mlir_ctx)),
            TensorType.INT8: IntegerType.get_unsigned(8, mlir_ctx),
            TensorType.FLOAT64: F64Type.get(mlir_ctx),
            TensorType.COMPLEX128: ComplexType.get(F64Type.get(mlir_ctx)),
            TensorType.UINT64: IntegerType.get_unsigned(64, mlir_ctx),
            TensorType.RESOURCE: None,
            TensorType.VARIANT: None,
            TensorType.UINT32: IntegerType.get_unsigned(32, mlir_ctx),
            TensorType.UINT16: IntegerType.get_unsigned(16, mlir_ctx),
        }

    def get_quantized_type(self, tensor):
            quantParam = tensor.quantization
            if quantParam.Details() is not None:
                raise ValueError("Cannot handle experimental quantization")
            is_signed = tensor.type_str is 'INT8' or tensor.type_str is 'INT16' or tensor.type_str is 'INT32'
            storage_type = self.type_to_mlir[tensor.type]
            # TFlite uses narrow-range [u]int8 for constant buffers of quantized weights.
            # Since we don't know which ones are weights, we represent this optimization
            # as a change in the storage bounds for the type for all constants of this type.
            is_constant = tensor.buffer is not None
            is_weight_buffer = is_constant and (storage_type.width == 8) and is_signed
            storage_min = (
                quant.QuantizedType.default_minimum_for_integer(  # type: ignore
                    is_signed, storage_type.width
                )
                + is_weight_buffer
            )
            storage_max = quant.QuantizedType.default_maximum_for_integer(  # type: ignore
                is_signed, storage_type.width
            )
            flags = 1 if is_signed else 0

            scale = quantParam.ScaleAsNumpy()
            zero_point = quantParam.ZeroPointAsNumpy()
            quantized_dimension = quantParam.QuantizedDimension()
            if len(scale) > 1:
                return quant.UniformQuantizedPerAxisType.get(  # type: ignore
                    flags,
                    self.type_to_mlir[tensor.type],
                    self.type_to_mlir[TensorType.FLOAT32],
                    scale,
                    zero_point,
                    quantized_dimension,
                    storage_min,
                    storage_max,
                )
            return quant.UniformQuantizedType.get(  # type: ignore
                flags,
                self.type_to_mlir[tensor.type],
                self.type_to_mlir[TensorType.FLOAT32],
                scale[0],
                zero_point[0],
                storage_min,
                storage_max,
            )

    def __get_tensor_type(self, tensor):
        def getCalibratedQuantizedType(tensor):
            quantParam = tensor.quantization
            raw_elem_type = self.type_to_mlir[tensor.type]
            min = quantParam.MinAsNumpy()
            max = quantParam.MaxAsNumpy()
            return quant.CalibratedQuantizedType.get(raw_elem_type, min, max)  # type: ignore

        is_intermediate = False
        quantParam = tensor.quantization
        elem_type = self.type_to_mlir[tensor.type]
        if tensor.is_quantized:
            elem_type = self.get_quantized_type(tensor)
        # Intermediate tensors with calibration value (but not scale and zero points)
        # should return calibrated quantized type.
        if is_intermediate and quantParam is not None:
            elem_type = getCalibratedQuantizedType(tensor)
        if tensor.shape is not None:
            return RankedTensorType.get(tensor.shape, elem_type)
        return UnrankedTensorType.get(elem_type)

    def __nhwc2nchw(self, tensor):
        # "layout" is a marker to ensure process each tensor once.
        if hasattr(tensor, "layout"):
            return tensor
        if self.shape_infer:
            shape = self.shape_infer(tensor.id)
            if shape:  # constant tensor does not offer shape
                tensor.shape = shape

        if len(tensor.shape) != 4:
            return tensor
        n, h, w, c = tensor.shape  # type: ignore
        tensor.shape = (n, c, h, w)
        if tensor.buffer is not None:
            tensor.buffer = tensor.buffer.transpose([0, 3, 1, 2])
        tensor.layout = "NCHW"
        return tensor

    def __shape_infer(self, input_shapes):
        from .TFLiteInterpreter import TFLiteInterpreter

        if input_shapes is None:
            return None
        input_shapes_ = input_shapes.copy()
        for index, shape in enumerate(input_shapes_):
            if len(shape) == 4:
                n, c, h, w = shape
                input_shapes_[index] = (n, h, w, c)

        tfi = TFLiteInterpreter(self.tflite_file)
        inputs = {
            org_i["name"]: usr_i for org_i, usr_i in zip(tfi.inputs, input_shapes_)
        }
        tfi.reshape(**inputs)

        def get_shape(index: int):
            return tfi.tensor(index)().shape

        return get_shape

    def __create_weight_op(self, tensor):
        # constant variable/op
        tensor_type = self.__get_tensor_type(tensor)
        name_loc = Location.name(tensor.name)
        op = Operation.create(Top.WeightOp, results=[tensor_type], loc=name_loc)
        self.mlir.insert_point.insert(op)
        self.constant[tensor.name] = tensor.buffer
        return op.results[0]

    def pad_op(self, op):
        paddings = op.inputs[1].buffer
        if paddings.shape[0] == 4:
            paddings = paddings[[0, 3, 1, 2], :]
        op.inputs = [op.inputs[0]]  # remove ins[1]
        attr = {"paddings": self.mlir.ArrayAttr(paddings.flatten())}
        return "top.Pad", attr

    def add_op(self, op):
        from .tflite.AddOptions import AddOptions

        op_options = op.builtin_options
        param = AddOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception(
                "Not supported ActivationFunctionType: {}!".format(fused_active)
            )
        attr = {"do_relu": BoolAttr.get(fused_active == 1)}
        return Top.AddOp, attr

    def pool_attr(self, op):
        from .tflite.Pool2DOptions import Pool2DOptions

        op_options = op.builtin_options
        param = Pool2DOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        stride = np.array([param.StrideH(), param.StrideW()])
        kernel_size = np.array([param.FilterHeight(), param.FilterWidth()])
        input_size = np.array(op.inputs[0].shape[1:3], dtype=np.int64)  # NHWC
        padding = _compute_pad(stride, [1, 1], input_size, kernel_size, param.Padding())
        attr = {
            "kernel_shape": self.mlir.ArrayAttr([param.FilterHeight(),
                                                 param.FilterWidth()]),
            "strides": self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "pads": self.mlir.ArrayAttr(padding),
            "do_relu": BoolAttr.get(fused_active == 1),
        }
        return attr

    def maxpool_op(self, op):
        return Top.MaxPoolOp, self.pool_attr(op)

    def avgpool_op(self, op):
        return Top.AvgPoolOp, self.pool_attr(op)

    def conv_2d_op(self, op):
        from .tflite.Conv2DOptions import Conv2DOptions
        from .tflite.Padding import Padding

        op_options = op.builtin_options
        param = Conv2DOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        kernel_shape = op.inputs[1].shape
        fused_active = param.FusedActivationFunction()
        padding = [0, 0, 0, 0]  # VALID padding
        if param.Padding() == Padding.SAME:
            # high, width
            stride = np.array([param.StrideH(), param.StrideW()])
            dilation_rate = np.array(
                [param.DilationHFactor(), param.DilationWFactor()], dtype=np.int64
            )
            kernel_size = np.array(kernel_shape[1:3], dtype=np.int64)
            input_size = np.array(op.inputs[0].shape[1:3], dtype=np.int64)  # NHWC
            effective_filter_size = (kernel_size - 1) * dilation_rate + 1
            output_size = (input_size + stride - 1) // stride
            padding_needed = np.int64(
                (output_size - 1) * stride + effective_filter_size - input_size
            )
            padding_needed = padding_needed.clip(min=0)
            # For odd values of total padding, add more padding at the 'right'
            # side of the given dimension.
            padding_before = padding_needed // 2
            padding_after = padding_needed - padding_before
            padding = [
                padding_before[0],
                padding_before[1],
                padding_after[0],
                padding_after[1],
            ]

        if fused_active not in [0, 1]:
            raise Exception(
                "Not supported ActivationFunctionType: {}!".format(fused_active)
            )
        attr = {
            "kernel_shape": self.mlir.ArrayAttr(kernel_shape[1:-1]),
            "strides": self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "dilations": self.mlir.ArrayAttr(
                [param.DilationHFactor(), param.DilationWFactor()]
            ),
            "pads": self.mlir.ArrayAttr(padding),
            "do_relu": BoolAttr.get(fused_active == 1),
        }
        return Top.ConvOp, attr

    def depthwise_2d_op(self, op):
        from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
        from .tflite.Padding import Padding

        op_options = op.builtin_options
        param = DepthwiseConv2DOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        in_shape = op.inputs[0].shape
        kernel_shape = op.inputs[1].shape
        fused_active = param.FusedActivationFunction()
        padding = [0, 0, 0, 0]  # VALID padding
        # high, width
        stride = np.array([param.StrideH(), param.StrideW()])
        dilation_rate = np.array([param.DilationHFactor(), param.DilationWFactor()], dtype=np.int64)
        kernel_size = np.array(kernel_shape[1:3], dtype=np.int64)
        input_size = np.array(in_shape[1:3], dtype=np.int64)  # NHWC
        padding = _compute_pad(stride, dilation_rate, input_size, kernel_size, param.Padding())

        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {
            "kernel_shape": self.mlir.ArrayAttr(kernel_shape[1:-1]),
            "strides": self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "dilations": self.mlir.ArrayAttr(
                [param.DilationHFactor(), param.DilationWFactor()]
            ),
            "pads": self.mlir.ArrayAttr(padding),
            "do_relu": BoolAttr.get(fused_active == 1),
            "group": IntegerAttr.get(self.type_to_mlir[TensorType.INT64],
                                     in_shape[3] // kernel_shape[0]),
        }
        return Top.ConvOp, attr

    def fully_connected_op(self, op):
        from .tflite.FullyConnectedOptions import FullyConnectedOptions

        f, c = op.inputs[1].shape
        op.inputs[1].shape = (c, f)
        op.inputs[1].buffer = op.inputs[1].buffer.transpose([1, 0])
        op.inputs[1].layout = "NCHW"
        for x in op.outputs:
            self.__nhwc2nchw(x)
        op_options = op.builtin_options
        param = FullyConnectedOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception(
                "Not supported ActivationFunctionType: {}!".format(fused_active)
            )
        attr = {
            "do_relu": BoolAttr.get(fused_active == 1),
        }
        return Top.MatMulOp, attr

    def mean_op(self, op):
        args = op.inputs[1].buffer
        op.inputs = [op.inputs[0]]
        if args[0] == 1 and args[1] == 2:  # dimension reduced
            kernel_shape = [op.inputs[0].shape[1], op.inputs[0].shape[2]]
            attr = {
                "kernel_shape": self.mlir.ArrayAttr(kernel_shape),
                "strides": self.mlir.ArrayAttr([1, 1]),
                "pads": self.mlir.ArrayAttr([0, 0, 0, 0]),
            }
            return Top.AvgPoolOp, attr
        else:
            raise ValueError("Only support reduction in H and W dimensions.")

    def softmax_op(self, op):
        from .tflite.SoftmaxOptions import SoftmaxOptions
        op_options = op.builtin_options
        param = SoftmaxOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        beta = param.Beta()
        return "top.Softmax", {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], 1),
            "beta": FloatAttr.get(self.type_to_mlir[TensorType.FLOAT64], beta)
        }

    def convert_subgraph(self, subgraph):
        class symbolTable:
            symbol_table = {}

            def __init__(self, gen_value_func):
                self.gen_value_func = gen_value_func

            def __getitem__(self, tensor):
                if tensor.id not in self.symbol_table:
                    if tensor.buffer is None:
                        raise Exception(
                            "Tensor '{}' is not constant!".format(tensor.name)
                        )
                    if tensor.shape != tuple(tensor.buffer.shape):
                        raise Exception(
                            "Tensor shape is ambiguous! '{t_s}' vs '{b_s}'".format(
                                t_s=tensor.shape, b_s=tensor.buffer.shape
                            )
                        )
                    op = self.gen_value_func(tensor)
                    self.symbol_table[tensor.id] = op
                    return op
                return self.symbol_table[tensor.id]

            def update(self, other):
                self.symbol_table.update(other)

        symbol_table = symbolTable(self.__create_weight_op)
        for idx, input in enumerate(subgraph.inputs):
            input_shape = self.input_shapes[idx]
            channel_axis = -1 if self.preprocess_args['channel_format'] == 'nhwc' else 1
            image = (len(input_shape) == 4 and input_shape[channel_axis] <=4) or \
                    (len(input_shape) == 3) # gray
            if not self.preprocess_args or not image:
                input_op = self.mlir.create_input_op(input.name, idx, **{})
            else:
                preprocess_hint = {
                    'mean': self.preprocess_args['mean'],
                    'scale': self.preprocess_args['scale'],
                    'pixel_format': self.preprocess_args["pixel_format"],
                    'channel_format': self.preprocess_args["channel_format"],
                    'pad_type': self.preprocess_args["pad_type"],
                    'resize_dims': self.preprocess_args['resize_dims'],
                    'keep_aspect_ratio': self.preprocess_args['keep_aspect_ratio'],
                    'pad_value': self.preprocess_args['pad_value']
                }
                input_op = self.mlir.create_input_op(input.name, idx, **preprocess_hint)
            symbol_table.update({input.id: input_op})

        def add_operation(operation):
            op_type, attributes = self.BuiltinOptionsToAttributes[operation.type](
                operation
            )

            if operation.type is 'RESHAPE':
                operands = [symbol_table[self.__nhwc2nchw(operation.inputs[0])]]
            else:
                operands = [symbol_table[self.__nhwc2nchw(x)] for x in operation.inputs]
            rst_type = [
                self.__get_tensor_type(self.__nhwc2nchw(x)) for x in operation.outputs
            ]
            name_loc = Location.fused(
                [Location.name(x.name) for x in operation.outputs]
            )
            op = Operation.create(
                op_type,
                results=rst_type,
                operands=operands,
                attributes=attributes,
                loc=name_loc,
            )
            self.mlir.insert_point.insert(op)
            symbol_table.update(dict(zip((x.id for x in operation.outputs), op.results)))  # type: ignore

        for op in subgraph.operators:
            add_operation(op)

        return [symbol_table[x] for x in subgraph.outputs]

    def generate_mlir(self, mlir_file: str):
        return_op = self.convert_subgraph(self.graph)
        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        np.savez(self.weight_file, **self.constant)
        print("Save mlir file: {}".format(mlir_file))
