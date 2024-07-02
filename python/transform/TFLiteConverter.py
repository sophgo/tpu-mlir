# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Union, Iterable, List
from .MLIRImporter import MLIRImporter, State, Platform
from .BaseConverter import BaseConverter
from mlir.ir import *
import mlir.dialects.quant as quant
import numpy as np
from .tflite.TensorType import TensorType
import logging
import copy

logger = logging.getLogger("root")


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
            for k, v in vars(BuiltinOperator).items() if isinstance(k, str) and isinstance(v, int)
        }

        self.tensorTypeStr = {
            v: k
            for k, v in vars(TensorType).items() if isinstance(k, str) and isinstance(v, int)
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
                self.is_quantized = bool(self.quantization and self.quantization.ZeroPointLength())
                self.shape_signature = self.T.ShapeSignatureAsNumpy()
                self.buffer = self._buffer()

            def _buffer(self):
                bf = ctx.buffer(self.T.Buffer())
                if bf is None or bf.DataIsNone():
                    return None
                return (bf.DataAsNumpy().view(self.TFLType2Np[self.type])  # type: ignore
                        .reshape(self.shape))

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
                self.type = ctx.opType[max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())]
                self.builtin_options = self.Op.BuiltinOptions()
                self.inputs = list(self._inputs())
                self.outputs = list(self._outputs())

            def __repr__(self):
                s = "{type} (\n{modstr}\n)"
                modstr = "\n".join([
                    "inputs (\n{}\n)".format(_indent(self.inputs, 2)),
                    "outputs (\n{}\n)".format(_indent(self.outputs, 2)),
                ])
                return s.format(type=self.type, modstr=_indent(modstr, 2))

            def _inputs(self):
                for i in self.Op.InputsAsNumpy():
                    if i == -1:
                        yield None
                    else:
                        yield Tensor(self.G.Tensors(i), i)

            def _outputs(self):
                for i in self.Op.OutputsAsNumpy():
                    yield Tensor(self.G.Tensors(i), i)

        class Graph:

            def __init__(self, G):
                self.G = G
                self.name = "main" if G.Name() is None else self.G.Name().decode()
                self.inputs = [Tensor(self.G.Tensors(i), i) for i in self.G.InputsAsNumpy()]
                self.outputs = [Tensor(self.G.Tensors(i), i) for i in self.G.OutputsAsNumpy()]

            @property
            def operators(self):
                for i in range(self.G.OperatorsLength()):
                    yield Operator(self.G, self.G.Operators(i))

            def __repr__(self) -> str:
                s = "{name} (\n{modstr}\n)"
                modstr = "\n".join(["inputs (\n{}\n)".format(_indent(self.inputs, 2))] +
                                   ["outputs (\n{}\n)".format(_indent(self.outputs, 2))] +
                                   ["body (\n{}\n)".format(_indent(self.operators, 2))])
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
    ID = 0

    def __init__(self,
                 model_name: str,
                 tflite_file: str,
                 input_shapes=None,
                 output_names: list = [],
                 preprocess_args: dict = {},
                 shape_influencing_input_names: list = []):
        super().__init__()
        self.shape_influencing_input_names = shape_influencing_input_names
        self.model_name = model_name
        self.tflite_file = tflite_file
        self.tflie = TFLiteReader(tflite_file)
        self.graph = next(self.tflie.subgraph)
        self.preprocess_args = {}
        if 'preprocess_list' in preprocess_args:
            if preprocess_args['preprocess_list'] is not None:
                for input_index in preprocess_args['preprocess_list']:
                    assert( 0 < input_index <= len(self.graph.inputs)
                        and "Please check --preprocess_list is right input")
            else:
                preprocess_args['preprocess_list'] = [ i + 1 for i in range(len(self.graph.inputs)) ]
        self.need_transpose = False
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.need_transpose = True
                self.preprocess_args = preprocess_args
        self.shape_infer = self.__shape_infer(input_shapes)

        for x in self.graph.inputs:
            self.__nhwc2nchw(x)

        self.input_names = [x.name for x in self.graph.inputs]
        if len(output_names) == 0:
            self.output_names = [x.name for x in self.graph.outputs]
        else:
            self.output_names = output_names
        self.input_shapes = [x.shape for x in self.graph.inputs]
        for x in self.graph.inputs:
            self.addShape(x.name, x.shape)
        self.output_shapes = []
        self.outputs = []
        output_dict = dict()
        for op in self.graph.operators:
            for out in op.outputs:
                if out.name in self.output_names:
                    output_dict[out.name] = out
                    self.__nhwc2nchw(out)
                    self.addShape(out.name, out.shape)

        #for declare func user, keep order
        for out_name in self.output_names:
            if out_name in output_dict:
                self.outputs.append(output_dict[out_name])
                self.output_shapes.append(self.shapes[out_name])

        self.mlir = MLIRImporter(
            self.input_shapes,
            self.output_shapes,
            model_name=self.model_name,
            platform=Platform.TFLITE,
            state=State.TOP_QUANTIZED,
            do_declare=False,
        )
        self.weight_file = self.mlir.weight_file
        self.constant = {}
        self.type_to_mlir = self.__type2mlir(self.mlir.ctx)
        self.BuiltinOptionsToAttributes = {
            "ADD": self.add_op,
            "SUB": self.sub_op,
            "PAD": self.pad_op,
            "PADV2": self.pad_op,
            "SOFTMAX": self.softmax_op,
            "MEAN": self.mean_op,
            "SUM": self.sum_op,
            "REDUCE_MAX": self.reduce_max_op,
            "REDUCE_MIN": self.reduce_min_op,
            "CONV_2D": self.conv_2d_op,
            "DECONV_2D": self.deconv_2d_op,
            "DEPTHWISE_CONV_2D": self.depthwise_2d_op,
            "FULLY_CONNECTED": self.fully_connected_op,
            "MAX_POOL_2D": self.maxpool_op,
            "AVERAGE_POOL_2D": self.avgpool_op,
            "DEQUANTIZE": lambda _: ("top.Cast", {}, False),
            "QUANTIZE": lambda _: ("top.Cast", {}, False),
            "CAST": lambda _: ("top.Cast", {}, False),
            "RESHAPE": lambda _: ("top.Reshape", {}, False),
            "CONCATENATION": self.concat_op,
            "LOGISTIC": lambda _: ("top.Sigmoid", {}, False),
            "MUL": self.mul_op,
            "RESIZE_NEAREST_NEIGHBOR": self.resize_nearest_op,
            "STRIDED_SLICE": self.stride_slice_op,
            "SPLIT": self.split_op,
            "PACK": self.pack_op,
            "UNPACK": self.unpack_op,
            "GATHER": self.gather_op,
            "TRANSPOSE": self.transpose_op,
        }
        input_types = []
        for x in self.graph.inputs:
            if x.is_quantized:
                input_types.append(self.get_quantized_type(x))
            else:
                input_types.append(self.TFLType2MLIRImporterTypeStr[x.type])
        output_types = []
        for x in self.outputs:
            if x.is_quantized:
                output_types.append(self.get_quantized_type(x))
            else:
                output_types.append(self.TFLType2MLIRImporterTypeStr[x.type])
        self.mlir.declare_func(input_types, output_types)

    def __del__(self):
        if self.mlir != None:
            del self.mlir

    def __type2mlir(self, mlir_ctx):
        # yapf: disable
        return {
            TensorType.FLOAT32: F32Type.get(mlir_ctx),
            TensorType.FLOAT16: F16Type.get(mlir_ctx),
            TensorType.INT32: IntegerType.get_signed(32, mlir_ctx),
            # tensorflow/tensorflow/compiler/mlir/lite/flatbuffer_import.cc::155
            TensorType.UINT8: IntegerType.get_unsigned(8, mlir_ctx),
            TensorType.INT64: IntegerType.get_signless(64, mlir_ctx),
            TensorType.STRING: None,
            TensorType.BOOL: IntegerType.get_signless(1, mlir_ctx),
            TensorType.INT16: IntegerType.get_signed(16, mlir_ctx),
            TensorType.COMPLEX64: ComplexType.get(F32Type.get(mlir_ctx)),
            TensorType.INT8: IntegerType.get_signed(8, mlir_ctx),
            TensorType.FLOAT64: F64Type.get(mlir_ctx),
            TensorType.COMPLEX128: ComplexType.get(F64Type.get(mlir_ctx)),
            TensorType.UINT64: IntegerType.get_unsigned(64, mlir_ctx),
            TensorType.RESOURCE: None,
            TensorType.VARIANT: None,
            TensorType.UINT32: IntegerType.get_unsigned(32, mlir_ctx),
            TensorType.UINT16: IntegerType.get_unsigned(16, mlir_ctx),
        }
        # yapf: enable

    def get_quantized_type(self, tensor):
        quantParam = tensor.quantization
        if quantParam.Details() is not None:
            raise ValueError("Cannot handle experimental quantization")
        is_signed = tensor.type_str in ['INT8', 'INT16', 'INT32']
        storage_type = self.type_to_mlir[tensor.type]
        # TFLite uses narrow-range [u]int8 for constant buffers of quantized weights.
        # Since we don't know which ones are weights, we represent this optimization
        # as a change in the storage bounds for the type for all constants of this type.
        is_constant = tensor.buffer is not None
        is_weight_buffer = is_constant and (storage_type.width == 8) and is_signed
        storage_min = (
            quant.QuantizedType.default_minimum_for_integer(  # type: ignore
                is_signed, storage_type.width) + is_weight_buffer)
        storage_max = quant.QuantizedType.default_maximum_for_integer(  # type: ignore
            is_signed, storage_type.width)
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

    def __get_tensor_type(self, tensor, shape=None):

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
        if shape is not None:
            return RankedTensorType.get(shape, elem_type)
        if tensor.shape is not None:
            return RankedTensorType.get(tensor.shape, elem_type)
        return UnrankedTensorType.get(elem_type)

    def __nhwc2nchw(self, tensor, need=False):
        if not self.need_transpose and need is False:
            return tensor
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
            if len(shape) == 4 and self.need_transpose is True:
                n, c, h, w = shape
                input_shapes_[index] = (n, h, w, c)

        tfi = TFLiteInterpreter(self.tflite_file)
        inputs = {org_i["name"]: usr_i for org_i, usr_i in zip(tfi.inputs, input_shapes_)}
        tfi.reshape(**inputs)

        def get_shape(index: int):
            return tfi.tensor(index)().shape

        return get_shape

    def __create_weight_op(self, tensor):
        # constant variable/op
        if len(tensor.shape) == 0:
            self.constant[tensor.name] = tensor.buffer.reshape(1)
        else:
            self.constant[tensor.name] = tensor.buffer
        tensor_type = self.__get_tensor_type(tensor, self.constant[tensor.name].shape)
        name_loc = Location.name(tensor.name)
        op = Operation.create("top.Weight", results=[tensor_type], loc=name_loc)
        self.mlir.insert_point.insert(op)
        return op.results[0]

    def __shape_transpose(self, shape: List[int], order: List[int]):
        if len(shape) != 4 or len(order) != 4:
            return shape
        shape_ = []
        for i in range(len(order)):
            shape_.append(shape[order[i]])
        return shape_

    def __do_transpose(self, tensor, operand, order, name, same=True):
        shape = self.__shape_transpose(tensor.shape, order)
        if shape == tensor.shape:
            return operand
        tensor_type = self.__get_tensor_type(tensor, shape if same else None)
        name_loc = Location.name(name)
        attr = {
            "order": self.mlir.ArrayAttr(order),
        }
        op = Operation.create(
            "top.Permute",
            results=[tensor_type],
            operands=[operand],
            attributes=attr,
            loc=name_loc,
        )
        self.mlir.insert_point.insert(op)
        return op.results[0]

    def __axis_transpose(self, op, axis):
        if self.need_transpose is False:
            return axis
        in_dim = len(op.inputs[0].shape)
        if in_dim == 4:
            if axis == 0:
                return 0
            elif axis == 3:
                return 1
            else:
                return axis + 1
        else:
            return axis

    def __get_new_name(self):
        id = int(TFLiteConverter.ID)
        TFLiteConverter.ID += 1
        name = "tpumlir_tensor_" + str(id)
        return name

    def pad_op(self, op):
        paddings = op.inputs[1].buffer
        if paddings.shape[0] == 4 and self.need_transpose:
            paddings = paddings[[0, 3, 1, 2], :]
        paddings = paddings.transpose([1, 0])
        pad_val = op.outputs[0].quantization.ZeroPoint(0)
        if len(op.inputs) == 3:
            pad_val = op.inputs[2].buffer
        op.inputs = [op.inputs[0]]  # remove ins[1]
        attr = {
            "paddings": self.mlir.ArrayAttr(paddings.flatten()),
            "val": FloatAttr.get(self.type_to_mlir[TensorType.FLOAT64], pad_val),
            "mode":StringAttr.get("constant")
        }
        return "top.Pad", attr, self.need_transpose

    def add_op(self, op):
        from .tflite.AddOptions import AddOptions

        op_options = op.builtin_options
        param = AddOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {"do_relu": BoolAttr.get(fused_active == 1)}
        return "top.Add", attr, False

    def sub_op(self, op):
        from .tflite.SubOptions import SubOptions

        op_options = op.builtin_options
        param = SubOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {"do_relu": BoolAttr.get(fused_active == 1)}
        return "top.Sub", attr, False

    def mul_op(self, op):
        from .tflite.AddOptions import AddOptions

        op_options = op.builtin_options
        param = AddOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {"do_relu": BoolAttr.get(fused_active == 1)}
        return "top.Mul", attr, False

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
        return "top.MaxPool", self.pool_attr(op), True

    def avgpool_op(self, op):
        return "top.AvgPool", self.pool_attr(op), True

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
                [param.DilationHFactor(), param.DilationWFactor()], dtype=np.int64)
            kernel_size = np.array(kernel_shape[1:3], dtype=np.int64)
            input_size = np.array(op.inputs[0].shape[1:3], dtype=np.int64)  # NHWC
            effective_filter_size = (kernel_size - 1) * dilation_rate + 1
            output_size = (input_size + stride - 1) // stride
            padding_needed = np.int64((output_size - 1) * stride + effective_filter_size -
                                      input_size)
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

        if fused_active not in [0, 1, 3]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {
            "kernel_shape": self.mlir.ArrayAttr(kernel_shape[1:-1]),
            "strides": self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "dilations": self.mlir.ArrayAttr([param.DilationHFactor(),
                                              param.DilationWFactor()]),
            "pads": self.mlir.ArrayAttr(padding),
            "do_relu": BoolAttr.get(fused_active == 1 or fused_active == 3),  # relu6 to relu
        }
        return "top.Conv", attr, True

    def depthwise_2d_op(self, op):
        from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
        from .tflite.Padding import Padding

        op_options = op.builtin_options
        param = DepthwiseConv2DOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        in_shape = op.inputs[0].shape
        kernel_shape = op.inputs[1].shape
        op.inputs[1].shape = (kernel_shape[3], kernel_shape[0], kernel_shape[1], kernel_shape[2])
        op.inputs[1].buffer = op.inputs[1].buffer.transpose([3, 0, 1, 2])
        op.inputs[1].layout = "NCHW"
        fused_active = param.FusedActivationFunction()
        padding = [0, 0, 0, 0]  # VALID padding
        # high, width
        stride = np.array([param.StrideH(), param.StrideW()])
        dilation_rate = np.array([param.DilationHFactor(), param.DilationWFactor()], dtype=np.int64)
        kernel_size = np.array(kernel_shape[1:3], dtype=np.int64)
        input_size = np.array(in_shape[1:3], dtype=np.int64)  # NHWC
        padding = _compute_pad(stride, dilation_rate, input_size, kernel_size, param.Padding())

        if fused_active not in [0, 1, 3]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {
            "kernel_shape":
            self.mlir.ArrayAttr(kernel_shape[1:-1]),
            "strides":
            self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "dilations":
            self.mlir.ArrayAttr([param.DilationHFactor(),
                                 param.DilationWFactor()]),
            "pads":
            self.mlir.ArrayAttr(padding),
            "do_relu":
            BoolAttr.get(fused_active == 1 or fused_active == 3),  # relu6 to relu
            "group":
            IntegerAttr.get(self.type_to_mlir[TensorType.INT64], in_shape[3] // kernel_shape[0]),
        }
        return "top.Conv", attr, True

    def deconv_2d_op(self, op):
        from .tflite.TransposeConvOptions import TransposeConvOptions
        from .tflite.Padding import Padding

        op_options = op.builtin_options
        param = TransposeConvOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        kernel_shape = op.inputs[1].shape
        fused_active = param.FusedActivationFunction()
        padding = [0, 0, 0, 0]  # VALID padding
        if param.Padding() == Padding.SAME:
            # high, width
            stride = np.array([param.StrideH(), param.StrideW()])
            dilation_rate = np.array(
                [param.DilationHFactor(), param.DilationWFactor()], dtype=np.int64)
            kernel_size = np.array(kernel_shape[1:3], dtype=np.int64)
            input_size = np.array(op.inputs[0].shape[1:3], dtype=np.int64)  # NHWC
            effective_filter_size = (kernel_size - 1) * dilation_rate + 1
            output_size = (input_size + stride - 1) // stride
            padding_needed = np.int64((output_size - 1) * stride + effective_filter_size -
                                      input_size)
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

        if fused_active not in [0, 1, 3]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {
            "kernel_shape": self.mlir.ArrayAttr(kernel_shape[1:-1]),
            "strides": self.mlir.ArrayAttr([param.StrideH(), param.StrideW()]),
            "dilations": self.mlir.ArrayAttr([param.DilationHFactor(),
                                              param.DilationWFactor()]),
            "pads": self.mlir.ArrayAttr(padding),
            "do_relu": BoolAttr.get(fused_active == 1 or fused_active == 3),  # relu6 to relu
        }
        return "top.Deconv", attr, True

    def fully_connected_op(self, op):
        from .tflite.FullyConnectedOptions import FullyConnectedOptions

        op_options = op.builtin_options
        param = FullyConnectedOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        fused_active = param.FusedActivationFunction()
        keep_dims = param.KeepNumDims()
        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        attr = {
            "do_relu": BoolAttr.get(fused_active == 1),
            "keep_dims": BoolAttr.get(keep_dims),
        }
        if op.inputs[2] is not None:
            bias_shape = [1] * len(op.inputs[0].shape)
            bias_shape[-1] = op.inputs[2].shape[0]
            op.inputs[2].shape = tuple(bias_shape)
            op.inputs[2].buffer.shape = tuple(bias_shape)

        if op.inputs[1].buffer is not None:
            f, c = op.inputs[1].shape
            op.inputs[1].shape = (c, f)
            op.inputs[1].buffer = op.inputs[1].buffer.transpose([1, 0])
            op.inputs[1].layout = "NCHW"
        else:
            attr["right_transpose"] = BoolAttr.get(True)
        return "top.MatMul", attr, False

    def mean_op(self, op):
        args = op.inputs[1].buffer
        op.inputs = [op.inputs[0]]
        if args[0] == 1 and args[1] == 2:  # dimension reduced
            kernel_shape = [op.inputs[0].shape[1], op.inputs[0].shape[2]]
            attr = {
                "kernel_shape": self.mlir.ArrayAttr(kernel_shape),
                "strides": self.mlir.ArrayAttr([1, 1]),
                "pads": self.mlir.ArrayAttr([0, 0, 0, 0]),
                "keepdims": BoolAttr.get(len(op.inputs[0].shape) == len(op.outputs[0].shape)),
            }
            return "top.AvgPool", attr, True
        else:
            raise ValueError("Only support reduction in H and W dimensions.")

    def reduce_op(self, op, mode):
        from .tflite.ReducerOptions import ReducerOptions
        op_options = op.builtin_options
        param = ReducerOptions()
        param.Init(op_options.Bytes, op_options.Pos)

        args = op.inputs[1].buffer
        op.inputs = [op.inputs[0]]
        axes = [self.__axis_transpose(op, i) for i in args]
        attr = {
            "axes": self.mlir.ArrayAttr(axes),
            "keepdims": BoolAttr.get(param.KeepDims()),
            "mode": StringAttr.get(mode),
        }
        return "top.Reduce", attr, True

    # def mean_op(self, op):
    #     return self.reduce_op(op, "ReduceMean")

    def sum_op(self, op):
        return self.reduce_op(op, "ReduceSum")

    def reduce_min_op(self, op):
        return self.reduce_op(op, "ReduceMin")

    def reduce_max_op(self, op):
        return self.reduce_op(op, "ReduceMax")

    def softmax_op(self, op):
        from .tflite.SoftmaxOptions import SoftmaxOptions
        op_options = op.builtin_options
        param = SoftmaxOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        beta = param.Beta()
        axis = 1 if self.need_transpose else len(op.inputs[0].shape) - 1
        # axis = self.__axis_transpose(op, len(op.inputs[0].shape) - 1)
        return "top.Softmax", {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], axis),
            "beta": FloatAttr.get(self.type_to_mlir[TensorType.FLOAT64], beta)
        }, self.need_transpose

    def concat_op(self, op):
        from .tflite.ConcatenationOptions import ConcatenationOptions
        op_options = op.builtin_options
        param = ConcatenationOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        axis = param.Axis() if param.Axis() >= 0 else param.Axis() + len(op.inputs[0].shape)
        axis = self.__axis_transpose(op, axis) if self.need_transpose else axis
        fused_active = param.FusedActivationFunction()
        if fused_active not in [0, 1]:
            raise Exception("Not supported ActivationFunctionType: {}!".format(fused_active))
        return "top.Concat", {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], axis),
            "do_relu": BoolAttr.get(fused_active == 1),
        }, False

    def resize_nearest_op(self, op):
        from .tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
        op_options = op.builtin_options
        param = ResizeNearestNeighborOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        align_corner = param.AlignCorners()
        half_piexl = param.HalfPixelCenters()
        assert op.inputs[1].buffer is not None
        scale = op.inputs[1].buffer
        op.inputs = [op.inputs[0], None]
        coord_mode = "asymmetric"
        # round_mode = "down"
        if align_corner is True:
            coord_mode = "align_corners"
        if half_piexl is True:
            coord_mode = "half_pixel"
            # round_mode = "half_up"
        # elif align_corner is False and half_piexl is False:
        #     coord_mode = 2
        else:
            raise Exception("Not supported : Param {} {}!".format(align_corner, half_piexl))
        return "top.Interp", {
            "coord_mode":
            StringAttr.get(coord_mode),
            "mode":
            StringAttr.get("nearest"),
            "scale_h":
            FloatAttr.get(self.type_to_mlir[TensorType.FLOAT64], scale[0] / op.inputs[0].shape[1]),
            "scale_w":
            FloatAttr.get(self.type_to_mlir[TensorType.FLOAT64], scale[1] / op.inputs[0].shape[2]),
        }, False

    def stride_slice_op(self, op):
        from .tflite.StridedSliceOptions import StridedSliceOptions
        op_options = op.builtin_options
        param = StridedSliceOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        begin_mask = param.BeginMask()
        end_mask = param.EndMask()
        ellipsis_mask = param.EllipsisMask()
        new_axis_mask = param.NewAxisMask()
        shrink_axis_mask = param.ShrinkAxisMask()
        return "top.StridedSlice", {
            "begin_mask": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], begin_mask),
            "end_mask": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], end_mask),
            "ellipsis_mask": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], ellipsis_mask),
            "new_axis_mask": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], new_axis_mask),
            "shrink_axis_mask": IntegerAttr.get(self.type_to_mlir[TensorType.INT64],
                                                shrink_axis_mask),
        }, False

    def split_op(self, op):
        from .tflite.SplitOptions import SplitOptions
        op_options = op.builtin_options
        param = SplitOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        assert op.inputs[0].buffer is not None
        axis = op.inputs[0].buffer.tolist()
        op.inputs.pop(0)
        attr = {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], axis),
            "num": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], param.NumSplits()),
        }
        return "top.Split", attr, False

    def pack_op(self, op):
        from .tflite.PackOptions import PackOptions
        op_options = op.builtin_options
        param = PackOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        attr = {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], param.Axis()),
            "values_count": IntegerAttr.get(self.type_to_mlir[TensorType.INT64],
                                            param.ValuesCount()),
        }
        return "top.Pack", attr, False

    def unpack_op(self, op):
        from .tflite.UnpackOptions import UnpackOptions
        op_options = op.builtin_options
        param = UnpackOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        attr = {"axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], param.Axis())}
        return "top.Unpack", attr, False

    def gather_op(self, op):
        from .tflite.GatherOptions import GatherOptions
        op_options = op.builtin_options
        param = GatherOptions()
        param.Init(op_options.Bytes, op_options.Pos)
        attr = {
            "axis": IntegerAttr.get(self.type_to_mlir[TensorType.INT32], param.Axis()),
            # "num": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], param.BatchDims()),
        }
        return "top.Gather", attr, False

    def transpose_op(self, op):
        assert op.inputs[1].buffer is not None
        perm = op.inputs[1].buffer
        if len(perm) == 4 and self.need_transpose:
            n, h, w, c = perm
            perm = (n, c, h, w)
        axes = [self.__axis_transpose(op, i) for i in perm]
        op.inputs.pop(1)
        attr = {
            "order": self.mlir.ArrayAttr(axes),
            # "num": IntegerAttr.get(self.type_to_mlir[TensorType.INT64], param.BatchDims()),
        }
        return "top.Permute", attr, False

    def convert_subgraph(self, subgraph):

        class symbolTable:
            symbol_table = {}

            def __init__(self, gen_value_func):
                self.gen_value_func = gen_value_func

            def __getitem__(self, tensor):
                if tensor.id not in self.symbol_table:
                    if tensor.buffer is None:
                        raise Exception("Tensor '{}' is not constant!".format(tensor.name))
                    if tensor.shape != tuple(tensor.buffer.shape) and np.prod(
                            tensor.shape) != np.prod(tensor.buffer.shape):
                        raise Exception("Tensor shape is ambiguous! '{t_s}' vs '{b_s}'".format(
                            t_s=tensor.shape, b_s=tensor.buffer.shape))
                    op = self.gen_value_func(tensor)
                    self.symbol_table[tensor.id] = op
                    return op
                return self.symbol_table[tensor.id]

            def update(self, other):
                self.symbol_table.update(other)

        symbol_table = symbolTable(self.__create_weight_op)
        for idx, input in enumerate(subgraph.inputs):
            loc = Location.fused([Location.name(input.name)])
            input_data = None
            if self.shape_influencing_input_names:
                test_file = ''
                if isinstance(self.test_input, list):
                    assert self.test_input[0].endswith('.npz')
                    test_file = self.test_input[0]
                elif isinstance(self.test_input, str):
                    assert self.test_input.endswith('.npz')
                    test_file = self.test_input
                else:
                    raise ValueError("test_input npz file is necessary when shape_influencing_input_names is set")
                input_data = np.load(test_file)
            kwargs = copy.deepcopy(self.preprocess_args)
            if input.name in self.shape_influencing_input_names:
                assert input_data[input.name].ndim == 1, "input shape tensor should be 1D tensor"
                kwargs['shape_tensor'] = input_data[input.name]
            input_op = self.mlir.create_input_op(loc, idx, kwargs)
            symbol_table.update({input.id: input_op})

        def add_operation(operation):
            op_type, attributes, channel_last = self.BuiltinOptionsToAttributes[operation.type](
                operation)

            trans_suffix = ""
            if channel_last and self.need_transpose is False:
                # do transpose (NCHW to NCHW)
                operands = []
                for idx, x in enumerate(operation.inputs):
                    # name = operation.outputs[0].name + "_bm_trans" + str(idx)
                    if x is None:
                        operands.append(self.mlir.none_op)
                    elif x.id not in symbol_table.symbol_table.keys():
                        operands.append(symbol_table[self.__nhwc2nchw(x, True)])
                    else:
                        operands.append(
                            self.__do_transpose(x, symbol_table[x], [0, 3, 1, 2],
                                                self.__get_new_name()))
                        if len(x.shape) == 4:
                            trans_suffix = "_bm_trans"
                rst_type = [
                    self.__get_tensor_type(x, self.__shape_transpose(x.shape, [0, 3, 1, 2]))
                    for x in operation.outputs
                ]
            else:
                if operation.type == 'RESHAPE':
                    operands = [symbol_table[self.__nhwc2nchw(operation.inputs[0])]]
                else:
                    operands = []
                    for x in operation.inputs:
                        operands.append(symbol_table[self.__nhwc2nchw(x)]
                                        if x is not None else self.mlir.none_op)
                rst_type = [self.__get_tensor_type(self.__nhwc2nchw(x)) for x in operation.outputs]

            name_loc = Location.fused(
                [Location.name(x.name + trans_suffix) for x in operation.outputs])
            op = Operation.create(
                op_type,
                results=rst_type,
                operands=operands,
                attributes=attributes,
                loc=name_loc,
            )
            self.mlir.insert_point.insert(op)
            if channel_last and self.need_transpose is False:
                # do transpose (NCHW to NHWC)
                res = []
                for idx, x in enumerate(operation.outputs):
                    res.append(self.__do_transpose(x, op.results[idx], [0, 2, 3, 1], x.name, False))
            else:
                res = op.results
            symbol_table.update(dict(zip((x.id for x in operation.outputs), res)))  # type: ignore

        return_op = []
        graphs_out = dict()
        for op in subgraph.operators:
            add_operation(op)
            for out in op.outputs:
                if out is None: continue
                if out.name in self.output_names:
                    graphs_out[out.name] = out

        for out_name in self.output_names:
            if out_name in graphs_out:
                return_op.append(symbol_table[graphs_out[out_name]])

        return return_op

    def generate_mlir(self, mlir_file: str):
        return_op = self.convert_subgraph(self.graph)
        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        np.savez(self.weight_file, **self.constant)
        logger.info("Save mlir file: {}".format(mlir_file))
