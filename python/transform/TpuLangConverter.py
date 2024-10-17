# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import Union, Iterable, List, Tuple, get_origin, get_args
from .MLIRImporter import MLIRImporter, State, Platform
from .BaseConverter import BaseConverter
from mlir.ir import *
import mlir.dialects.quant as quant
from utils.mlir_shell import *
import numpy as np
import logging

logger = logging.getLogger("root")


supported_dtypes = ["float32", "float16", "int32", "uint32", "int16", "uint16", "int8", "uint8"]

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

def generate_name(op_name):
    import uuid
    unique_name = str(uuid.uuid4())
    return f"{op_name}_{unique_name}"

def auto_name(kwarg_name='out_name'):
    """
        auto generate name
    """

    def wrapper(func):
        def decorate(*args, **kwargs):
            if kwarg_name in kwargs:
                need_gen_name = kwargs[kwarg_name] is None
            else:
                need_gen_name = True
            if need_gen_name:
                kwargs[kwarg_name] = generate_name(func.__name__)
            return func(*args, **kwargs)

        return decorate
    return wrapper

def to_scalar(num):
    """
        convert `int` or `float` to `Scalar` type
    """
    def wrapper(func):

        def __to_scalar(data):
            if isinstance(data, float):
                return Scalar(data, "float32")
            elif isinstance(data, int):
                return Scalar(data, "int32")
            return data

        def decorate(*args, **kwargs):
            need_to_scalar = False
            for idx, v in enumerate(args):
                if isinstance(v, (int, float)) and idx < num:
                    need_to_scalar = True
            if need_to_scalar:
                new_args = []
                for idx, v in enumerate(args):
                    if isinstance(v, (int, float)) and idx < num:
                        new_args.append(__to_scalar(v))
                    else:
                        new_args.append(v)
                return func(*new_args, **kwargs)
            else:
                return func(*args, **kwargs)

        decorate.__name__ = func.__name__
        return decorate
    return wrapper

def annotation_check(func):
    """
        check python types of function arguments as annotations
    """
    def __type_instance(type0, type1):
        if get_origin(type1) is Union:
            ret = False
            for t in get_args(type1):
                ret = ret or __type_instance(type0, t)
            return ret
        if get_origin(type1) is list or get_origin(type1) is tuple:
            if not isinstance(type0, list) and not isinstance(type0, tuple):
                return False
            return all(isinstance(type, get_args(type1)) for type in type0)
        else:
            return isinstance(type0, type1)

    def decorate(*args, **kwargs):
        idx = 0
        for k, v in func.__annotations__.items():
            if idx >= len(args):
                if k in kwargs:
                    if kwargs[k] is None:
                        continue
                    assert __type_instance(kwargs[k], v), "code {} type error, {} expected type: {}".format(func.__code__, k, v)
            else:
                if args[idx] is None:
                    continue
                assert __type_instance(args[idx], v), "code {} type error, input #{} expected type: {}".format(func.__code__, idx, v)
            idx += 1

        return func(*args, **kwargs)

    decorate.__name__ = func.__name__
    return decorate

def check_dtype(dtype: str):
    assert dtype.lower() in supported_dtypes

# the decorator should stay behind auto_name()
def assert_with_out_name(func):
    def wrapper(*args, **kwargs):
        out_name = kwargs.get('out_name', '')
        out_names = kwargs.get('out_names', [''])
        try:
            return func(*args, **kwargs)
        except AssertionError as e:
            raise AssertionError(f"{e}. out_name: {out_name if out_name else ', '.join(out_names)}")
    wrapper.__name__ = func.__name__
    return wrapper

class Scalar:

    def __init__(self, value, dtype: str ='float32'):
        check_dtype(dtype)
        self.value = value
        self.dtype = dtype

    def __repr__(self):
        s = "scalar (\n{modstr}\n)"
        modstr = [self.value, self.dtype]
        return s.format(modstr=_indent(modstr, 2))

class Tensor:
    ID = 0

    def __init__(self,
                 shape: list = [],
                 name: str = None,
                 ttype: str ="neuron",
                 data: np.ndarray = None,
                 dtype: str = "float32",
                 scale: Union[float, List[float]] = None,
                 zero_point: Union[int, List[int]] = None):
        self.id = int(Tensor.ID)
        shape = shape if isinstance(shape, list) else [shape]
        self.shape = shape
        self.name = "BMTensor" + str(self.id) if name is None else name
        assert ttype.lower() in ["neuron", "coeff"]
        self.ttype = ttype.lower()
        check_dtype(dtype)
        self.dtype = dtype.lower()
        if data is not None:
            if self.dtype == "float16" and data.dtype == "float32":
                data = data.astype("float16")
            assert data.dtype == self.dtype
        if data is not None and tuple(shape) != tuple(data.shape):
            num = 1
            for s in shape:
                num *= s
            if num == data.size:
                data = data.reshape(shape)
            else:
                raise Exception("Tensor shape is ambiguous! '{t_s}' vs '{b_s}'".format(
                                t_s=shape, b_s=data.shape))
        self.buffer = data
        self.is_quantized: bool = False
        self.quantization(scale=scale, zero_point=zero_point)
        self.is_preprocess = False
        Tensor.ID += 1

    def reset(self):
        self.buffer = None

    def quantization(self,
                     scale: Union[float, List[float]] = None,
                     zero_point: Union[int, List[int]] = None):
        if self.dtype != "int8" and self.dtype != "uint8" and self.dtype != "int16" and self.dtype != "uint16":
            assert scale is None and zero_point is None, \
            "When dtype is not int8 or uint8, scale and zero_point must be None."
        if scale is not None:
            scale = [abs(s) for s in scale] if isinstance(scale, List) else abs(scale)
        if self.is_quantized is False:
            self.is_quantized = scale is not None or zero_point is not None
            self.scale = scale
            self.zero_point = zero_point
        else:
            if self.scale is None:
                self.scale = scale
            elif scale is not None:
                assert self.scale == scale
            if self.zero_point is None:
                self.zero_point = zero_point
            elif zero_point is not None:
                assert self.zero_point == zero_point

    def preprocess(self,
                   mean : List[float] = [0, 0, 0],
                   scale : List[float] = [1.0, 1.0, 1.0],
                   pixel_format : str = 'bgr',
                   channel_format : str = 'nchw',
                   resize_dims : List[int] = None,
                   keep_aspect_ratio : bool = False,
                   keep_ratio_mode : str = 'letterbox',
                   pad_value : int = 0,
                   pad_type : str = 'center'):
        self.mean = mean
        self.scale = scale
        assert pixel_format in ['rgb', 'bgr', 'gray', 'rgba', 'gbrg', 'grbg', 'bggr', 'rggb' ]
        assert channel_format in ['nhwc', 'nchw']
        self.pixel_format = pixel_format
        self.channel_format = channel_format
        if resize_dims == None:
            self.resize_dims = self.shape[-2:] if channel_format == 'nchw' else self.shape[-3 : -1]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.keep_ratio_mode = keep_ratio_mode
        self.pad_value = pad_value
        self.pad_type = pad_type
        self.is_preprocess = True

    def preprocess_to_dict(self):
        if self.is_preprocess:
            return {
                'resize_dims': self.resize_dims,
                'keep_aspect_ratio': self.keep_aspect_ratio,
                'keep_ratio_mode': self.keep_ratio_mode,
                'pad_value': self.pad_value,
                'pad_type': self.pad_type,
                'mean': self.mean,
                'scale': self.scale,
                'pixel_format': self.pixel_format,
                'channel_format': self.channel_format
            }
        return {}

    def __repr__(self):
        s = "tensor (\n{modstr}\n)"
        modstr = [self.id, self.name, self.shape, self.ttype, self.dtype, self.buffer]
        if self.is_quantized:
            modstr += [self.scale, self.zero_point]
        return s.format(modstr=_indent(modstr, 2))

class Operator:

    def __init__(self,
                 op_name: str,
                 inputs: List[Tensor],
                 outputs: List[Tensor],
                 params: dict = {}):
        self.op_name = op_name
        self.params = params
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        s = "{type} (\n{modstr}\n)"
        modstr = "\n".join([
            "inputs (\n{}\n)".format(_indent(self.inputs, 2)),
            "outputs (\n{}\n)".format(_indent(self.outputs, 2)),
        ])
        return s.format(type=self.op_name, modstr=_indent(modstr, 2))

    def __del__(self):
        del self.inputs
        del self.outputs

class Graph:

    def __init__(self, name="main"):
        self.name = name
        self.operators: List[Operator] = []
        self._inputs = None
        self._outputs = None
        self.tensors_dict = dict()

    def __del__(self):
        del self.operators

    def check_tensors(self, op: Operator):
        for tensor in op.inputs + op.outputs:
            if tensor is None:
                continue
            if tensor.name in self.tensors_dict.keys():
                assert tensor.id == self.tensors_dict[tensor.name], "Tensor name must be uinque. {} has been used".format(tensor.name)
            else:
                self.tensors_dict[tensor.name] = tensor.id

    def insert_op(self, op):
        # op = Operator(op_name, params=params, inputs=inputs, outputs=outputs)
        self.check_tensors(op)
        self.operators.append(op)

    def reset(self):
        for op in self.operators:
            for tensor in op.inputs + op.outputs:
                if isinstance(tensor, Tensor):
                    tensor.reset()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[Tensor]):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[Tensor]):
        self._outputs = outputs

    def quantized_type_inference(self):
        # operators that needs no quantization.
        assign_ops = ["Permute", "Reshape", "Tile", "Split", "Repeat", "Relu", \
                      "Unsqueeze", "Unsqueeze", "Expand", "Slice", "Copy", \
                      "LeakyRelu", "Abs", "Pad"]
        for op in self.operators:
            if op.op_name.split('.')[1] in assign_ops and len(op.inputs) == 1 and len(op.outputs) == 1 and op.inputs[0].is_quantized:
                input = op.inputs[0]
                output = op.outputs[0]
                output.quantization(scale=input.scale, zero_point=input.zero_point)
        for op in list(reversed(self.operators)):
            if op.op_name.split('.')[1] in assign_ops and len(op.inputs) == 1 and len(op.outputs) == 1 and op.outputs[0].is_quantized:
                input = op.inputs[0]
                output = op.outputs[0]
                input.quantization(scale=output.scale, zero_point=output.zero_point)

    def __repr__(self) -> str:
        s = "{name} (\n{modstr}\n)"
        modstr = "\n".join(["inputs (\n{}\n)".format(_indent(self.inputs, 2))] +
                           ["outputs (\n{}\n)".format(_indent(self.outputs, 2))] +
                           ["body (\n{}\n)".format(_indent(self.operators, 2))])
        return s.format(name=self.name, modstr=_indent(modstr, 2))

class TpuLangConverter(BaseConverter):
    MLIRImporterTypeStr = {
        "float64": "F64",
        "float32": "F32",
        "float16": "F16",
        "int8": "INT8",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
        "uint8": "UINT8",
        "uint16": "UINT16",
        "uint32": "UINT32",
        "uint64": "UINT64",
        "bool": "BOOL",
        "dict": "DICT",
    }

    def __init__(self, name: str, graph: Graph, mode: str, no_save: bool = False):
        super().__init__(no_save=no_save)
        self.model_name = name
        self.model = graph
        self.load_model()
        state = State.TOP_QUANTIZED if mode == "quantized" else State.TOP_F32
        self.init_MLIRImporter(state)
        self.weight_file = self.mlir.weight_file
        self.constant = {}
        self.type_to_mlir = self.__type2mlir(self.mlir.ctx)
        for tensor in self.model.inputs:
            if tensor.is_quantized:
                self.input_types.append(self.get_quantized_type(tensor))
            else:
                self.input_types.append(self.MLIRImporterTypeStr[tensor.dtype])
        for tensor in self.model.outputs:
            if tensor.is_quantized:
                self.output_types.append(self.get_quantized_type(tensor))
            else:
                self.output_types.append(self.MLIRImporterTypeStr[tensor.dtype])
        self.mlir.declare_func(self.input_types, self.output_types)

    def __del__(self):
        if self.mlir != None:
            del self.mlir

    def load_model(self):
        self.num_input = len(self.model.inputs)
        self.input_types = []
        self.output_types = []
        for tensor in self.model.inputs:
            self.input_names.append(tensor.name)
            self.addShape(tensor.name, tensor.shape)
        for tensor in self.model.outputs:
            self.output_names.append(tensor.name)
            self.addShape(tensor.name, tensor.shape)

    def init_MLIRImporter(self, state:State):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.output_names:
            if self.getShape(_name) != [1]:
                output_shapes.append(self.getShape(_name))
            else:
                output_shapes.append([])

        # init importer
        self.mlir = MLIRImporter(input_shapes,
                                 output_shapes,
                                 self.model_name,
                                 platform=Platform.TPULANG,
                                 state=state,
                                 do_declare=False,
                                 no_save=self.no_save)

    def __create_weight_op(self, tensor: Tensor):
        # constant variable/op
        tensor_type = self.__get_tensor_type(tensor)
        name_loc = Location.name(tensor.name)
        attrs = dict()
        if self.no_save:
            attrs["inline_bytes"] = StringAttr.get(tensor.buffer.tobytes())
        else:
            self.constant[tensor.name] = tensor.buffer
        op = Operation.create("top.Weight", results=[tensor_type], loc=name_loc,
                              attributes=attrs)
        self.mlir.insert_point.insert(op)
        return op.results[0]

    def attr_to_mlir(self, params: dict):
        attrs = {}
        def _attr_convert(attr):
            if attr[2] is False:
                return self.mlir.ArrayAttr(attr[0], self.MLIRImporterTypeStr[attr[1]])
            elif attr[1].find("int") >= 0:
                return IntegerAttr.get(self.type_to_mlir[attr[1]], attr[0])
            elif attr[1] == "bool":
                return BoolAttr.get(attr[0])
            elif attr[1] == "string":
                return StringAttr.get(attr[0])
            else:
                return FloatAttr.get(self.type_to_mlir[attr[1]], attr[0])

        for key, value in params.items():
            # DictArrayAttr for custom op
            if value[1] == "dict" and not value[2]:
                array_attr = []
                for i, dict in enumerate(value[0]):
                    sub_dict = {}
                    for k, v in dict[0].items():
                        sub_dict[k] = _attr_convert(v)
                    dict_attr = DictAttr.get(sub_dict)
                    array_attr.append(dict_attr)
                attrs[key] = self.mlir.ArrayAttr(array_attr, self.MLIRImporterTypeStr[value[1]])
            else:
                attrs[key] = _attr_convert(value)
        return attrs

    def __type2mlir(self, mlir_ctx):
        return {
            "float64": F64Type.get(mlir_ctx),
            "float32": F32Type.get(mlir_ctx),
            "float16": F16Type.get(mlir_ctx),
            "int64": IntegerType.get_signless(64, mlir_ctx),
            "int32": IntegerType.get_signed(32, mlir_ctx),
            "int16": IntegerType.get_signed(16, mlir_ctx),
            "int8": IntegerType.get_signed(8, mlir_ctx),
            "uint32": IntegerType.get_unsigned(32, mlir_ctx),
            "uint16": IntegerType.get_unsigned(16, mlir_ctx),
            "uint8": IntegerType.get_unsigned(8, mlir_ctx),
            "bool": IntegerType.get_signless(1, mlir_ctx),
        }

    def get_quantized_type(self, tensor: Tensor):
        if tensor.is_quantized is False:
            raise ValueError("do not have quantized type")
        storage_type = self.type_to_mlir[tensor.dtype]
        # is_constant = tensor.buffer is not None
        is_signed = tensor.dtype == "int8" or tensor.dtype == "int16" or tensor.dtype == "int32"
        storage_min = (
            quant.QuantizedType.default_minimum_for_integer(  # type: ignore
                is_signed, storage_type.width))
        storage_max = quant.QuantizedType.default_maximum_for_integer(  # type: ignore
            is_signed, storage_type.width)
        flags = 1 if is_signed else 0
        scale = tensor.scale if tensor.scale is not None else 1.0
        zero_point = tensor.zero_point if tensor.zero_point != None else 0
        is_perchannel = (isinstance(scale, List) and len(scale) > 1) or (isinstance(zero_point, List) and len(zero_point) > 1)
        if is_perchannel:
            length = len(scale) if isinstance(scale, List) else len(zero_point)
            zero_point = zero_point if isinstance(zero_point, List) or zero_point == None else [zero_point] * length
            return quant.UniformQuantizedPerAxisType.get(  # type: ignore
                flags, storage_type, self.type_to_mlir["float32"],
                scale if isinstance(scale, List) else [scale] * length,
                zero_point, 1,
                storage_min, storage_max)
        else:
            scale = scale[0] if isinstance(scale, List) else scale
            zero_point = zero_point[0] if isinstance(zero_point, List) else zero_point
            return quant.UniformQuantizedType.get(  # type: ignore
                flags, storage_type, self.type_to_mlir["float32"], scale, zero_point, storage_min,
                storage_max)

    def __get_tensor_type(self, tensor: Tensor):
        if tensor is None:
            self.type_to_mlir["float32"]
            return NoneType.get()
        else:
            type = self.type_to_mlir[tensor.dtype]
            output_shapes = tensor.shape
            if tensor.is_quantized:
                type = self.get_quantized_type(tensor)
        if output_shapes == []:
            return UnrankedTensorType.get(type)
        if output_shapes is None:
            return NoneType.get()
        if isinstance(output_shapes, tuple):
            output_shapes = list(output_shapes)
        assert (isinstance(output_shapes, list))
        assert (len(output_shapes) > 0)
        if not isinstance(output_shapes[0], list) and output_shapes[0] is not None:
            return RankedTensorType.get(tuple(output_shapes), type)
        # multi output
        out_types = []
        for s in output_shapes:
            if s == []:
                out_types.append(UnrankedTensorType.get(type))
            elif s is None:
                out_types.append(NoneType.get())
            else:
                out_types.append(RankedTensorType.get(tuple(s), type))
        return out_types

    def __get_tensor_loc(self, tensor: Tensor):
        if tensor is None:
            return Location.unknown(self.mlir.mlir_module.context)
        else:
            return Location.name(tensor.name)

    def convert_subgraph(self, subgraph: Graph):

        class symbolTable:
            symbol_table = {}

            def __init__(self, gen_value_func):
                self.gen_value_func = gen_value_func

            def __getitem__(self, tensor: Tensor):
                if tensor is None: return self.symbol_table[-1]
                if tensor.id not in self.symbol_table:
                    if tensor.buffer is None and tensor.ttype != 'coeff':
                        raise Exception("Tensor '{}' is not constant!".format(tensor.name))
                    if tuple(tensor.shape) != tuple(tensor.buffer.shape):
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
            p_dict = input.preprocess_to_dict()
            if input.is_preprocess:
                p_dict["preprocess_list"] = [idx + 1]
            input_op = self.mlir.create_input_op(loc, idx, p_dict)
            symbol_table.update({input.id: input_op})

        def add_operation(operation: Operator):
            # get operation attributes
            attributes = self.attr_to_mlir(operation.params)
            # get input operands
            operands = []
            for tensor in operation.inputs:
                operands.append(symbol_table[tensor] if tensor is not None else self.mlir.none_op)
            rst_type = [self.__get_tensor_type(x) for x in operation.outputs]
            name_loc = Location.fused([self.__get_tensor_loc(x) for x in operation.outputs])
            op = Operation.create(
                operation.op_name,
                results=rst_type,
                operands=operands,
                attributes=attributes,
                loc=name_loc,
            )
            self.mlir.insert_point.insert(op)
            for x, res in zip(operation.outputs, op.results):
                if x is not None:
                    symbol_table.update({x.id : res})
                else:
                    symbol_table.update({-1 : res})

        return_op = []
        graph_outs = dict()
        for op in subgraph.operators:
            try :
                add_operation(op)
            except:
                raise Exception("Error: Convert operation {} failed!".format(op.op_name))
            for out in op.outputs:
                if out is None: continue
                if out.name in self.output_names:
                    graph_outs[out.name] = out

        for out_name in self.output_names:
            return_op.append(symbol_table[graph_outs[out_name]])

        return return_op

    def get_mlir_txt(self):
        return_op = self.convert_subgraph(self.model)
        self.mlir.create_return_op(return_op)
        return self.mlir.print_module()

    def generate_mlir(self, mlir_file: str, log_level="normal"):
        mlir_txt = self.get_mlir_txt()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
            if log_level != "quiet":
                logger.info("Save mlir file: {}".format(mlir_file))
        for k,v in self.constant.items():
            if v.dtype == 'float16':
                self.constant[k] = v.view('uint16')
        np.savez(self.weight_file, **self.constant)
        if log_level != "quiet":
            logger.info("Save weight file: {}".format(self.weight_file))
