# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from mlir.ir import *
from typing import List


class Top:
    # NOTICE: Please add the Op alphabetically !!!
    AbsOp = 'top.Abs'
    AdaptiveAvgPoolOp = 'top.AdaptiveAvgPool'
    AddOp = 'top.Add'
    ArgOp = 'top.Arg'
    AddConstOp = 'top.AddConst'
    AvgPoolOp = 'top.AvgPool'
    BatchNormOp = 'top.BatchNorm'
    ClipOp = 'top.Clip'
    ConcatOp = 'top.Concat'
    ConvOp = 'top.Conv'
    CompareOp = 'top.Compare'
    CompareConstOp = 'top.CompareConst'
    ConstantFillOp = 'top.ConstantFill'
    CosOp = 'top.Cos'
    CoshOp = 'top.Cosh'
    Depth2SpaceOp = 'top.Depth2Space'
    DequantizeLinearOp = 'top.DequantizeLinear'
    DeconvOp = 'top.Deconv'
    DetectionOutputOp = 'top.DetectionOutput'
    DivOp = 'top.Div'
    EluOp = 'top.Elu'
    ErfOp = 'top.Erf'
    ExpOp = 'top.Exp'
    FlattenOp = 'top.Flatten'
    FloorOp = 'top.Floor'
    FrcnDetection = 'top.FrcnDetection'
    GatherOp = 'top.Gather'
    GELUOp = 'top.GELU'
    GroupNormOp = 'top.GroupNorm'
    GRUOp = 'top.GRU'
    HardSigmoidOp = 'top.HardSigmoid'
    HardSwishOp = 'top.HardSwish'
    InputOp = 'top.Input'
    InstanceNormOp = 'top.InstanceNorm'
    InterpOp = 'top.Interp'
    LayerNormOp = 'top.LayerNorm'
    LeakyReluOp = 'top.LeakyRelu'
    ListOp = 'top.List'
    LRNOp = 'top.LRN'
    LSTMOp = 'top.LSTM'
    LogOp = 'top.Log'
    MaskedFillOp = 'top.MaskedFill'
    MatMulOp = 'top.MatMul'
    MaxPoolOp = 'top.MaxPool'
    MaxPoolWithMaskOp = 'top.MaxPoolWithMask'
    MaxUnpoolOp = 'top.MaxUnpool'
    MaxOp = 'top.Max'
    MinOp = 'top.Min'
    MishOp = 'top.Mish'
    MulOp = 'top.Mul'
    MulConstOp = 'top.MulConst'
    NonZeroOp = 'top.NonZero'
    NormalizeOp = 'top.Normalize'
    PermuteOp = 'top.Permute'
    PadOp = 'top.Pad'
    PackOp = 'top.Pack'
    PixelNormOp = 'top.PixelNorm'
    PowOp = 'top.Pow'
    PriorBoxOp = 'top.PriorBox'
    PReluOp = 'top.PRelu'
    Proposal = 'top.Proposal'
    QuantizeLinearOp = 'top.QuantizeLinear'
    Reciprocal = 'top.Reciprocal'
    ReshapeOp = 'top.Reshape'
    ReluOp = 'top.Relu'
    ReduceOp = 'top.Reduce'
    ReverseOp = 'top.Reverse'
    RoiAlignOp = 'top.RoiAlign'
    ROIPooling = 'top.ROIPooling'
    RetinaFaceDetection = 'top.RetinaFaceDetection'
    ScatterElementsOp = 'top.ScatterElements'
    ScatterNDOp = 'top.ScatterND'
    SubOp = 'top.Sub'
    SliceOp = 'top.Slice'
    SliceAxisOp = 'top.SliceAxis'
    ShapeOp = 'top.Shape'
    SigmoidOp = 'top.Sigmoid'
    SiLUOp = 'top.SiLU'
    SizeOp = 'top.Size'
    SinOp = 'top.Sin'
    SinhOp = 'top.Sinh'
    SoftmaxOp = 'top.Softmax'
    SoftplusOp = 'top.Softplus'
    SqueezeOp = 'top.Squeeze'
    ScaleOp = 'top.Scale'
    SubOp = 'top.Sub'
    SplitOp = 'top.Split'
    SqrtOp = 'top.Sqrt'
    ShuffleChannelOp = 'top.ShuffleChannel'
    TileOp = 'top.Tile'
    RepeatOp = 'top.Repeat'
    TanOp = 'top.Tan'
    TanhOp = 'top.Tanh'
    TopKOp = 'top.TopK'
    TransposeOp = 'top.Transpose'
    TupleOp = 'top.Tuple'
    UnTupleOp = 'top.UnTuple'
    UnpackOp = 'top.Unpack'
    UnsqueezeOp = 'top.Unsqueeze'
    UpsampleOp = 'top.Upsample'
    ViewOp = 'top.View'
    WeightOp = 'top.Weight'
    WhereOp = 'top.Where'
    YoloDetection = 'top.YoloDetection'
    ZerosOp = 'top.Zeros'
    IfOp = 'top.If'
    LoopOp = 'top.Loop'

class State:
    TOP_F32 = 'TOP_F32'
    TOP_QUANTIZED = 'TOP_QUANTIZED'


class Platform:
    ONNX = "ONNX"
    TORCH = "TORCH"
    TFLITE = "TFLITE"
    CAFFE = "CAFFE"
    TPULANG = "TPULANG"


def get_weight_file(model_name: str, state: str, chip: str):
    name = "{}_{}_{}_origin_weight.npz".format(model_name, state, chip)
    return name.lower()


def checkType(obj, type):
    if not isinstance(obj, type):
        raise AttributeError('{} is not {}'.format(obj, type))


class MLIRImporter(object):

    def __init__(self,
                 input_shapes: list,
                 output_shapes: list,
                 model_name: str,
                 platform: str = Platform.ONNX,
                 input_types: list = [],
                 output_types: list = [],
                 state: str = State.TOP_F32,
                 do_declare: bool = True):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert (len(model_name) > 0)
        self.model_name = model_name
        self.state = state
        self.chip = "ALL"
        self.platform = platform
        self.weight_file = get_weight_file(self.model_name, self.state, self.chip)
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.num_input = len(self.input_shapes)
        self.num_output = len(self.output_shapes)
        self.load_weight = dict()
        self.F32Type = F32Type.get()
        self.insert_point_save_flag = False
        self.mlir_type = {
            "INT8": IntegerType.get_signless(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signless(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signed(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get()
        }
        if do_declare:
            self.declare_func(input_types, output_types)

    def __del__(self):
        try:
            self.loc.__exit__(None, None, None)
        except:
            pass
        try:
            self.ctx.__exit__(None, None, None)
        except:
            pass

    def ArrayAttr(self, data: list, data_type: str = 'INT64'):
        assert (data_type in self.mlir_type)
        if data_type.find("INT") >= 0:
            return ArrayAttr.get([IntegerAttr.get(self.mlir_type[data_type], x) for x in data])
        if data_type == 'F32':
            return ArrayAttr.get([FloatAttr.get_f32(x) for x in data])
        if data_type == 'F64':
            return ArrayAttr.get([FloatAttr.get_f64(x) for x in data])
        raise RuntimeError("unsupport data type:{}".format(data_type))

    # shape: [] => [* x f32]; None => NoneType; [None, None] => [NoneType, NoneType]
    # type: None => f32; or type
    def get_tensor_type(self, output_shapes, type=None):
        if type is None:
            type = self.F32Type
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

    def get_value_type(self, value):
        _type = str(value.type)
        _type = _type.split('<')[-1].split('x')[-1].split('>')[0]
        if _type == "f32":
            return self.mlir_type['F32']
        elif _type == "i8":
            return self.mlir_type['INT8']
        elif _type == "ui8":
            return self.mlir_type['UINT8']
        elif _type == "i32" or _type == "si32":
            return self.mlir_type['INT32']
        else:
            raise RuntimeError("No support {}".format(_type))

    def buildOp(self, op_type, operands, output_types: list, region_num = 0, **kargs):
        """
            op_type: String
            inputOpreands: List[pybind.op]
            output_types: List[pybind.op]
            kargs: Dict
        """
        num_output = len(output_types)
        names = kargs['name']
        if isinstance(names, str):
            loc = Location.fused([Location.name(names)])
        elif isinstance(names, list):
            loc = Location.fused([Location.name(n) for n in names])
        else:
            raise RuntimeError("Unknown names:{}".format(names))
        del kargs["name"]
        op = Operation.create(
            op_type,
            results=output_types,
            operands=operands,
            loc=loc,
            attributes=kargs,
            regions=region_num,
        )
        self.insert_point.insert(op)

        assert (num_output == len(output_types))
        if num_output > 1:
            return tuple(op.results)
        else:
            return op.result

    def buildBlock(self, region, arg_types, **kargs):
        block = Block.create_at_start(region, arg_types)

    def reconfig_insert_point(self, block):
        self.insert_point_back = self.insert_point \
                        if not self.insert_point_save_flag else self.insert_point_back
        self.insert_point = InsertionPoint(block)
        self.insert_point_save_flag = True

    def restore_insert_point(self):
        self.insert_point = self.insert_point_back
        self.insert_point_save_flag = False

    def create_input_op(self, name, index, **kargs):
        assert (index < len(self.func_args))
        param = {}
        if 'scale' in kargs:
            param['scale'] = ArrayAttr.get([FloatAttr.get_f64(x) for x in kargs['scale']])
        if 'mean' in kargs:
            param['mean'] = ArrayAttr.get([FloatAttr.get_f64(x) for x in kargs['mean']])
        if 'resize_dims' in kargs:
            param['resize_dims'] = ArrayAttr.get(
                [IntegerAttr.get(self.mlir_type['INT64'], x) for x in kargs['resize_dims']])
        if 'keep_aspect_ratio' in kargs:
            param['keep_aspect_ratio'] = BoolAttr.get(kargs['keep_aspect_ratio'])
        if 'pad_type' in kargs:
            param['pad_type'] = StringAttr.get(kargs['pad_type'])
        if 'pad_value' in kargs:
            param['pad_value'] = IntegerAttr.get(self.mlir_type['INT64'], kargs['pad_value'])
        if 'pixel_format' in kargs:
            param['pixel_format'] = StringAttr.get(kargs['pixel_format'])
        if 'channel_format' in kargs:
            param['channel_format'] = StringAttr.get(kargs['channel_format'])
        if 'model_format' in kargs:
            param['model_format'] = StringAttr.get(kargs['model_format'])
        op = Operation.create(Top.InputOp,
                              results=[self.input_types[index]],
                              operands=[self.func_args[index]],
                              loc=Location.fused([Location.name(name)]),
                              attributes=param)
        self.insert_point.insert(op)
        return op.results[0]

    def create_weight_op(self, name, output_shape, data_type="F32"):
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != output_shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        tensor_type = RankedTensorType.get(output_shape, self.mlir_type[data_type])
        op = Operation.create(Top.WeightOp,
                              results=[tensor_type],
                              loc=Location.fused([Location.name(name)]))
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, output_shape, data_type)
        return result

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def create_yield_op(self, Operands):
        yield_op = Operation.create("Top.YieldOp", operands=Operands, results=[])
        self.insert_point.insert(yield_op)
        return yield_op

    def create_if_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        region = IntegerAttr.get(self.mlir_type['INT64'], kargs["region"]).value
        return self.buildOp(Top.IfOp, operands, [output_type], region, **param)

    def create_loop_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        region = IntegerAttr.get(self.mlir_type['INT64'], kargs["region"]).value
        return self.buildOp(Top.LoopOp, operands, output_type, region, **param)

    def create_subgraph_input_op(self, name, type, val, **kargs):
        param = {}
        op = Operation.create(Top.InputOp,
                              results=[type],
                              operands=[val],
                              loc=Location.fused([Location.name(name)]),
                              attributes=param)
        self.insert_point.insert(op)
        return op.results[0]

    def create_range_op(self, operands, output_shape, **kargs):
        # output_type = self.get_tensor_type(output_shape)
        # param = {'name': kargs['name']}
        # return self.buildOp(Top.RangeOp, operands, [output_type], **param)
        pass

    def print_module(self):
        mlir_format = self.mlir_module.operation.get_asm(enable_debug_info=True)
        return mlir_format

    def declare_func(self, input_types: list = [], output_types: list = []):
        if len(input_types) == 0:
            input_types = self.num_input * ['F32']
        if len(output_types) == 0:
            output_types = self.num_output * ['F32']

        self.input_types = list()
        self.output_types = list()
        for _shape, _type in zip(self.input_shapes, input_types):
            if isinstance(_type, str):
                self.input_types.append(RankedTensorType.get(_shape, self.mlir_type[_type]))
            else:
                self.input_types.append(RankedTensorType.get(_shape, _type))
        for _shape, _type in zip(self.output_shapes, output_types):
            t = _type
            if isinstance(_type, str):
                t = self.mlir_type[_type]
            self.output_types.append(self.get_tensor_type(_shape, t))
        args_txt = str()
        for _idx, _type in enumerate(self.input_types):
            args_txt += "%args{}: {} loc(unknown)".format(_idx, _type.__str__())
            if (_idx + 1) < self.num_input:
                args_txt += ", "

        output_txt = str()
        for _idx, _type in enumerate(self.output_types):
            output_txt += _type.__str__()
            if (_idx + 1) < self.num_output:
                output_txt += ", "
        if self.num_output > 1:
            output_txt = "({})".format(output_txt)
        main_func = """
            module attributes {{module.name = \"{name}\", module.weight_file= \"{weight_file}\", module.platform=\"{platform}\", module.state=\"{state}\", module.chip=\"{chip}\"}} {{
                func.func @main({args}) -> {output} {{
                    %0 = \"top.None\"() : () -> none loc(unknown)
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                   weight_file=self.weight_file,
                   platform=self.platform,
                   state=self.state,
                   chip=self.chip,
                   args=args_txt,
                   output=output_txt)
        self.mlir_module = Module.parse(main_func, self.ctx)
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.none_op = self.entry_block.operations[0].operation.results[0]

        self.func_args = list()
        for i in self.entry_block.arguments:
            self.func_args.append(i)
