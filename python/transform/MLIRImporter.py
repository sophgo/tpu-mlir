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
    Depth2SpaceOp = 'top.Depth2Space'
    DequantizeLinearOp = 'top.DequantizeLinear'
    DeconvOp = 'top.Deconv'
    DetectionOutputOp = 'top.DetectionOutput'
    DivOp = 'top.Div'
    EluOp = 'top.Elu'
    ErfOp = 'top.Erf'
    ExpOp = 'top.Exp'
    FloorOp = 'top.Floor'
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
    QuantizeLinearOp = 'top.QuantizeLinear'
    Reciprocal = 'top.Reciprocal'
    ReshapeOp = 'top.Reshape'
    ReluOp = 'top.Relu'
    ReduceOp = 'top.Reduce'
    ReverseOp = 'top.Reverse'
    RoiAlignOp = 'top.RoiAlign'
    ScatterElementsOp = 'top.ScatterElements'
    ScatterNDOp = 'top.ScatterND'
    SubOp = 'top.Sub'
    SliceOp = 'top.Slice'
    SigmoidOp = 'top.Sigmoid'
    SiLUOp = 'top.SiLU'
    SoftmaxOp = 'top.Softmax'
    SoftplusOp = 'top.Softplus'
    SqueezeOp = 'top.Squeeze'
    ScaleOp = 'top.Scale'
    SubOp = 'top.Sub'
    SplitOp = 'top.Split'
    SqrtOp = 'top.Sqrt'
    ShuffleChannelOp = 'top.ShuffleChannel'
    TileOp = 'top.Tile'
    TileExOp = 'top.TileEx'
    TanhOp = 'top.Tanh'
    TopKOp = 'top.TopK'
    TransposeOp = 'top.Transpose'
    TupleOp = 'top.Tuple'
    UnpackOp = 'top.Unpack'
    UpsampleOp = 'top.Upsample'
    WeightOp = 'top.Weight'
    WhereOp = 'top.Where'
    YoloDetection = 'top.YoloDetection'
    Proposal = 'top.Proposal'
    ROIPooling = 'top.ROIPooling'
    FrcnDetection = 'top.FrcnDetection'
    RetinaFaceDetection = 'top.RetinaFaceDetection'


class State:
    TOP_F32 = 'TOP_F32'
    TOP_QUANTIZED = 'TOP_QUANTIZED'


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

    def get_tensor_type(self, output_shape: list, type=None):
        type = F32Type.get() if type is None else type
        if output_shape is None:
            return UnrankedTensorType.get(type)
        else:
            return RankedTensorType.get(tuple(output_shape), type)

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

    def buildOp(self, op_type, operands, output_types: list, **kargs):
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
        )
        self.insert_point.insert(op)

        assert (num_output == len(output_types))
        if num_output > 1:
            return tuple(op.results)
        else:
            return op.result

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

    def create_add_op(self, operands, output_shape, **kargs):
        if len(operands) < 2:
            raise RuntimeError("input operand must great than 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        if "coeff" in kargs:
            param['coeff'] = self.ArrayAttr(kargs['coeff'], 'F64')
        return self.buildOp(Top.AddOp, operands, [output_type], **param)

    def create_sub_op(self, operands, output_shape, **kargs):
        if len(operands) < 2:
            raise RuntimeError("input operand must great than 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        if "coeff" in kargs:
            param['coeff'] = self.ArrayAttr(kargs['coeff'], 'F64')
        return self.buildOp(Top.SubOp, operands, [output_type], **param)

    def create_mul_op(self, operands, output_shape, **kargs):
        if len(operands) < 2:
            raise RuntimeError("input operand must great than 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.MulOp, operands, [output_type], **param)

    def create_pow_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'exponent': FloatAttr.get_f64(kargs['exponent'])}
        return self.buildOp(Top.PowOp, operands, [output_type], **param)

    def create_add_const_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'const_val': FloatAttr.get_f64(kargs['const_val']),
            'do_relu': BoolAttr.get(False)
        }
        return self.buildOp(Top.AddConstOp, operands, [output_type], **param)

    def create_mul_const_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'const_val': FloatAttr.get_f64(kargs['const_val']),
            'do_relu': BoolAttr.get(False)
        }
        return self.buildOp(Top.MulConstOp, operands, [output_type], **param)

    def create_avgpool_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'kernel_shape': self.ArrayAttr(kargs['kernel_shape']),
            'strides': self.ArrayAttr(kargs['strides']),
            'pads': self.ArrayAttr(kargs['pads']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            'count_include_pad': BoolAttr.get(kargs['count_include_pad']),
        }
        return self.buildOp(Top.AvgPoolOp, operands, [output_type], **param)

    def create_batchnorm_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'epsilon': FloatAttr.get_f64(kargs['epsilon']),
        }
        return self.buildOp(Top.BatchNormOp, operands, [output_type], **param)

    def create_concat_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
        }
        return self.buildOp(Top.ConcatOp, operands, [output_type], **param)

    def create_conv_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'kernel_shape': self.ArrayAttr(kargs['kernel_shape']),
            'strides': self.ArrayAttr(kargs['strides']),
            'dilations': self.ArrayAttr(kargs['dilations']),
            'pads': self.ArrayAttr(kargs['pads']),
            'group': IntegerAttr.get(self.mlir_type['INT64'], kargs['group']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        if 'inserts' in kargs:
            param['inserts'] = self.ArrayAttr(kargs['inserts'])
        return self.buildOp(Top.ConvOp, operands, [output_type], **param)

    def create_depth2space_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        p = {
            'name': kargs['name'],
            "block_h": IntegerAttr.get(self.mlir_type['INT64'], kargs['block_h']),
            "block_w": IntegerAttr.get(self.mlir_type['INT64'], kargs['block_w']),
            "is_CRD": BoolAttr.get(kargs['is_CRD']),
            "is_inversed": BoolAttr.get(kargs['is_inversed']),
            "in_is_NCHW": BoolAttr.get(True),
            "out_is_NCHW": BoolAttr.get(True),
            "swap_cr": BoolAttr.get(False),
        }
        return self.buildOp(Top.Depth2SpaceOp, operands, [output_type], **p)

    def create_leaky_relu_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'alpha': FloatAttr.get_f64(kargs['alpha']),
        }
        return self.buildOp(Top.LeakyReluOp, operands, [output_type], **param)

    def create_maxpool_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'kernel_shape': self.ArrayAttr(kargs['kernel_shape']),
            'strides': self.ArrayAttr(kargs['strides']),
            'pads': self.ArrayAttr(kargs['pads']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        return self.buildOp(Top.MaxPoolOp, operands, [output_type], **param)

    def create_maxpool_with_mask_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'kernel_shape': self.ArrayAttr(kargs['kernel_shape']),
            'strides': self.ArrayAttr(kargs['strides']),
            'pads': self.ArrayAttr(kargs['pads']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        return self.buildOp(Top.MaxPoolWithMaskOp, operands, [output_type, output_type], **param)

    def create_permute_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'order': self.ArrayAttr(kargs['order']),
        }
        return self.buildOp(Top.PermuteOp, operands, [output_type], **param)

    def create_transpose_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'dim0': IntegerAttr.get(self.mlir_type['INT64'], kargs['dim0']),
            'dim1': IntegerAttr.get(self.mlir_type['INT64'], kargs['dim1']),
        }
        return self.buildOp(Top.TransposeOp, operands, [output_type], **param)

    def create_matmul_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        return self.buildOp(Top.MatMulOp, operands, [output_type], **param)

    def create_normalize_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'across_spatial': BoolAttr.get(kargs['across_spatial']),
            'channel_shared': BoolAttr.get(kargs['channel_shared']),
        }
        return self.buildOp(Top.NormalizeOp, operands, [output_type], **param)

    def create_relu_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        if 'relu_limit' in kargs:
            param['relu_limit'] = FloatAttr.get_f64(kargs['relu_limit'])
        return self.buildOp(Top.ReluOp, operands, [output_type], **param)

    def create_tuple_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.TupleOp, operands, [output_type], **param)

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def create_reshape_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        return self.buildOp(Top.ReshapeOp, operands, [output_type], name=kargs['name'])

    def create_unsqueeze_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        return self.buildOp(Top.ReshapeOp, operands, [output_type], name=kargs['name'])

    def create_reverse_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis'])
        }
        return self.buildOp(Top.ReverseOp, operands, [output_type], **param)

    def create_slice_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'offset': self.ArrayAttr(kargs['offset']),
            'steps': self.ArrayAttr(kargs['steps']),
        }
        return self.buildOp(Top.SliceOp, operands, [output_type], **param)

    def create_sigmoid_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'scale': FloatAttr.get_f64(kargs['scale']),
            'bias': FloatAttr.get_f64(kargs['bias']),
        }
        if 'log' in kargs:
            param['log'] = BoolAttr.get(kargs['log'])
        return self.buildOp(Top.SigmoidOp, operands, [output_type], **param)

    def create_upsample_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'scale_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_h']),
            'scale_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_w']),
        }
        return self.buildOp(Top.UpsampleOp, operands, [output_type], **param)

    def create_interp_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'scale_h': FloatAttr.get_f64(kargs['scale_h']),
            'scale_w': FloatAttr.get_f64(kargs['scale_w']),
            'mode': StringAttr.get(kargs['mode']),
            'coord_mode': StringAttr.get(kargs['coordinate_transformation_mode'])
        }
        return self.buildOp(Top.InterpOp, operands, [output_type], **param)

    def create_maxunpool_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'scale_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_h']),
            'scale_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_w']),
        }
        return self.buildOp(Top.MaxUnpoolOp, operands, [output_type], **param)

    def create_softmax_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
        }
        if 'log' in kargs:
            param['log'] = BoolAttr.get(kargs['log'])
        return self.buildOp(Top.SoftmaxOp, operands, [output_type], **param)

    def create_softplus_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.SoftplusOp, operands, [output_type], **param)

    def create_log_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.LogOp, operands, [output_type], **param)

    def create_exp_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.ExpOp, operands, [output_type], **param)

    def create_tanh_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.TanhOp, operands, [output_type], **param)

    def create_mish_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.MishOp, operands, [output_type], **param)

    def create_elu_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'alpha': FloatAttr.get_f64(kargs['alpha']),
        }
        return self.buildOp(Top.EluOp, operands, [output_type], **param)

    def create_erf_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.ErfOp, operands, [output_type], **param)

    def create_pad_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'paddings': self.ArrayAttr(kargs['paddings']),
            'val': FloatAttr.get_f64(kargs['val']),
            'mode': IntegerAttr.get(self.mlir_type['INT64'], kargs['mode'])
        }
        return self.buildOp(Top.PadOp, operands, [output_type], **param)

    def create_div_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must be 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.DivOp, operands, [output_type], **param)

    def create_reciprocal_op(self, operands, output_shape, **kargs):
        if len(operands) != 1:
            raise RuntimeError("input operand must be 1")
        output_type = self.get_tensor_type(output_shape)
        param = {
            "name": kargs["name"],
            "do_relu": BoolAttr.get(False),
            "const_val": FloatAttr.get_f64(kargs["const_val"]),
        }
        return self.buildOp(Top.Reciprocal, operands, [output_type], **param)

    def create_squeeze_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axes': self.ArrayAttr(kargs['axes']),
        }
        return self.buildOp(Top.SqueezeOp, operands, [output_type], **param)

    def create_clip_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'min': FloatAttr.get_f64(kargs['min']),
            'max': FloatAttr.get_f64(kargs['max']),
        }
        return self.buildOp(Top.ClipOp, operands, [output_type], **param)

    def create_conv_transpose_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)

        param = {
            'name': kargs['name'],
            'kernel_shape': self.ArrayAttr(kargs['kernel_shape']),
            'strides': self.ArrayAttr(kargs['strides']),
            # 'dilations': self.ArrayAttr(kargs['dilations']),
            'pads': self.ArrayAttr(kargs['pads']),
            'group': IntegerAttr.get(self.mlir_type['INT64'], kargs['group']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        if 'inserts' in kargs:
            param['inserts'] = self.ArrayAttr(kargs['inserts'])
        return self.buildOp(Top.DeconvOp, operands, [output_type], **param)

    def create_scale_op(self, operands, output_shape, **kargs):
        if len(operands) < 3:
            raise RuntimeError("input operand must equal to 3")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.ScaleOp, operands, [output_type], **param)

    def create_lrn_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            "name": kargs["name"],
            "size": IntegerAttr.get(self.mlir_type["INT64"], kargs["size"]),
        }

        def add_if_has(key):
            if key in kargs:
                param[key] = FloatAttr.get_f64(kargs[key])

        add_if_has("alpha")
        add_if_has("beta")
        add_if_has("bias")

        return self.buildOp(Top.LRNOp, operands, [output_type], **param)

    def create_gru_op(self, operands, out_shapes, **kargs):
        out_types = list()
        for s in out_shapes:
            if s is not None and len(s) == 0:
                out_types.append(NoneType.get())
            else:
                out_types.append(self.get_tensor_type(s))
        param = {
            'name': kargs['name'],
            'hidden_size': IntegerAttr.get(self.mlir_type["INT64"], kargs["hidden_size"]),
            'bidirectional': BoolAttr.get(kargs['bidirectional']),
            'batch_first': BoolAttr.get(kargs['batch_first']),
        }
        return self.buildOp(Top.GRUOp, operands, out_types, **param)

    def create_lstm_op(self, operands, out_shapes, **kargs):
        out_types = list()
        for s in out_shapes:
            if s is not None and len(s) == 0:
                out_types.append(NoneType.get())
            else:
                out_types.append(self.get_tensor_type(s))
        param = {
            'name': kargs['name'],
            'hidden_size': IntegerAttr.get(self.mlir_type["INT64"], kargs["hidden_size"]),
            'bidirectional': BoolAttr.get(kargs['bidirectional']),
            'batch_first': BoolAttr.get(kargs['batch_first']),
        }
        return self.buildOp(Top.LSTMOp, operands, out_types, **param)

    def create_topk_op(self, operands, out_shapes, **kargs):
        out_types = list()
        for s in out_shapes:
            if s is not None and len(s) == 0:
                out_types.append(NoneType.get())
            else:
                out_types.append(self.get_tensor_type(s))
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type["INT64"], kargs["axis"]),
            'K': IntegerAttr.get(self.mlir_type["INT64"], kargs["K"]),
            'largest': BoolAttr.get(kargs['largest']),
            'sorted': BoolAttr.get(kargs['sorted']),
        }
        return self.buildOp(Top.TopKOp, operands, out_types, **param)

    def create_gather_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
        }
        return self.buildOp(Top.GatherOp, operands, [output_type], **param)

    def create_tile_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
            'tile': IntegerAttr.get(self.mlir_type['INT64'], kargs['tile']),
        }
        return self.buildOp(Top.TileOp, operands, [output_type], **param)

    def create_tile_ex_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'repeats': self.ArrayAttr(kargs['repeats']),
        }
        return self.buildOp(Top.TileExOp, operands, [output_type], **param)

    def create_max_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.MaxOp, operands, [output_type], **param)

    def create_min_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.MinOp, operands, [output_type], **param)

    def create_abs_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.AbsOp, operands, [output_type], **param)

    def create_prelu_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.PReluOp, operands, [output_type], **param)

    def create_reduce_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'axes': self.ArrayAttr(kargs['axes']),
            'keepdims': IntegerAttr.get(self.mlir_type['INT64'], kargs['keepdims']),
            'mode': StringAttr.get(kargs['mode']),
        }
        return self.buildOp(Top.ReduceOp, operands, [output_type], **param)

    def create_arg_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        out_types = list()
        for s in output_shape:
            if s is not None and len(s) == 0:
                out_types.append(NoneType.get())
            else:
                out_types.append(self.get_tensor_type(s))
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
            'keepdims': IntegerAttr.get(self.mlir_type['INT64'], kargs['keepdims']),
            'mode': StringAttr.get(kargs['mode']),
        }
        return self.buildOp(Top.ArgOp, operands, out_types, **param)

    def create_sqrt_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.SqrtOp, operands, [output_type], **param)

    def create_where_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'x_is_const': BoolAttr.get(kargs['x_is_const']),
            'y_is_const': BoolAttr.get(kargs['x_is_const']),
            'x_const_val': FloatAttr.get_f64(kargs['x_const_val']),
            'y_const_val': FloatAttr.get_f64(kargs['y_const_val'])
            }
        return self.buildOp(Top.WhereOp, operands, [output_type], **param)

    def create_masked_fill_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'inversed': BoolAttr.get(kargs['inversed']),
            'const_val': FloatAttr.get_f64(kargs['const_val'])
        }
        return self.buildOp(Top.MaskedFillOp, operands, [output_type], **param)

    def create_compare_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'mode': StringAttr.get(kargs['mode'])}
        return self.buildOp(Top.CompareOp, operands, [output_type], **param)

    def create_compare_const_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'mode': StringAttr.get(kargs['mode']),
            'const_val': FloatAttr.get_f64(kargs['const_val']),
            'inversed': BoolAttr.get(kargs['inversed'])
        }
        return self.buildOp(Top.CompareConstOp, operands, [output_type], **param)

    def create_hsigmoid_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'alpha': FloatAttr.get_f64(kargs['alpha']),
            'beta': FloatAttr.get_f64(kargs['beta'])
        }
        return self.buildOp(Top.HardSigmoidOp, operands, [output_type], **param)

    def create_hswish_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.HardSwishOp, operands, [output_type], **param)

    def create_gelu_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.GELUOp, operands, [output_type], **param)

    def create_priorbox_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'min_size': self.ArrayAttr(kargs['min_size'], 'F64'),
            'max_size': self.ArrayAttr(kargs['max_size'], 'F64'),
            'aspect_ratios': self.ArrayAttr(kargs['aspect_ratios'], 'F64'),
            'variance': self.ArrayAttr(kargs['variance'], 'F64'),
            'clip': BoolAttr.get(kargs['clip']),
            'step_h': FloatAttr.get_f64(kargs['step_h']),
            'step_w': FloatAttr.get_f64(kargs['step_w']),
            'img_h': IntegerAttr.get(self.mlir_type['INT64'], int(kargs['step_h'])),
            'img_w': IntegerAttr.get(self.mlir_type['INT64'], int(kargs['step_w'])),
            'offset': FloatAttr.get_f64(kargs['offset']),
            'num_priors': IntegerAttr.get(self.mlir_type['INT64'], kargs['num_priors']),
            'use_default_aspect_ratio': BoolAttr.get(kargs['use_default_aspect_ratio']),
        }
        return self.buildOp(Top.PriorBoxOp, operands, [output_type], **param)

    def create_detection_output_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name':
            kargs['name'],
            'num_classes':
            IntegerAttr.get(self.mlir_type['INT64'], kargs['num_classes']),
            'share_location':
            BoolAttr.get(kargs['share_location']),
            'background_label_id':
            IntegerAttr.get(self.mlir_type['INT64'], kargs['background_label_id']),
            'nms_threshold':
            FloatAttr.get_f64(kargs['nms_threshold']),
            'top_k':
            IntegerAttr.get(self.mlir_type['INT64'], kargs['top_k']),
            'code_type':
            StringAttr.get(kargs['code_type']),
            'keep_top_k':
            IntegerAttr.get(self.mlir_type['INT64'], kargs['keep_top_k']),
            'confidence_threshold':
            FloatAttr.get_f64(kargs['confidence_threshold']),
        }
        return self.buildOp(Top.DetectionOutputOp, operands, [output_type], **param)

    def create_yolo_detection_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'class_num': IntegerAttr.get(self.mlir_type['INT64'], kargs['class_num']),
            'net_input_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['net_input_h']),
            'net_input_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['net_input_w']),
            'keep_topk': IntegerAttr.get(self.mlir_type['INT64'], kargs['keep_topk']),
            'spp_net': BoolAttr.get(kargs['spp_net']),
            'tiny': BoolAttr.get(kargs['tiny']),
            'yolo_v4': BoolAttr.get(kargs['yolo_v4']),
            'nms_threshold': FloatAttr.get_f64(kargs['nms_threshold']),
            'obj_threshold': FloatAttr.get_f64(kargs['obj_threshold']),
            'anchors': StringAttr.get(kargs['anchors']),
        }
        return self.buildOp(Top.YoloDetection, operands, [output_type], **param)

    def create_shuffle_channel_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'group': IntegerAttr.get(self.mlir_type['INT64'], kargs['group']),
        }
        return self.buildOp(Top.ShuffleChannelOp, operands, [output_type], **param)

    def create_proposal_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'net_input_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['net_input_h']),
            'net_input_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['net_input_w']),
            'feat_stride': IntegerAttr.get(self.mlir_type['INT64'], kargs['feat_stride']),
            'anchor_base_size': IntegerAttr.get(self.mlir_type['INT64'], kargs['anchor_base_size']),
            'rpn_obj_threshold': FloatAttr.get_f64(kargs['rpn_obj_threshold']),
            'rpn_nms_threshold': FloatAttr.get_f64(kargs['rpn_nms_threshold']),
            'rpn_nms_post_top_n': IntegerAttr.get(self.mlir_type['INT64'],
                                                  kargs['rpn_nms_post_top_n'])
        }
        return self.buildOp(Top.Proposal, operands, [output_type], **param)

    def create_roipooling_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'pooled_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['pooled_h']),
            'pooled_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['pooled_w']),
            'spatial_scale': FloatAttr.get_f64(kargs['spatial_scale'])
        }
        return self.buildOp(Top.ROIPooling, operands, [output_type], **param)

    def create_frcn_detection_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'class_num': IntegerAttr.get(self.mlir_type['INT64'], kargs['class_num']),
            'keep_topk': IntegerAttr.get(self.mlir_type['INT64'], kargs['keep_topk']),
            'obj_threshold': FloatAttr.get_f64(kargs['obj_threshold']),
            'nms_threshold': FloatAttr.get_f64(kargs['nms_threshold']),
        }
        return self.buildOp(Top.FrcnDetection, operands, [output_type], **param)

    def create_floor_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.FloorOp, operands, [output_type], **param)

    def create_retinaface_detection_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'keep_topk': IntegerAttr.get(self.mlir_type['INT64'], kargs['keep_topk']),
            'confidence_threshold': FloatAttr.get_f64(kargs['confidence_threshold']),
            'nms_threshold': FloatAttr.get_f64(kargs['nms_threshold']),
        }
        return self.buildOp(Top.RetinaFaceDetection, operands, [output_type], **param)

    def create_qlinear_op(self, operands, output_shape, axis=1, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape, self.mlir_type['SINT8'])
        param = {
            'name': kargs['name'],
            'y_scale': self.ArrayAttr(kargs['y_scale'], 'F64'),
            'y_zero_point': self.ArrayAttr(kargs['y_zero_point'], 'INT32'),
            'axis': IntegerAttr.get(self.mlir_type['INT64'], axis)
        }
        return self.buildOp(Top.QuantizeLinearOp, operands, [output_type], **param)

    def create_deqlinear_op(self, operands, output_shape, axis=1, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            'x_scale': self.ArrayAttr(kargs['x_scale'], 'F64'),
            'x_zero_point': self.ArrayAttr(kargs['x_zero_point'], 'INT32'),
            'axis': IntegerAttr.get(self.mlir_type['INT64'], axis)
        }
        return self.buildOp(Top.DequantizeLinearOp, operands, [output_type], **param)

    def create_layer_norm_op(self, operands, output_shapes, **kargs):
        # get_value_type
        out_types = list()
        for s in output_shapes:
            if s is not None and len(s) == 0:
                out_types.append(NoneType.get())
            else:
                out_types.append(self.get_tensor_type(s))
        param = {
            'name': kargs['name'],
            'normalized_shape': self.ArrayAttr(kargs['normalized_shape']),
            'axis': IntegerAttr.get(self.mlir_type['INT32'], kargs['axis']),
            'eps': FloatAttr.get_f64(kargs['eps'])
        }
        return self.buildOp(Top.LayerNormOp, operands, out_types, **param)

    def create_pixel_norm_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name'], 'eps': FloatAttr.get_f64(kargs['eps'])}
        return self.buildOp(Top.PixelNormOp, operands, [output_type], **param)

    def create_instance_norm_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.F32Type)
        param = {'name': kargs['name'], 'eps': FloatAttr.get_f64(kargs['eps'])}
        return self.buildOp(Top.InstanceNormOp, operands, [output_type], **param)

    def create_group_norm_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.F32Type)
        param = {
            'name': kargs['name'],
            "num_groups": IntegerAttr.get(self.mlir_type['INT64'], kargs['num_groups']),
            'eps': FloatAttr.get_f64(kargs['eps'])
        }
        return self.buildOp(Top.GroupNormOp, operands, [output_type], **param)

    def create_scatter_elements_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.mlir_type['F32'])
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis'])
        }
        if kargs['reduction'] != None:
            param['reduction'] = kargs['reduction']
        return self.buildOp(Top.ScatterElementsOp, operands, [output_type], **param)

    def create_scatternd_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        if kargs['reduction'] != None:
            param['reduction'] = kargs['reduction']
        return self.buildOp(Top.ScatterNDOp, operands, [output_type], **param)

    def create_roi_align_op(self, operands, output_shape, **kargs):
        output_type = self.get_tensor_type(output_shape)
        param = {
            'name': kargs['name'],
            "mode": StringAttr.get(kargs['mode']),
            "output_height": IntegerAttr.get(self.mlir_type['INT64'], kargs["output_height"]),
            "output_width": IntegerAttr.get(self.mlir_type['INT64'], kargs["output_width"]),
            "sampling_ratio": IntegerAttr.get(self.mlir_type['INT64'], kargs["sampling_ratio"]),
            "spatial_scale": FloatAttr.get_f64(kargs["spatial_scale"]),
            "align_corners": BoolAttr.get(kargs["align_corners"])
        }
        return self.buildOp(Top.RoiAlignOp, operands, [output_type], **param)

    def create_silu_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = self.get_tensor_type(output_shape)
        param = {'name': kargs['name']}
        return self.buildOp(Top.SiLUOp, operands, [output_type], **param)

    def create_nonzero_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.F32Type)
        param = {'name': kargs['name'], 'order': StringAttr.get(kargs["order"])}
        return self.buildOp(Top.NonZeroOp, operands, [output_type], **param)

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
            module attributes {{module.name = \"{name}\", module.weight_file= \"{weight_file}\", module.state=\"{state}\", module.chip=\"{chip}\"}} {{
                func.func @main({args}) -> {output} {{
                    %0 = \"top.None\"() : () -> none loc(unknown)
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                   weight_file=self.weight_file,
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
