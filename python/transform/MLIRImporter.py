# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from mlir.ir import *


class Top:
    WeightOp = 'top.Weight'
    InputOp = 'top.Input'
    AddOp = 'top.Add'
    AvgPoolOp = 'top.AvgPool'
    BatchNormOp = 'top.BatchNorm'
    ConcatOp = 'top.Concat'
    ConvOp = 'top.Conv'
    Depth2SpaceOp = 'top.Depth2Space'
    MulOp = 'top.Mul'
    MulConstOp = 'top.MulConst'
    MatMulOp = 'top.MatMul'
    MaxPoolOp = 'top.MaxPool'
    MaxPoolWithMaskOp = 'top.MaxPoolWithMask'
    MaxUnpoolOp = 'top.MaxUnpool'
    PermuteOp = 'top.Permute'
    ReshapeOp = 'top.Reshape'
    ReluOp = 'top.Relu'
    SliceOp = 'top.Slice'
    SigmoidOp = 'top.Sigmoid'
    LeakyReluOp = 'top.LeakyRelu'
    UpsampleOp = 'top.Upsample'
    SoftmaxOp = 'top.Softmax'
    LogOp = 'top.Log'
    ExpOp = 'top.Exp'
    PadOp = 'top.Pad'
    DivOp = 'top.Div'
    SqueezeOp = 'top.Squeeze'
    ClipOp = 'top.Clip'
    DeconvOp = 'top.Deconv'
    ScaleOp = 'top.Scale'
    LSTMOp = 'top.LSTM'
    GatherOp = 'top.Gather'
    TileOp = 'top.Tile'
    MaxOp = 'top.Max'
    MinOp = 'top.Min'
    AbsOp = 'top.Abs'
    PReluOp = 'top.PRelu'
    SubOp = 'top.Sub'

class State:
    TOP_F32 = 'TOP_F32'
    TOP_QUANTIZED = 'TOP_QUANTIZED'


def get_weight_file(model_name: str, state: str, chip: str):
    name = "{}_{}_{}_weight.npz".format(model_name, state, chip)
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
        self.mlir_type = {
            "INT8": IntegerType.get_signless(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signless(16),
            "INT32": IntegerType.get_signless(32),
            "INT64": IntegerType.get_signless(64),
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

    def get_value_type(self, value):
        _type = str(value.type)
        _type = _type.split('<')[-1].split('x')[-1].split('>')[0]
        if _type == "f32":
            return self.mlir_type['F32']
        elif _type == "i8":
            return self.mlir_type['INT8']
        elif _type == "ui8":
            return self.mlir_type['UINT8']
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
        if num_output == 1:
            loc = Location.fused([Location.name(kargs['name'])])
        else:
            assert (isinstance(kargs["name"], list))
            assert (num_output == len(kargs["name"]))
            loc = Location.fused([Location.name(n) for n in kargs["name"]])
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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        if "coeff" in kargs:
            param['coeff'] = self.ArrayAttr(kargs['coeff'], 'F64')
        return self.buildOp(Top.AddOp, operands, [output_type], **param)

    def create_mul_op(self, operands, output_shape, **kargs):
        if len(operands) < 2:
            raise RuntimeError("input operand must great than 2")
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.MulOp, operands, [output_type], **param)

    def create_mul_const_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'epsilon': FloatAttr.get_f64(kargs['epsilon']),
        }
        return self.buildOp(Top.BatchNormOp, operands, [output_type], **param)

    def create_concat_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        p = {
            'name': kargs['name'],
            "block_h": IntegerAttr.get(self.mlir_type['INT64'], kargs['block_h']),
            "block_w": IntegerAttr.get(self.mlir_type['INT64'], kargs['block_w']),
            "is_CRD": BoolAttr.get(kargs['is_CRD']),
            "is_inversed": BoolAttr.get(kargs['is_inversed']),
        }
        return self.buildOp(Top.Depth2SpaceOp, operands, [output_type], **p)

    def create_leaky_relu_op(self, operands, output_shape, **kargs):
        """
            operands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'order': self.ArrayAttr(kargs['order']),
        }
        return self.buildOp(Top.PermuteOp, operands, [output_type], **param)

    def create_matmul_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

        param = {
            'name': kargs['name'],
            'do_relu': BoolAttr.get(kargs['do_relu']),
        }
        return self.buildOp(Top.MatMulOp, operands, [output_type], **param)

    def create_relu_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
        }
        if 'relu_limit' in kargs:
            param['relu_limit'] = FloatAttr.get_f64(kargs['relu_limit'])
        return self.buildOp(Top.ReluOp, operands, [output_type], **param)

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def create_reshape_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        return self.buildOp(Top.ReshapeOp, operands, [output_type], name=kargs['name'])

    def create_slice_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'offset': self.ArrayAttr(kargs['offset']),
            'steps': self.ArrayAttr(kargs['steps']),
        }
        return self.buildOp(Top.SliceOp, operands, [output_type], **param)

    def create_sigmoid_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'scale': FloatAttr.get_f64(kargs['scale']),
            'bias': FloatAttr.get_f64(kargs['bias']),
        }
        return self.buildOp(Top.SigmoidOp, operands, [output_type], **param)

    def create_upsample_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'scale_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_h']),
            'scale_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_w']),
        }
        return self.buildOp(Top.UpsampleOp, operands, [output_type], **param)

    def create_maxunpool_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'scale_h': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_h']),
            'scale_w': IntegerAttr.get(self.mlir_type['INT64'], kargs['scale_w']),
        }
        return self.buildOp(Top.MaxUnpoolOp, operands, [output_type], **param)

    def create_softmax_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
        }
        return self.buildOp(Top.SoftmaxOp, operands, [output_type], **param)

    def create_log_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.LogOp, operands, [output_type], **param)

    def create_exp_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
        }
        return self.buildOp(Top.ExpOp, operands, [output_type], **param)

    def create_pad_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.DivOp, operands, [output_type], **param)

    def create_squeeze_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'axes': self.ArrayAttr(kargs['axes']),
        }
        return self.buildOp(Top.SqueezeOp, operands, [output_type], **param)

    def create_clip_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))

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
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        return self.buildOp(Top.ScaleOp, operands, [output_type], **param)

    def create_lstm_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'have_bias': BoolAttr.get(kargs['have_bias']),
            'bidirectional': BoolAttr.get(kargs['bidirectional']),
            'batch_first': BoolAttr.get(kargs['batch_first']),
        }
        return self.buildOp(Top.LSTMOp, operands, [output_type], **param)

    def create_gather_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
        }
        return self.buildOp(Top.GatherOp, operands, [output_type], **param)

    def create_tile_op(self, operands, output_shape, **kargs):
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {
            'name': kargs['name'],
            'axis': IntegerAttr.get(self.mlir_type['INT64'], kargs['axis']),
            'tile': IntegerAttr.get(self.mlir_type['INT64'], kargs['tile']),
        }
        return self.buildOp(Top.TileOp, operands, [output_type], **param)

    def create_max_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name']}
        return self.buildOp(Top.MaxOp, operands, [output_type], **param)

    def create_min_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name']}
        return self.buildOp(Top.MinOp, operands, [output_type], **param)

    def create_abs_op(self, operands, output_shape, **kargs):
        # get_value_type
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name']}
        return self.buildOp(Top.AbsOp, operands, [output_type], **param)

    def create_prelu_op(self, operands, output_shape, **kargs):
        if len(operands) != 2:
            raise RuntimeError("input operand must equal 2")
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name']}
        return self.buildOp(Top.PReluOp, operands, [output_type], **param)

    def create_sub_op(self, operands, output_shape, **kargs):
        if len(operands) < 2:
            raise RuntimeError("input operand must great than 2")
        output_type = RankedTensorType.get(tuple(output_shape), self.get_value_type(operands[0]))
        param = {'name': kargs['name'], 'do_relu': BoolAttr.get(False)}
        if "coeff" in kargs:
            param['coeff'] = self.ArrayAttr(kargs['coeff'], 'F64')
        return self.buildOp(Top.SubOp, operands, [output_type], **param)

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
            if isinstance(_type, str):
                self.output_types.append(RankedTensorType.get(_shape, self.mlir_type[_type]))
            else:
                self.output_types.append(RankedTensorType.get(_shape, _type))
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
