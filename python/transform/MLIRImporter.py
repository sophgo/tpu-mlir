# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from mlir.ir import *
import mlir.dialects.top as top


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


class MLIRImporter(object):

    def __init__(self,
                 input_shapes: list,
                 output_shapes: list,
                 model_name: str,
                 platform: str = Platform.ONNX,
                 input_types: list = [],
                 output_types: list = [],
                 state: str = State.TOP_F32,
                 do_declare: bool = True,
                 run_mode: str = "STATIC",
                 no_save: bool = False):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert (len(model_name) > 0)
        self.no_save = no_save
        self.model_name = model_name
        self.state = state
        self.chip = "ALL"
        self.run_mode = run_mode
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
            "INT8": IntegerType.get_signed(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signed(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signed(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),  #special
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get(),
            "DICT": DictAttr.get(),
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
        if data_type == 'DICT':
            # the data in list has been transformed to DictAttr
            return ArrayAttr.get(data)
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

    def create_input_op(self, loc, index, kargs: dict = {}):
        assert (index < len(self.func_args))
        init_args = {}
        channel_axis = 1
        shape = self.input_shapes[index]
        if 'channel_format' in kargs:
            if kargs['channel_format'] == 'nhwc':
                channel_axis = -1
            if (len(shape) == 4 and shape[channel_axis] <= 4) or len(shape) == 3:
                init_args = {
                    k: StringAttr.get(v) if isinstance(v, str) else v
                    for k, v in kargs.items()
                }
        if 'preprocess_list' in kargs and kargs['preprocess_list'] is not None:
            if index + 1 in kargs['preprocess_list'] :
                    init_args["do_preprocess"] = 1
            if 'preprocess_list' in init_args:
                del init_args["preprocess_list"]
        if 'shape_tensor' in kargs:
            init_args["shape_tensor"] = kargs['shape_tensor']
        init_args["loc"] = loc
        init_args["ip"] = self.insert_point
        init_args["input"] = self.func_args[index]
        init_args["output"] = self.input_types[
            index] if self.platform in [Platform.TFLITE, Platform.TPULANG] else self.input_op_types[index]
        input_op = top.InputOp(**init_args)
        return input_op.output

    def create_weight_op(self, name, output_shape, data_type="F32"):
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != output_shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        attrs = dict()
        if self.no_save:
            attrs["inline_bytes"] = StringAttr.get(tensor.buffer.tobytes())
        tensor_type = RankedTensorType.get(output_shape, self.mlir_type[data_type])
        op = Operation.create("top.Weight",
                              results=[tensor_type],
                              loc=Location.fused([Location.name(name)]),
                              attributes=attrs)
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, output_shape, data_type)
        return result

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def create_yield_op(self, Operands):
        yield_op = Operation.create("top.Yield", operands=Operands, results=[])
        self.insert_point.insert(yield_op)
        return yield_op

    def create_if_op(self, operands, output_shape, **kargs):
        region = IntegerAttr.get(self.mlir_type['INT64'], kargs["region"]).value
        op = Operation.create("top.If",
                              results=[self.get_tensor_type(output_shape)],
                              operands=operands,
                              loc=Location.fused([Location.name(x) for x in kargs['name']]),
                              attributes=dict(),
                              regions=region)
        self.insert_point.insert(op)
        return op.result

    def create_loop_op(self, operands, output_shape, **kargs):
        region = IntegerAttr.get(self.mlir_type['INT64'], kargs["region"]).value
        op = Operation.create("top.Loop",
                              results=self.get_tensor_type(output_shape),
                              operands=operands,
                              loc=Location.fused([Location.name(x) for x in kargs['name']]),
                              attributes=dict(),
                              regions=region)
        self.insert_point.insert(op)
        return op.results

    def create_subgraph_input_op(self, name, type, val, **kargs):
        param = {}
        op = Operation.create("top.Input",
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
        self.input_op_types = list()
        self.output_types = list()
        for _shape, _type in zip(self.input_shapes, input_types):
            self.input_op_types.append(RankedTensorType.get(_shape, self.F32Type))
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
        result_types = output_txt
        result_var_name = "%1"
        if self.num_output > 1:
            output_txt = "({})".format(output_txt)
            result_types = output_txt[1:-1]
            result_var_name = ",".join([f"%1#{var_id}" for var_id in range(self.num_output)])
        main_func = """
            module @\"{name}\" attributes {{module.weight_file= \"{weight_file}\", module.platform=\"{platform}\", module.state=\"{state}\", module.chip=\"{chip}\", module.top_run_mode=\"{run_mode}\"}} {{
                func.func @main({args}) -> {output} {{
                    %0 = \"top.None\"() : () -> none loc(unknown)
                    %1:{last_output_num} = \"Placeholder.Op\"() : () -> {output}
                    return {result_var} : {result_types}
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                   weight_file="" if self.no_save else self.weight_file,
                   platform=self.platform,
                   state=self.state,
                   chip=self.chip,
                   run_mode=self.run_mode,
                   args=args_txt,
                   output=output_txt,
                   last_output_num=self.num_output,
                   result_var=result_var_name,
                   result_types=result_types)
        self.mlir_module = Module.parse(main_func, self.ctx)
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.none_op = self.entry_block.operations[0].operation.results[0]
        # remove Placeholder.Op and return Op.
        # These operations are placeholders and are only used to generate a legal MLIR code.
        self.entry_block.operations[2].operation.erase()
        self.entry_block.operations[1].operation.erase()

        self.func_args = list()
        for i in self.entry_block.arguments:
            self.func_args.append(i)
