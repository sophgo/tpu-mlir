from mlir.ir import *
import mlir.dialects.top as top
import numpy as np
import torch
import operator

class State:
    TOP_F32 = 'TOP_F32'
    TOP_F16 = 'TOP_F16'
    TOP_BF16 = 'TOP_F32'

class FxMIIRImportor(object):
    def __init__(self,model_name,weight_file,args):
        self.model_name = model_name
        self.weight_file = weight_file
        self.args = args
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.weights_data = dict()
        self.load_weight = dict()
        # self.F32Type = F32Type.get()
        self.mlir_type = {
              "F32": F32Type.get(),
              "F16": F16Type.get(),
              "BF16": BF16Type.get()
          }

    def __del__(self):
        import traceback
        try:
            self.loc.__exit__(None, None, None)
        except Exception as e:
            tb_info = traceback.format_exc()
            print(tb_info)
            print("failed exit loc!")
            pass
        try:
            self.ctx.__exit__(None, None, None)
        except Exception as e:
            tb_info = traceback.format_exc()
            print(tb_info)
            print("failed exit ctx!")
            pass

    def convert_scalar_param(self, scalar):
        if not isinstance(scalar, (int, float)):
            return np.atleast_2d(scalar).astype(np.float32)
        else:
            return np.atleast_1d(scalar).astype(np.float32)

    # shape: [] => [* x f32]; None => NoneType; [None, None] => [NoneType, NoneType]
    # type: None => f32; or type
    def get_tensor_type(self, output_shapes, type=None):
        if type is None:
            if self.args.fp == "fp16":
                type = F16Type.get()
            else:
                type = F32Type.get()
        if output_shapes == []:
            return UnrankedTensorType.get(type)
        if output_shapes is None:
            return NoneType.get()
        if isinstance(output_shapes, tuple):
            output_shapes = list(output_shapes)
        assert (isinstance(output_shapes, list))
        assert (len(output_shapes) > 0)
        if not isinstance(output_shapes[0], list) and output_shapes[0] is not None:
            return RankedTensorType.get(tuple(output_shapes), type) #why?
        # multi output
        out_types = []
        if isinstance(type, list):
            for s,t in zip(output_shapes, type):
                if s == []:
                    out_types.append(UnrankedTensorType.get(t))
                elif s is None:
                    out_types.append(NoneType.get())
                else:
                    out_types.append(RankedTensorType.get(tuple(s), t))
        else:
            for s in output_shapes:
                if s == []:
                    out_types.append(UnrankedTensorType.get(type))
                elif s is None:
                    out_types.append(NoneType.get())
                else:
                    out_types.append(RankedTensorType.get(tuple(s), type))
        return out_types


    def get_dtype(self, type1):
        # if self.args.fp == "fp16":
        #     return F16Type.get()  #??? todo
        # else:
        #     return F32Type.get()
        # dtype = None
        if type1 == torch.float16:
            dtype = self.mlir_type['F16']
        else:
            dtype = self.mlir_type['F32']
        # elif type1 == torch.float32:
        #     dtype = self.mlir_type['F32']
        # elif type1 == torch.int64:
        #     dtype = self.mlir_type['INT64']
        # elif type1 == torch.int32:
        #     dtype = self.mlir_type['INT32']
        # elif type1 == torch.bool:
        #     dtype = self.mlir_type['BOOL']
        return dtype

    def get_output_dtypes(self, node):
        dtypes = []
        if 'val' in node.meta:
            if isinstance(node.meta['val'], (tuple,list)):
                dtypes = [i.dtype if i is not None else None for i in node.meta['val'] ]
            else:
                dtypes.append(node.meta['val'].dtype)
        else:
            if self.args.fp == "fp16":
                dtypes.append(torch.float16)
            else:
                dtypes.append(torch.float32)
        dtypes = [self.get_dtype(i) if i is not None else None for i in dtypes]
        # if len(dtypes) == 1:
        #     dtypes = dtypes[0]
        return dtypes

    def nodeIsBelongToChanStyle(self, node):
        ChanStyleOp = [torch.ops.aten._native_batch_norm_legit_functional]
        if node.target in ChanStyleOp:
            return True
        elif node.target == operator.getitem and node.args[0].target in ChanStyleOp:
            return True
        else:
            for user in node.users:
                if user.target in ChanStyleOp:
                    return True
                elif hasattr(node.meta['val'], 'size') and len(node.meta['val'].size())==1 and 'convolution' in user.name: #conv bias
                    return True
                # user.target in [torch.ops.aten.convolution]
        return False

    def get_output_shapes(self, node, exclude_num = 0):
        shapes = []
        if isinstance(node.meta['val'], (tuple,list)):
            shapes = [list(i.size()) if i is not None else None for i in node.meta['val']]
        else:
            shapes.append(list(node.meta['val'].size()))

        if self.nodeIsBelongToChanStyle(node):
            shapes = [[1,i[0],1,1] if len(i) == 1 else i for i in shapes]
        # shapes = [[1] if i is not None and i == [] else i for i in shapes]
        for _ in range(exclude_num):
            shapes.pop()
        return shapes

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.ctx)
        elif isinstance(names, torch.fx.Node):
            return Location.fused([Location.name(names.name)], context=self.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def create_input_op(self, node, func_arg, operands, mix_dtype_input=False):
        output_dtypes = self.get_output_dtypes(node)
        output_shapes = self.get_output_shapes(node)
        init_args = {}
        init_args["loc"] = self.get_loc(node)
        init_args["ip"] = self.insert_point
        init_args["input"] = func_arg
        init_args["output"] = RankedTensorType.get(output_shapes[0], output_dtypes[0]) #F16?
        input_op = top.InputOp(**init_args).output
        if 'tensor_meta' in node.meta \
            and node.meta['tensor_meta'].requires_grad \
            and node.meta['tensor_meta'].dtype != torch.float32:
#             input_op = top.WeightReorderOp(*self.get_tensor_type(output_shapes, F32Type.get()),
#                                         input_op,
#                                         loc=self.get_loc(node.name + '_reorder'),
#                                         ip=self.insert_point).output
            if node.meta['tensor_meta'].dtype == torch.float16:
                new_op2 = top.WeightReorderOp(*self.get_tensor_type(output_shapes, F16Type.get()),
                                            input_op,
                                            loc=self.get_loc(f'{node}_WeightReorder'),
                                            ip=self.insert_point).output
            else:
                new_op2 = top.WeightReorderOp(*self.get_tensor_type(output_shapes, F32Type.get()),
                                            input_op,
                                            loc=self.get_loc(f'{node}_WeightReorder'),
                                            ip=self.insert_point).output
            operands[node] = new_op2
            return
        if 'tensor_meta' in node.meta \
            and node.meta['tensor_meta'].requires_grad \
            and node.meta['tensor_meta'].dtype == torch.float32 \
            and mix_dtype_input == True \
            and output_shapes[0][0] != 1:

            new_op2 = top.DtypeCastOp(*self.get_tensor_type(output_shapes, F16Type.get()),
                                    input_op,
                                    loc=self.get_loc(f'{node}_DtypeCast'),
                                    ip=self.insert_point).output
            # new_op3 = top.WeightReorderOp(*self.get_tensor_type(output_shapes, F16Type.get()),
            #                             new_op2,
            #                             loc=self.get_loc(f'{node}_WeightReorder'),
            #                             ip=self.insert_point).output
            # operands[node] = new_op3
            operands[node] = new_op2
            return
        operands[node] = input_op

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.weights_data:
            tensor_npz[name] = self.weights_data[name]
        np.savez(weight_file, **tensor_npz)

    def create_weight_op(self, name, arg, data_type = 'F32'):
        arg_t = self.convert_scalar_param(arg)
        arg_shape = list(arg_t.shape)
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != arg_shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        tensor_type = RankedTensorType.get(arg_shape, self.mlir_type[data_type])
        op = Operation.create("top.Weight",
                              results=[tensor_type],
                              loc=Location.fused([Location.name(name)]))
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, arg_shape, data_type)
        self.weights_data[name] = arg_t
        return result

    def create_constant_weight_op(self,name,shape,val,data_type = 'F32'):
        name = f'{name}_c'
        tensor_type = RankedTensorType.get(list(shape), self.mlir_type[data_type])
        op = Operation.create("top.Weight",
                            results=[tensor_type],
                            loc=Location.fused([Location.name(name)]))
        shape = tuple(shape)
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, shape, data_type)
        self.weights_data[name] = np.full(shape,val,dtype = np.float32)
        return result

    def print_module(self):
        mlir_format = self.mlir_module.operation.get_asm(enable_debug_info=True)
        return mlir_format

    def createMlirModuleAndInput(self,input_nodes,in_args_txt, output_args_txt,operands):
        num_output = len(output_args_txt.split(','))
        result_var_name = "%1"
        result_types = output_args_txt
        if num_output > 1:
            result_var_name = ",".join([f"%1#{var_id}" for var_id in range(num_output)])
            result_types = output_args_txt[1:-1]
        main_func = """
            module attributes {{sym_name = \"{name}\", module.weight_file= \"{weight_file}\", module.platform=\"TORCH\", module.state=\"{state}\", module.chip=\"{chip}\", module.train={train}}} {{
                func.func @main({args}) -> {output} {{
                    %0 = \"top.None\"() : () -> none loc(unknown)
                    %1:{last_output_num} = \"Placeholder.Op\"() : () -> {output}
                    return {result_var} : {result_types}
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                    weight_file=self.weight_file,
                    state=State.TOP_F16 if self.args.fp == "fp16" else State.TOP_F32,
                    chip="ALL",
                    train='true',
                    args=in_args_txt,
                    last_output_num=num_output,
                    result_var=result_var_name,
                    result_types=result_types,
                    output=output_args_txt)
        print(f'main_func:\n{main_func}\nmain_func end')
        self.mlir_module = Module.parse(main_func, self.ctx)
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.none_op = self.entry_block.operations[0].operation.results[0]
        self.entry_block.operations[2].operation.erase()
        self.entry_block.operations[1].operation.erase()
        mix_dtype_input = False
        for _, node in input_nodes:
            if ('tensor_meta' in node.meta) and (node.meta['tensor_meta'].dtype == torch.float16):
                mix_dtype_input = True
                break
        for node, arg in zip(input_nodes, self.entry_block.arguments):
            self.create_input_op(node[1], arg, operands, mix_dtype_input)

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.weights_data:
            tensor_npz[name] = self.weights_data[name]
        np.savez(weight_file, **tensor_npz)
