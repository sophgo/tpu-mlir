# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# ONNX Node define:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md

from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter

import numpy as np
import paddle
from numbers import Number
import os
import copy
import mlir.dialects.top as top
from mlir.ir import *
from utils.pad_setting import set_auto_pad
from utils.auto_remove import file_mark, file_clean




class BaseNode():

    def __init__(self, info):
        #创建一些基本的属性
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])                                                                        #这里面有很多个attr，所以是用一个字典来对应
        self.inputs = list(info["inputs"])                                                                      #输入主要对应的是具体的节点，所以是一个list
        self.outputs = list(info["outputs"])

class PaddleNode(BaseNode):

    def __init__(self, node):
        info = dict()
        unuse_attr = ['op_device','op_role','op_role_var','op_namescope']
        info["name"] = node["output"][0]
        info["op_type"] = node["op_type"]
        info["attrs"] = [(attr, node["attrs"][attr])
                         for attr in node["attrs"] if attr not in unuse_attr]
        info["inputs"] = node["input"]
        info["outputs"] = node["output"]
        super().__init__(info)
        self.node_proto = node

class PaddleConverter(BaseConverter):

    #定义初始化参数以及所需要传进去的参数：
    def __init__(self,
                 model_name:str,
                 paddle_file,
                 input_shapes:list,
                 output_names:list,
                 preprocess_args:dict = {}
                 ):

        super().__init__()

        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir  = None
        self.node_name_mapping = {}
        self.load_paddle_model(paddle_file,input_shapes,output_names)
        self.init_MLIRImporter()
        self.preprocess_args = {}
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.preprocess_args = preprocess_args
        self.converted_nodes = list()
        self.paddleop_factory = {
            #注意：请按照字母顺序添加算子！！！
            "cast" : lambda node:self.convert_cast_op(node),
            "concat" : lambda node: self.convert_concat_op(node),
            "conv2d" : lambda node: self.convert_conv_op(node),
            "elementwise_add" : lambda node: self.convert_add_op(node),
            "elementwise_mul" : lambda node: self.convert_mul_op(node),
            "elementwise_pow" : lambda node: self.convert_pow_op(node),
            "elementwise_sub" : lambda node: self.convert_sub_op(node),
            "equal" : lambda node: self.convert_cmp_op(node),
            "fill_constant" : lambda node: self.convert_constantFill_op(node),
            "gather" :  lambda node: self.convert_gather_op(node),
            "matmul_v2" : lambda node: self.convert_matmul_op(node),
            "nearest_interp_v2" : lambda node: self.convert_upsample_op(node),
            "pool2d" : lambda node:self.convert_maxpool_op(node),
            "range" : lambda node: self.convert_range_op(node),
            "reshape2" : lambda node : self.convert_reshape_op(node),
            "shape" : lambda node : self.convert_shape_op(node),
            "sigmoid" : lambda node: self.convert_sigmoid_op(node),
            "strided_slice" : lambda node: self.convert_slice_op(node),
            "transpose2" : lambda node: self.convert_transpose_op(node),
            "unsqueeze2" : lambda node: self.convert_unsqueeze_op(node),
            "where" : lambda node: self.convert_where_op(node),


        }
    def get_outputs(self,output_names):
        return [x for x in self.fetch_targets if x.op.idx in output_names]

    def get_inputs(self,all_valid_inputs):
        return [x for x in all_valid_inputs]
    #获取输入的类型
    def get_input_types(self):
        input_types = []
        for input in self.get_inputs(self.all_valid_inputs):
          if self.block_op.var(input).dtype in [
              paddle.int32,paddle.int64
          ]:
              input_types.append('INT32')
          else:
              input_types.append('F32')
        return input_types
      #这个是干嘛用的？
    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    #获取输出的类型
    def get_output_types(self,output_names):
        output_types = []
        for output in self.get_outputs(output_names):
            if output.dtype in [
            paddle.int32,paddle.int64
            ]:
                output_types.append('INT32')
            else:
                output_types.append('F32')
        return output_types


    def select_unuse(self, names):
        # print("names:",names)
        for name in names:
            if name in self.all_weights:
                self.all_weights.pop(name)
            if name in self.all_values:
                self.all_values.pop(name)
            if name in self.all_inputs:
                self.all_inputs.pop(name)
            if name in self.all_nodes:
                cur_node = self.all_nodes.pop(name)
                for o in cur_node["output"]:
                    if o in self.all_nodes:
                        self.all_nodes.pop(o)
                self.select_unuse(cur_node["input"])

    def select_output(self,output_names:list):
        self.all_outputs = []
        self.all_inputs = {}

        #获取模型的输入
        for x in self.feed_target_names:
            self.all_inputs[x] = self.block_op.var(x)               #输入的部分是要得到各种参数，而不仅仅是name
        #self.all_valid_inputs = self.all_inputs
        #获取模型的输出
        for x in self.fetch_targets:
            #先判别x是否在预定的输出参数中：feed_target是一个list,而里面每个参数是一个paddle.fluid.framework.Variable类
            #预定的输出参数是一个索引值，所以我想的是根据op的索引值来得到对应的值：根据打印出来的结果是：对应的索引值是663、311、457、587，而663的是不用到的
            #同时这里注意，只需要获取对应输出的name，而不需要其他信息

            if str(x.op.idx) in output_names:
                # print("################")

                self.all_outputs.append(x.op.output_arg_names[0])    #得到已经预设的输出参数名字
                # print(self.all_outputs)
                output_names.remove(str(x.op.idx))                                               #将已经得到名字的预定输出给移除
                if len(output_names) == 0:
                    break
        #获取模型的value》》》对应于onnx的value_info
        self.all_values = {}
        for var_name in self.block_op.vars:
            self.all_values[var_name] = self.block_op.var(var_name)
        #获取模型节点的输入和输出信息
        self.all_nodes = {}
        for op in self.block_op.ops:
          if 'Out' in op.output_names:
              output_vars = op.output('Out')
          elif 'Output' in op.output_names:
              output_vars = op.output('Output')
          self.all_nodes[output_vars[0]] = {"input" : op.input_arg_names,
                                      "output":output_vars,
                                      "op_name":output_vars,
                                      "op_type":op.type,
                                      "attr":op.all_attrs()}
        if len(output_names) != 0:
            raise RuntimeError("Error, can't find {} in model".format(output_names))

        #获取模型的权重参数
        self.all_weights = {}
        for w in range(0,len(self.model.parameters())):
            self.all_weights[self.model.parameters()[w].name ] = self.model.parameters()[w].value()

        # remove unused node
        self.select_unuse(self.all_outputs)
        #下面是写关于将有效的主干网络对应的节点保留起来的值
        self.all_valid_values = {}
        self.all_valid_nodes = {}
        self.all_valid_weights = {}
        self.all_valid_inputs = {}
        for x in self.feed_target_names:
            if x not in self.all_inputs:
                self.all_valid_inputs[x] = self.block_op.var(x)
        #将主干的values值进行保存
        for var_name in self.block_op.vars:
            if var_name not in self.all_values:
                # print("keep_values!:",var_name)
                self.all_valid_values[var_name] = self.block_op.var(var_name)
        #将主干的weight进行保存
        for w in range(0,len(self.model.parameters())):
            if self.model.parameters()[w].name not in self.all_weights:
                self.all_valid_weights[self.model.parameters()[w].name ] = self.model.parameters()[w].value()

        #将主干的node进行保存
        for op in self.block_op.ops:
            if 'Out' in op.output_names:
                output_vars = op.output('Out')
            elif 'Output' in op.output_names:
                output_vars = op.output('Output')
            if output_vars[0] not in self.all_nodes:
                all_attrs = op.all_attrs()
                if op.type == 'conv2d':
                    all_attrs.update({"kernel_shape":self.all_valid_weights[op.input_arg_names[0]].shape[2:]})
                self.all_valid_nodes[output_vars[0]] = {"input" : op.input_arg_names,
                                            "output":output_vars,
                                            "op_name":output_vars,
                                            "op_type":op.type,
                                            "attrs":all_attrs}


    def load_paddle_model(self,paddle_file,input_shapes,output_names):

        if isinstance(paddle_file,str):
            print("start loading paddle_file!!!\n")
            paddle.enable_static()
            exe = paddle.static.Executor(paddle.CPUPlace())
            #下面是得到模型文件里面的输入、输出、program
            self.feed_target_names = list()
            self.fetch_targets = list()
            [self.inference_program,self.feed_target_names,self.fetch_targets] = (paddle.static.load_inference_model(paddle_file, exe))
            paddle.disable_static()
            print(paddle_file,":",type(paddle_file))

            self.model = paddle.jit.load(paddle_file)
            self.block_op = self.inference_program.block(0)
            print("loading has finished!!!\n")
            print("----------------------------------")
        else:
            print("load_failed!!!",paddle_file)
            self.model = paddle_file
        if output_names:
            self.select_output(output_names)
        self.input_names = self.feed_target_names                       #获取输入
        print("input_names:",self.input_names)
        self.num_input = len(self.input_names)                          #获取输入的长度
        self.input_shapes = [list(self.block_op.var(self.feed_target_names[0]).shape)]    #得到输入的shape,这里的输出的部分是【【-1，3，-1，-1】】，不知道是否需要调整？
        #获取输入的类型
        self.input_types = self.get_input_types()

        #获取输出的类型
        self.output_types = self.get_output_types(output_names)
        print("After assigning input_shape:")
        #print(self.model.parameters())
        for tensor in self.model.parameters():
        #    if tensor.name in self.all_valid_weights:
            if tensor.name == 'x2paddle_0':
                name = 'x2paddle_773'
            elif tensor.name == 'x2paddle_1':
                name = 'x2paddle_774'
            elif tensor.name == 'x2paddle_2':
                name = 'x2paddle_775'
            elif tensor.name == 'x2paddle_3':
                name = 'x2paddle_777'
            elif tensor.name == 'x2paddle_4':
                name = 'x2paddle_780'
            elif tensor.name == 'x2paddle_5':
                name = 'x2paddle_784'
            elif tensor.name == 'x2paddle_6':
                name = 'x2paddle_793'
            elif tensor.name == 'x2paddle_7':
                name = 'x2paddle_806'
            else:
                name = tensor.name
            data = tensor.value().numpy().astype(np.float32)
            self.addWeight(name,data)
        self.add_shape_info(input_shapes)
    def add_shape_info(self,input_shapes,flag = True):
        unk_op = []
        nodes_with_shape = []
        constants = []
        for n in self.all_valid_nodes:
            if self.all_valid_nodes[n]['op_type'] == 'fill_constant':
                constants.append(self.all_valid_nodes[n]['output'])
        for val in self.all_valid_values:
            if val == 'feed':
                continue
            shape = self.all_valid_values[val].shape
            if np.any(np.array(shape) <= 0):
                unk_op.append(val)
            elif list(np.array(shape)) == [] and val not in constants:
                unk_op.append(val)
            else:
                self.addShape(val,shape)
            nodes_with_shape.append(val)
        # print("nodes_with_shape:",nodes_with_shape)
        for output in self.all_outputs:
            if not self.isWeight(output):
                for op in self.block_op.ops:
                    if op.output_arg_names[0] == output:
                        self.output_names.append(output)
                        shape = self.block_op.var(output).shape
            var_exists = 'shape' in locals() or 'shape' in globals()
            if var_exists:
                if flag and (np.any(np.array(shape) <= 0) or len(shape) == 0):
                    unk_op.append(output)
                else:
                    self.addShape(output,shape)
                nodes_with_shape.append(output)
        full_nodes = []
        no_list = ["cast","fill_constant","TopK"]            #这个no_list我不知道是用来干啥
                                                      #但是发现paddle里面有一些算子是onnx里面没有的
                                                      #fill_any_like、nearest_interp_v2、pool2d
                                                      #根据参考的onnx，需要特殊处理里面loop算子，
                                                      #但是我发现里面没有这个算子。所以我把那个处理给剔除了
                                                      #同时我不知道那个TOPK算子在这里是要用来干嘛
                                                      #只是由于自己理解觉得他可能有用，所以我就写上去了
        for n in self.all_valid_nodes:
            if self.all_valid_nodes[n]['op_type'] in no_list:
                continue
            for name in self.all_valid_nodes[n]['output']:
                if not name:
                    continue
                full_nodes.append(name)
        unk_op.extend(set(full_nodes) - set(nodes_with_shape))
        unk_op = list(set(unk_op))
        if (flag and unk_op):
            unk_shape = self.get_unk_shape(unk_op,input_shapes)
            for n,s in unk_shape:
                self.addShape(n,list(s))
    #看了一下 发现好像这个函数可以不用写的，这个是用来判别NonMaxSuppression
    #>>>>破案，这个是自己后续需要写的，因为这个是调整动态shape的函数
    def get_unk_shape(self,unk_op,input_shape):
    #这个函数是为了获取shape不知道的输出
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        for i in self.all_valid_inputs:
            inputs = i
        out = exe.run(self.inference_program,
                     feed = {i:np.random.random(input_shape[0]).astype('float32')},
                     fetch_list = unk_op)
        outs_shape = [o.shape for o in out]
        assert (len(outs_shape) == len(unk_op))
        return zip(unk_op,outs_shape)


    #这里应该是shape的分配函数
    def input_shape_assign(self,input_shapes):
        inputs = self.get_inputs(self.all_valid_inputs)
        outputs = self.get_outputs(output_names)
        shape_changed = False
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))
        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input,len(input_shapes))
        for idx,input in enumerate(inputs):
            _dims = self.block_op.var(input).shape
            num_dims = len(_dims)
            if no_shape == False:
                check_shape(num_dims,len(input_shapes[idx]))
            _shape = []
            for _i,_dim in enumerate(_dims):
                if _dim <= 0:
                    if no_shape:
                        assert 0, "Please check --input_shapes formula or check if there is any dynamic dim"
                    else:
                        _dim = input_shapes[idx][_i]
                elif not no_shape and input_shapes[idx][_i] != _dim:
                    _dim = input_shapes[idx][_i]
                    shape_changed = True
                _shape.append(_dim)
            self.addShape(input,_shape)
        for o in outputs:
            _odims = list(o.shape)
            for i in range(len(_odims)):
                if _odims[i] <= 0 or shape_changed:
                    _odims[i] = '?'

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.output_names:
            output_shapes.append(self.getShape(_name))
        #init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, Platform.PADDLE,
                                 self.input_types)
        self.weight_file = self.mlir.weight_file

    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
        #add input op
        for idx, _name in enumerate(self.input_names):
            input_ = self.mlir.create_input_op(self.get_loc(_name),idx,self.preprocess_args)
            self.addOperand(_name,input_)
        #用于提示不支持的算子
        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.converted_nodes.clear()
        for n in self.all_valid_nodes:
            node = PaddleNode(self.all_valid_nodes[n])
            if node.op_type != 'feed':
                self.converted_nodes.append(node)
        #检测支持的算子类型
        unsupported = set()
        for n in self.converted_nodes:
            if n.op_type not in self.paddleop_factory:
                unsupported.add(n.op_type)
            if n.op_type == 'reshape2':
                if len(n.inputs) == 2:
                    n.inputs = n.inputs[1:]
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        for n in self.converted_nodes:
            self.paddleop_factory.get(n.op_type,lambda x: NoneAndRaise(x))(n)
        return_op = list()
        for idx,_name in enumerate(self.output_names):
            op = self.getOperand(_name)             #将名字传进去，得到算子
            return_op.append(op)                   #将返回的算子添加进去

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file,"w" ) as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)

    def convert_add_op(self,paddle_node):
        assert (paddle_node.op_type == "elementwise_add")
        assert (len(paddle_node.inputs) == 2)
        output_shape = self.getShape(paddle_node.name)

        lhs = paddle_node.inputs[0]
        rhs = paddle_node.inputs[1]
        #print("nodes:",paddle_node.name)
        #print("l:",lhs)
        #print("r:",rhs)
        if self.isWeight(lhs) and not self.isWeight(rhs):
            paddle_node.inputs[0],paddle_node.inputs[1] = paddle_node.inputs[1],paddle_node.inputs[0]
            self.convert_add_op(paddle_node)
            return
        name = "{}_{}".format(paddle_node.name,paddle_node.op_type)
        if not self.isWeight(lhs) and self.isWeight(rhs):
            is_scale = False
            if len(output_shape) > 1:
                opd1_num_elem = np.prod(self.getShape(rhs))
                channel = output_shape[1]
                if opd1_num_elem == channel:
                    rhs_shape = self.getShape(rhs)
                    axis = len(output_shape) - len(rhs_shape)
                    if axis > 1:
                        # the second dim (channel) need broadcast
                        is_scale = False
                    elif rhs_shape[1 - axis] == channel:
                        # all dim except channel is 1 need broadcast, use scaleop
                        is_scale = True
                    else:
                        # channel need broadcast, use addop
                        is_scale = False
            lhs_op = self.getOp(lhs)
            if self.isScalar(rhs):
                new_op = top.AddConstOp(self.mlir.get_tensor_type(output_shape),
                                        lhs_op,
                                        self.getScalar(rhs),
                                        do_relu = False,
                                        loc = self.get_loc(name),
                                        ip = self.mlir.insert_point).output
            elif is_scale:
                bias = self.getWeight(rhs)
                weight_data = np.ones_like(bias)
                self.addWeight(name + "_scale",weight_data)
                weight_op = self.getWeightOp(name + '_scale')
                bias_op = self.getWeightOp(rhs)
                new_op = top.ScaleOp(self.mlir.get_tensor_type(output_shape),
                                     lhs_op,
                                     weight_op,
                                     bias_op,
                                     loc = self.get_loc(name),
                                     ip = self.mlir.insert_point).output
            else:
                rhs_op = self.getOp(rhs)
                new_op = top.AddOp(self.mlir.get_tensor_type(output_shape),
                                   [lhs_op,rhs_op],
                                   loc = self.get_loc(name),
                                   ip = self.mlir.insert_point).outout
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = top.AddOp(self.mlir.get_tensor_type(output_shape),
                               [lhs_op,rhs_op],
                               loc = self.get_loc(name),
                               ip = self.mlir.insert_point).output
        self.addOperand(paddle_node.name,new_op)
        #print("nodes:",paddle_node.name)
        #print("self.operands:",self.operands)

    def convert_cast_op(self,paddle_node):
        assert (paddle_node.op_type == "cast")
        if self.isWeight(paddle_node.inputs[0]):
            data = self.getWeight(paddle_node.input[0])
            self.addWeight(paddle_node.name, data)
        else:
            op = self.getOperand(paddle_node.inputs[0])
            self.addOperand(paddle_node.name, op)

    def convert_concat_op(self,paddle_node):
        assert (paddle_node.op_type == "concat")
        output_shape = self.getShape(paddle_node.name)
        num_dims = len(output_shape)
        axis = paddle_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        operands = list()
        weight_data = None
        for x in paddle_node.inputs:
            x_shape = self.getShape(x)
            num_elem = np.prod(x_shape)
            if num_elem == 0:
                print("WARNING:{}'s shape is strange {}".format(x, x_shape))
                continue
            if self.isWeight(x):
                data = self.getWeight(x)
                if weight_data is not None:
                    weight_data = np.concatenate((weight_data, data),axis = axis)
                else:
                    weight_data = data
                continue
            else:
                if weight_data is not None:
                    w_name = x + "_weight"
                    self.addWeight(w_name , weight_data)
                    operands.append(self.getWeightOp(w_name))
                    weight_data = None
                operands.append(self.getOperand(x))
        if len(operands) == 0:
            self.addWeight(paddle_node.name,weight_data)
            return
        if weight_data is not None:
            w_name = paddle_node.name + "_weight"
            self.addWeight(w_name,weight_data)
            operands.append(self.getWeightOp(w_name))
        if len(operands) == 1:
            self.addOperand(paddle_node.name,operands[0])
            return
        new_op = top.ConcatOp(self.mlir.get_tensor_type(output_shape),
                            operands,
                            axis = axis,
                            loc = self.get_loc("{}_{}".format(paddle_node.name,paddle_node.op_type)),
                            ip = self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_constantFill_op(self,paddle_node):
        assert (paddle_node.op_type == "fill_constant")
        paddle_tensor = paddle_node.attrs['value']
        np_tensor = np.array(paddle_tensor).astype('float32')
        self.addWeight(paddle_node.name,np_tensor)


    def convert_cmp_op(self,paddle_node):
        supports = {"equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And"}
        assert (paddle_node.op_type in supports)
        assert (len(paddle_node.inputs) == 2)
        lhs = paddle_node.inputs[0]
        rhs = paddle_node.inputs[1]
        output_shape = self.getShape(paddle_node.name)
        if self.isScalar(lhs):
            rhs_opd = self.getOp(rhs)
            cmp_op = top.CompareConstOp(self.mlir.get_tensor_type(output_shape),
                                        rhs_opd,
                                        mode=StringAttr.get(paddle_node.op_type),
                                        const_val=self.getScalar(lhs),
                                        inversed=True,
                                        loc=self.get_loc("{}_{}".format(
                                            paddle_node.name, paddle_node.op_type)),
                                        ip=self.mlir.insert_point).output
        elif self.isScalar(rhs):
            lhs_opd = self.getOp(lhs)
            cmp_op = top.CompareConstOp(self.mlir.get_tensor_type(output_shape),
                                        lhs_opd,
                                        mode=StringAttr.get(paddle_node.op_type),
                                        const_val=self.getScalar(rhs),
                                        inversed=False,
                                        loc=self.get_loc("{}_{}".format(
                                            paddle_node.name, paddle_node.op_type)),
                                        ip=self.mlir.insert_point).output
        else:
            rhs_opd = self.getOp(rhs)
            lhs_opd = self.getOp(lhs)
            cmp_op = top.CompareOp(self.mlir.get_tensor_type(output_shape),
                                   lhs_opd,
                                   rhs_opd,
                                   mode=StringAttr.get(paddle_node.op_type),
                                   loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                   paddle_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, cmp_op)

    def convert_conv_op(self,paddle_node):
        assert (paddle_node.op_type == "conv2d")
        op = self.getOperand(paddle_node.inputs[1])    #卷积算子对应俩个输入，一个是权重，一个是上一层的mul结果
        kernel_shape = paddle_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = paddle_node.attrs.get("dilations", dim * [1])
        group = paddle_node.attrs.get("group", 1)
        strides = paddle_node.attrs.get("strides", dim * [1])
        auto_pad = paddle_node.attrs.get("auto_pad", None)
        input_shape = self.getShape(paddle_node.inputs[1])
        pads = []
        pads_adjust = []
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        if len(pads) == 0:
            pads = paddle_node.attrs.get("paddings", dim * 2 * [0])
            if len(pads) == 2:
                height = pads[0]
                weight = pads[1]
                pads_adjust = [height,weight,height,weight]

        operands = list()
        operands.append(op)
        filter_op = self.getOp(paddle_node.inputs[0])
        operands.append(filter_op)
        if len(paddle_node.inputs) > 2:
            bias_op = self.getWeightOp(paddle_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        output_shape = self.getShape(paddle_node.name)
        new_op = top.ConvOp(self.mlir.get_tensor_type(output_shape),
                            *operands,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            pads=pads_adjust,
                            group=group,
                            do_relu=False,
                            loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)


    def convert_gather_op(self,paddle_node):
        assert (paddle_node.op_type == "gather")
        in0 = self.getOp(paddle_node.inputs[1])
        in0_shape = self.getShape(paddle_node.inputs[1])
        out_shape = self.getShape(paddle_node.name)
        axis = paddle_node.attrs.get('axis', 0)
        name = "{}_{}".format(paddle_node.name, paddle_node.op_type)
        if self.isScalar(paddle_node.inputs[0]):
            offset = int(self.getScalar(paddle_node.inputs[0]))
            if offset < 0:
                offset = in0_shape[axis] + offset
            slice_offset = [0] * len(in0_shape)
            slice_steps = [1] * len(in0_shape)
            slice_ends = [in0_shape[i] for i in range(len(in0_shape))]
            slice_offset[axis] = offset
            slice_ends[axis] = offset + 1
            slice_shape = list(np.take(np.ones(in0_shape), np.array([offset]), axis=axis).shape)

            slice_op = top.SliceOp(self.mlir.get_tensor_type(slice_shape),
                                   in0,
                                   self.mlir.none_op,
                                   self.mlir.none_op,
                                   self.mlir.none_op,
                                   offset=list(slice_offset),
                                   steps=list(slice_steps),
                                   ends=list(slice_ends),
                                   loc=self.get_loc("{}_Slice".format(paddle_node.name)),
                                   ip=self.mlir.insert_point).output
            new_op = top.ReshapeOp(self.mlir.get_tensor_type(out_shape),
                                   slice_op,
                                   loc=self.get_loc(name),
                                   ip=self.mlir.insert_point).output
            self.addOperand(paddle_node.name, new_op)
            return
        indices = self.getOp(paddle_node.inputs[1])
        new_op = top.GatherOp(self.mlir.get_tensor_type(out_shape),
                              in0,
                              indices,
                              axis=axis,
                              loc=self.get_loc(name),
                              ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_range_op(self,paddle_node):
        assert (paddle_node.op_type == "range")
        print(self.getShape(paddle_node.inputs[0]))
        print(self.getShape(paddle_node.inputs[1]))
        print(self.getShape(paddle_node.inputs[2]))
        start_op = self.getOp(paddle_node.inputs[0])
        limit_op = self.getOp(paddle_node.inputs[1])
        delta_op = self.getOp(paddle_node.inputs[2])
        p = {'name': "{}_{}".format(paddle_node.name, paddle_node.op_type)}
        new_op = self.mlir.create_range_op([start_op, limit_op, delta_op], [], **p)
        self.addOperand(paddle_node.name, new_op)

    def convert_reshape_op(self,paddle_node):
        assert (paddle_node.op_type == "reshape2")
        x = paddle_node.inputs[0]
        operands = list()
        if self.isWeight(x):

            operands.append(self.getWeightOp(x))
            output_shape = self.getShape(paddle_node.name)
            new_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                               *operands,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point).output
        #op = self.getOperand(paddle_node.inputs[0])
        else:
            output_shape = self.getShape(paddle_node.name)
            #op = paddle_node.inputs[0]
            op = self.getOperand(paddle_node.inputs[0])
            new_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_matmul_op(self,paddle_node):
        assert (paddle_node.op_type == "matmul_v2" )
        #处理俩个或多个张量的乘积

        #这俩个参数，我在paddle的attr里面没有看到，所以就注释了
        #去看了onnx的手册，发现这俩个是缩放因子，默认是0
        # alpha = onnx_node.attrs.get('alpha', 1)
        # beta = onnx_node.attrs.get('beta', 1)
        #下面的参数是用来判别是否需要对对应的输入做转置
        trans_x = paddle_node.attrs.get('trans_x',0)
        trans_y = paddle_node.attrs.get('trans_y',0)
        #用于支持更多情况
        assert (trans_x == False)
        operands = list()
        x = paddle_node.inputs[0]
        y = paddle_node.inputs[1]
        if self.isWeight(x):
            if trans_x == True:
                _tensor = self.getWeight(x)
                _tensor = np.ascontiguousarray(np.transpose(_tensor,(1,0)))
                x += '_fix'
                self.addWeight(x,_tensor)
            operands.append(self.getWeightOp(x))
        else:
            operands.append(self.getOperand(x))

        if self.isWeight(y):
            if trans_y == True:
                _tensor = self.getWeight(y)
                _tensor = np.ascontiguousarray(np.transpose(_tensor,(1,0)))
                y += '_fix'
                self.addWeight(x,_tensor)
            operands.append(self.getWeightOp(y))
        else:
            operands.append(self.getOperand(y))
          #如果输入超过俩个
        if len(paddle_node.inputs) > 2:
            z = paddle_node.inputs[2]
            if self.isWeight(z):
                operands.append(self.getWeightOp(z))
            else:
                operands.append(self.getOperand(z))
        else:
            operands.append(self.mlir.none_op)

        print(paddle_node.name)
        output_shape = self.getShape(paddle_node.name)
        new_op = top.MatMulOp(self.mlir.get_tensor_type(output_shape),
                              *operands,
                              do_relu = False,
                              loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                              ip=self.mlir.insert_point).output

        self.addOperand(paddle_node.name,new_op)

        print(paddle_node.name)
        print("*operand:",*operands)
        print("self.operand:",self.operands[paddle_node.name])

    def convert_mul_op(self,paddle_node):
        assert (paddle_node.op_type == "elementwise_mul")
        assert (len(paddle_node.inputs) == 2)
        lhs = paddle_node.inputs[0]
        rhs = paddle_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            paddle_node.inputs[0] , paddle_node.inputs[1] = rhs, lhs
            self.convert_mul_op(paddle_node)
            return
        name = "{}_{}".format(paddle_node.name, paddle_node.op_type)
        if (not self.isWeight(lhs)) and self.isWeight(rhs):
            op0 = self.getOperand(lhs)
            rhs = rhs
            output_shape = self.getShape(paddle_node.name)
            if self.isScalar(rhs):
                mul_const_op = top.MulConstOp(self.mlir.get_tensor_type(output_shape),
                                              op0,
                                              const_val=self.getScalar(rhs),
                                              loc=self.get_loc(name),
                                              ip=self.mlir.insert_point).output
                self.addOperand(paddle_node.name, mul_const_op)
                return
            weight_num_elem = np.prod(self.getShape(rhs))
            channel = output_shape[1]
            if weight_num_elem == channel:
                weight = self.getWeight(rhs)
                offset_data = np.zeros_like(weight)
                self.addWeight(name + '_bias', offset_data)
                weight_op = self.getWeightOp(rhs)
                offset_op = self.getWeightOp(name + '_bias')
                scale_op = top.ScaleOp(self.mlir.get_tensor_type(output_shape),
                                       op0,
                                       weight_op,
                                       offset_op,
                                       loc=self.get_loc(name),
                                       ip=self.mlir.insert_point).output
                self.addOperand(paddle_node.name, scale_op)
                return
            const_op = self.getWeightOp(rhs)
            scale_op = top.MulOp(self.mlir.get_tensor_type(output_shape), [op0, const_op],
                                 loc=self.get_loc(name),
                                 ip=self.mlir.insert_point).output
            self.addOperand(paddle_node.name, scale_op)
            return
        else:
            op0 = self.getOperand(lhs)
            op1 = self.getOperand(rhs)
            output_shape = self.getShape(paddle_node.name)
            mul_op = top.MulOp(self.mlir.get_tensor_type(output_shape), [op0, op1],
                               loc=self.get_loc(name),
                               ip=self.mlir.insert_point).output
            self.addOperand(paddle_node.name, mul_op)
            return

    def convert_maxpool_op(self,paddle_node):
        assert (paddle_node.op_type == "pool2d")
        op = self.getOperand(paddle_node.inputs[0])
        ceil_mode = paddle_node.attrs.get("ceil_mode", False)
        kernel_shape = paddle_node.attrs['ksize']
        #下面这个参数我没找到对应的attrs
        count_include_pad = paddle_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        strides = paddle_node.attrs.get("strides", kernel_shape)
        input_shape = self.getShape(paddle_node.inputs[0])
        auto_pad = paddle_node.attrs.get("adaptive", None)
        pads = []
        pads_adjust = []
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        if len(pads) == 0:
            pads = paddle_node.attrs.get("paddings", dim * 2 * [0])
            if len(pads) == 2:
                height = pads[0]
                weight = pads[1]
                pads_adjust = [height,weight,height,weight]
        if ceil_mode:
            for i in [0, 1]:
                remain_pixel = (input_shape[i + 2] + 2 * pads[i] - kernel_shape[i]) % strides[i]
                if remain_pixel > 0:
                    if ceil_mode:
                        pads[i + 2] += (strides[i] - remain_pixel)
                    else:
                        pads[i + 2] -= remain_pixel
        output_shape = self.getShape(paddle_node.name)
        new_op = top.MaxPoolOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads_adjust,
                               count_include_pad=count_include_pad,
                               do_relu=False,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)


    def convert_pow_op(self,paddle_node):
        assert (paddle_node.op_type == "elementwise_pow")
        assert (len(paddle_node.inputs) == 2)
        base = paddle_node.inputs[0]
        expn = paddle_node.inputs[1]
        if self.isScalar(expn):
            base_op = self.getOp(base)
            expn_const = self.getScalar(expn)
            output_shape = self.getShape(paddle_node.name)
            if expn_const == 1.0:
                self.addOperand(paddle_node.name, base_op)
                return
            if expn_const == 2.0:
                mul_op = top.MulOp(self.mlir.get_tensor_type(output_shape), [base_op, base_op],
                                   loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                   paddle_node.op_type)),
                                   ip=self.mlir.insert_point).output
                self.addOperand(paddle_node.name, mul_op)
                return
            else:
                pow_op = top.PowOp(self.mlir.get_tensor_type(output_shape),
                                   base_op,
                                   exponent=expn_const,
                                   loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                   paddle_node.op_type)),
                                   ip=self.mlir.insert_point).output
                self.addOperand(paddle_node.name, pow_op)
        else:
            raise RuntimeError("Not implemented")

    def convert_shape_op(self,paddle_node):
        assert (paddle_node.op_type == "shape")
        input = paddle_node.inputs[0]
        input_shape = self.getShape(input)
        input_dims = len(input_shape)
        start = paddle_node.attrs.get("start", 0)
        end = paddle_node.attrs.get("end", input_dims)
        op = self.getOp(input)
        mid_name = "{}_{}_{}".format(paddle_node.name, paddle_node.op_type, 0)
        final_name = "{}_{}".format(paddle_node.name, paddle_node.op_type)
        no_slice = start == 0 and end == input_dims
        if no_slice:
            mid_name = final_name
        new_op = top.ShapeOp(self.mlir.get_tensor_type([input_dims]),
                             op,
                             loc=self.get_loc(mid_name),
                             ip=self.mlir.insert_point).output
        if not no_slice:
            new_op = top.SliceOp(self.mlir.get_tensor_type([end - start]),
                                 new_op,
                                 self.mlir.none_op,
                                 self.mlir.none_op,
                                 self.mlir.none_op,
                                 offset=[start],
                                 steps=[1],
                                 ends=[end],
                                 loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                 paddle_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_sigmoid_op(self,paddle_node):
        assert (paddle_node.op_type == "sigmoid")
        op = self.getOperand(paddle_node.inputs[0])
        scale = paddle_node.attrs.get('scale', 1) #若没有这个参数 则返回1
        bias = paddle_node.attrs.get('bias', 0)
        output_shape = self.getShape(paddle_node.name)
        new_op = top.SigmoidOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               scale=scale,
                               bias=bias,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_slice_op(self,paddle_node):
        assert (paddle_node.op_type == "strided_slice")
        input_shape = self.getShape(paddle_node.inputs[0])
        output_shape = self.getShape(paddle_node.name)
        starts = []
        ends = []
        axes = []
        num_input = len(paddle_node.inputs)
        num_dims = len(input_shape)
        if num_input > 1:
            starts = self.getWeight(paddle_node.inputs[1]).astype(int)
            ends = self.getWeight(paddle_node.inputs[2]).astype(int)
            axes = self.getWeight(paddle_node.inputs[3]).astype(int) if num_input > 3 else list(
                np.arange(num_dims))
            steps = self.getWeight(
                paddle_node.inputs[4]).astype(int) if num_input > 4 else [1] * len(axes)
        else:
            starts = paddle_node.attrs.get('starts')
            ends = paddle_node.attrs.get('ends')
            axes = paddle_node.attrs.get('axes')
            if axes == None:
              axes_len = num_dims
              axes = [i for i in range(axes_len)]
            steps = [1] * len(axes)
        assert (len(starts) == len(ends))
        assert (len(axes) == len(ends))
        if self.isWeight(paddle_node.inputs[0]):
            tensor_data = self.getWeight(paddle_node.inputs[0])
            for start, end, axis, step in zip(starts, ends, axes, steps):
                start, end, axis, step = int(start), int(end), int(axis), int(step)
                if axis < 0:
                    axis = axis + num_dims
                s = slice(start, end, step)
                tensor_data = tensor_data[(slice(None), ) * axis + (s, )]
            self.addWeight(paddle_node.name, tensor_data)
            return
        op = self.getOperand(paddle_node.inputs[0])
        slice_shape = list(input_shape)
        slice_offset = [0] * num_dims
        slice_step = [1] * num_dims
        slice_end = [input_shape[i] for i in range(num_dims)]
        for start, end, axis, step in zip(starts, ends, axes, steps):
            start, end, axis, step = int(start), int(end), int(axis), int(step)
            if axis < 0:
                axis = axis + num_dims
            if end < 0:
                end = end + input_shape[axis]
            if start < 0:
                start = start + input_shape[axis]
            if end > input_shape[axis]:
                end = input_shape[axis]
            elif end < 0:
                if step < 0:
                    end = -1
                else:
                    end = input_shape[axis]
            slice_shape[axis] = (abs(end - start) + abs(step) - 1) // abs(step)
            slice_offset[axis] = start
            slice_step[axis] = step
            slice_end[axis] = end
        assert (slice_shape == output_shape)
        new_op = top.SliceOp(self.mlir.get_tensor_type(output_shape),
                             op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             offset=list(slice_offset),
                             steps=list(slice_step),
                             ends=list(slice_end),
                             loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                             ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)


    def convert_sub_op(self,paddle_node):
        assert (paddle_node.op_type == "elementwise_sub")
        assert (len(paddle_node.inputs) == 2)
        output_shape = self.getShape(paddle_node.name)
        lhs = paddle_node.inputs[0]
        rhs = paddle_node.inputs[1]
        name = "{}_{}".format(paddle_node.name, paddle_node.op_type)
        new_op = None
        if self.isScalar(lhs):
            # lhs_const + (-1 * rhs)

            rhs_op = self.getOp(rhs)
            unm_op = top.MulConstOp(self.mlir.get_tensor_type(output_shape),
                                    rhs_op,
                                    const_val=-1,
                                    loc=self.get_loc(name + "_unm"),
                                    ip=self.mlir.insert_point).output
            new_op = top.AddConstOp(self.mlir.get_tensor_type(output_shape),
                                    unm_op,
                                    const_val=self.getScalar(lhs),
                                    loc=self.get_loc(name),
                                    ip=self.mlir.insert_point).output
        elif self.isScalar(rhs):
            # lhs + (-rhs_const)
            lhs_op = self.getOp(lhs)
            new_op = top.AddConstOp(self.mlir.get_tensor_type(output_shape),
                                    lhs_op,
                                    const_val=-self.getScalar(rhs),
                                    loc=self.get_loc(name),
                                    ip=self.mlir.insert_point).output
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = top.SubOp(self.mlir.get_tensor_type(output_shape), [lhs_op, rhs_op],
                               loc=self.get_loc(name),
                               ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_transpose_op(self,paddle_node):
        assert (paddle_node.op_type == "transpose2")
        op = self.getOperand(paddle_node.inputs[0])
        input_shape = self.getShape(paddle_node.inputs[0])
        output_shape = self.getShape(paddle_node.name)
        # default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
        perm_default = list(np.arange(len(input_shape)))
        # perm_default = perm_default[:2] + perm_default[2:4][::-1] + [perm_default[-1]]
        perm_default = [perm_default[0], perm_default[1], perm_default[3], perm_default[4], perm_default[2]]
        transpose_perm = paddle_node.attrs.get('perm', perm_default)
        assert (len(input_shape) == len(transpose_perm))
        new_op = top.PermuteOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               order=transpose_perm,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)

    def convert_unsqueeze_op(self,paddle_node):
        assert (paddle_node.op_type == "unsqueeze2")
        op = self.getOperand(paddle_node.inputs[0])
        output_shape = self.getShape(paddle_node.name)
        # if self.opset < 13:
        #     axes = paddle_node.attrs.get('axes')
        # else:
        if len(paddle_node.inputs) == 1:
            axes = []
        else:
            axes = self.getWeight(paddle_node.inputs[1]).astype(int)
        new_op = top.UnsqueezeOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                               ip=self.mlir.insert_point,
                               axes=axes).output
        self.addOperand(paddle_node.name, new_op)

    def convert_upsample_op(self,paddle_node):
        assert (paddle_node.op_type == "nearest_interp_v2")
        mode = paddle_node.attrs.get("interp_method","nearest")
        op = self.getOperand(paddle_node.inputs[0])
        input_shape = self.getShape(paddle_node.inputs[0])
        scale_factor = []
        sizes = []
        scale_factor = np.array(paddle_node.attrs.get("scale",[1]))
        if (type(scale_factor) == np.ndarray and len(scale_factor.shape) == 2
            and scale_factor.shape[1] == 1):
            scale_factor = scale_factor.reshape(-1)
        sizes = input_shape[2:3] * scale_factor         #最近临只是对高度和宽度这俩个维度进行处理
                                                        #所以这里取NCHW的后俩个
        output_shape = []
        sizes_int = [int(i) for i in sizes]
        #output_shape[0] = input_shape[0]
        output_shape = input_shape[0:2] + sizes_int
        #output_shape.append(int(i) for i in sizes)
        scale_h = scale_factor[0]                       #这俩个不知道有什么作用，得到的都是-1
        scale_w = scale_factor[1]
        coord_mode = paddle_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        #这里的mode，onnx那边的类型是byte string，而paddle这边的type是string
        if mode == 'nearest' and scale_h == int(scale_h) and scale_w == int(scale_w):
            self.resize_to_upsample(paddle_node, op, input_shape, output_shape, scale_h, scale_w)
            return
        else:
            self.resize_to_interp(paddle_node,op, input_shape, output_shape, scale_h, scale_w, mode,
                                  coord_mode)
            return
            #print("this op is not supported!!!!")
    # when resize by linear or nearst, with float scale_h or float scale_w
    def resize_to_interp(self, paddle_node, op, input_shape, output_shape, scale_h, scale_w, mode,
                         coordinate_transformation_mode):
        new_op = top.InterpOp(self.mlir.get_tensor_type(output_shape),
                              op,
                              self.mlir.none_op,
                              scale_h=float(scale_h),
                              scale_w=float(scale_w),
                              mode=StringAttr.get(mode),
                              coord_mode=StringAttr.get(coordinate_transformation_mode),
                              loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)


    def resize_to_upsample(self,paddle_node,op,input_shape,output_shape,scale_h,scale_w):
        new_op = top.UpsampleOp(self.mlir.get_tensor_type(output_shape),
                                op,
                                scale_h=int(scale_h),
                                scale_w=int(scale_w),
                                loc=self.get_loc("{}_{}".format(paddle_node.name, paddle_node.op_type)),
                                ip=self.mlir.insert_point).output
        self.addOperand(paddle_node.name, new_op)
        # else:
        #     self.resize_to_interp(paddle_node, op, input_shape, output_shape, scale_h, scale_w, mode,
        #                           coord_mode)
        #     return




    def convert_where_op(self, paddle_node):
        assert (paddle_node.op_type == "where")
        assert (len(paddle_node.inputs) == 3)
        cond = paddle_node.inputs[0]
        tbrn = paddle_node.inputs[1]
        fbrn = paddle_node.inputs[2]
        cond_opd = self.getOp(cond)
        tbrn_opd = self.getOp(tbrn)
        fbrn_opd = self.getOp(fbrn)
        num_const = 0
        if self.isScalar(tbrn):
            num_const += 1
        # else:
        #     assert (self.getShape(cond) == self.getShape(tbrn)
        #             )  # do not support broadcastable case recently
        if self.isScalar(fbrn):
            num_const += 1
        # else:
        #     assert (self.getShape(cond) == self.getShape(fbrn)
        #             )  # do not support broadcastable case recently
        output_shape = self.getShape(paddle_node.name)
        if num_const == 0:
            new_op = top.WhereOp(self.mlir.get_tensor_type(output_shape),
                                 cond_opd,
                                 tbrn_opd,
                                 fbrn_opd,
                                 x_is_const=False,
                                 y_is_const=False,
                                 x_const_val=0,
                                 y_const_val=0,
                                 loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                 paddle_node.op_type)),
                                 ip=self.mlir.insert_point).output
        elif num_const == 1:
            brn_opd = fbrn_opd if self.isScalar(tbrn) else tbrn_opd
            if self.isScalar(tbrn):
                inversed = True
                const_val = self.getScalar(tbrn)
            else:
                inversed = False
                const_val = self.getScalar(fbrn)
            new_op = top.MaskedFillOp(self.mlir.get_tensor_type(output_shape),
                                      cond_opd,
                                      brn_opd,
                                      inversed=inversed,
                                      const_val=const_val,
                                      loc=self.get_loc("{}_{}".format(paddle_node.name,
                                                                      paddle_node.op_type)),
                                      ip=self.mlir.insert_point).output
        else:
            assert (0)  # TODO: to be implement
        self.addOperand(paddle_node.name, new_op)
