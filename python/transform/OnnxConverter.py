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
from .OnnxOpt import onnx_opt, ConstantFolding
from onnx import numpy_helper, mapping
from numbers import Number
import onnx
import numpy as np
from utils.pad_setting import set_auto_pad
from utils.auto_remove import file_mark, file_clean
import copy, sys
import mlir.dialects.top as top
from mlir.ir import *
from typing import List
import onnxsim.onnx_simplifier as onnxsim
import onnxruntime as rt
import logging
import copy
import time

logger = logging.getLogger("root")
sys.setrecursionlimit(1000000)

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = onnx.TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])
        self.shape_info = dict()

class OnnxNode(BaseNode):

    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0]
        info["op_type"] = node.op_type
        info["attrs"] = [(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
                         for attr in node.attribute]
        info["inputs"] = node.input
        info["outputs"] = node.output
        super().__init__(info)
        self.node_proto = node


class OnnxConverter(BaseConverter):

    def __init__(self,
                 model_name: str,
                 onnx_file,
                 input_shapes: list,
                 output_names: list,
                 test_input,
                 preprocess_args: dict = {},
                 static_shape=True,
                 onnx_sim="",
                 dynamic_shape_input_names=[],
                 shape_influencing_input_names=[],
                 dynamic=False,
                 dump_final_opt=True):
        super().__init__()

        self.dynamic_shape_input_names = dynamic_shape_input_names
        self.shape_influencing_input_names = shape_influencing_input_names
        # self.dynamic = dynamic
        if self.dynamic_shape_input_names or self.shape_influencing_input_names:
            self.dynamic = "manual"
            dynamic = True
        elif dynamic:
            self.dynamic = "auto"
        else:
            self.dynamic = "off"
        self.run_mode = "DYNAMIC" if dynamic else "STATIC"
        self.dynamic_shapes = dict()
        self.test_input = test_input
        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.node_name_mapping = {}  # used in onnx opt
        self.np_onnx_dt_map = [
            None, np.float32, np.uint8, np.int8, np.int16, np.int16, np.int32, np.int64, None,
            np.bool_, np.float16, np.float64, np.uint32, np.uint64, None, None, None
        ]
        self.onnx_sim = onnx_sim
        self.origin_output_names = output_names.copy()
        self.load_onnx_model(onnx_file, input_shapes, output_names, static_shape, dump_final_opt)
        self.init_MLIRImporter()
        self.unranked_type = self.mlir.get_tensor_type([])
        # some onnx may have strange domain, such as "ai.onnx.ml"
        for ver_info in self.model.opset_import:
            if ver_info.domain == "":
                self.opset = ver_info.version
                break
        self.preprocess_args = {}
        if 'preprocess_list' in preprocess_args:
            if preprocess_args['preprocess_list'] is not None:
                for input_index in preprocess_args['preprocess_list']:
                    assert( 0 < input_index <= self.num_input
                        and "Please check --preprocess_list is right input")
            else:
                preprocess_args['preprocess_list'] = [ i + 1 for i in range(self.num_input) ]
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.preprocess_args = preprocess_args
        self.converted_nodes = list()
        self.subgraph_initializer = None

        self.onnxop_factory = {
            # NOTICE: Please add the Op alphabetically !!!
            "Abs": lambda node: self.convert_abs_op(node),
            "Add": lambda node: self.convert_add_op(node),
            "Acos": lambda node: self.convert_arccos_op(node),
            "Atan": lambda node: self.convert_arctan_op(node),
            "Atanh": lambda node: self.convert_arctanh_op(node),
            "ArgMax": lambda node: self.convert_arg_op(node),
            "ArgMin": lambda node: self.convert_arg_op(node),
            "And": lambda node: self.convert_cmp_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Cast": lambda node: self.convert_cast_op(node),
            "Ceil": lambda node: self.convert_ceil_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "ConstantOfShape": lambda node: self.convert_constantofshape_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Cos": lambda node: self.convert_cos_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "ConvTranspose": lambda node: self.convert_conv_transpose_op(node),
            "CumSum": lambda node: self.convert_cumsum_op(node),
            "DepthToSpace": lambda node: self.convert_depth2space_op(node),
            "DequantizeLinear": lambda node: self.convert_deqlinear_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Einsum": lambda node: self.convert_einsum_op(node),
            "Elu": lambda node: self.convert_elu_op(node),
            "Erf": lambda node: self.convert_erf_op(node),
            "Exp": lambda node: self.convert_exp_op(node),
            "Expand": lambda node: self.convert_expand_op(node),
            "Equal": lambda node: self.convert_cmp_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Floor": lambda node: self.convert_floor_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "GatherElements": lambda node: self.convert_gather_elements_op(node),
            "GatherND": lambda node: self.convert_gathernd_op(node),
            "GELU": lambda node: self.convert_gelu_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "GroupNormalization": lambda node: self.convert_group_norm_op(node),
            "Greater": lambda node: self.convert_cmp_op(node),
            "GreaterOrEqual": lambda node: self.convert_cmp_op(node),
            "GridSample": lambda node: self.convert_grid_sampler_op(node),
            "GRU": lambda node: self.convert_gru_op(node),
            "HardSigmoid": lambda node: self.convert_hsigmoid_op(node),
            "HardSwish": lambda node: self.convert_hswish_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "InstanceNormalization": lambda node: self.convert_instance_norm_op(node),
            "LayerNormalization": lambda node: self.convert_layer_norm_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "Log": lambda node: self.convert_log_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "LSTM": lambda node: self.convert_lstm_op(node),
            "LogSoftmax": lambda node: self.convert_softmax_op(node),
            "Less": lambda node: self.convert_cmp_op(node),
            "LessOrEqual": lambda node: self.convert_cmp_op(node),
            "MatMul": lambda node: self.convert_gemm_op(node),
            "Max": lambda node: self.convert_max_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Min": lambda node: self.convert_min_op(node),
            "Mod": lambda node: self.convert_mod_op(node),
            "Mul": lambda node: self.convert_mul_op(node),
            "Neg": lambda node: self.convert_neg_op(node),
            "NonMaxSuppression": lambda node: self.convert_nms_op(node),
            "Not": lambda node: self.convert_not_op(node),
            "NonZero": lambda node: self.convert_nonzero_op(node),
            "OneHot": lambda node: self.convert_onehot_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "PixelNormalization": lambda node: self.convert_pixel_norm_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "Pow": lambda node: self.convert_pow_op(node),
            "QuantizeLinear": lambda node: self.convert_qlinear_op(node),
            "RandomNormalLike": lambda node: self.convert_random_normal_op(node),
            "Range": lambda node: self.convert_range_op(node),
            "Reciprocal": lambda node: self.convert_reciprocal_op(node),
            "ReduceMean": lambda node: self.convert_reduce_op(node),
            "ReduceMax": lambda node: self.convert_reduce_op(node),
            "ReduceMin": lambda node: self.convert_reduce_op(node),
            "ReduceL2": lambda node: self.convert_reduce_op(node),
            "ReduceL1": lambda node: self.convert_reduce_op(node),
            "ReduceProd": lambda node: self.convert_reduce_op(node),
            "ReduceSum": lambda node: self.convert_reduce_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "ReverseSequence": lambda node: self.convert_reverse_sequence_op(node),
            "RoiAlign": lambda node: self.convert_roi_align_op(node),
            "Round": lambda node: self.convert_round_op(node),
            "ScatterElements": lambda node: self.convert_scatter_elements_op(node),
            "ScatterND": lambda node: self.convert_scatternd_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid": lambda node: self.convert_sigmoid_op(node),
            "Sign": lambda node: self.convert_sign_op(node),
            "Sin": lambda node: self.convert_sin_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Softplus": lambda node: self.convert_softplus_op(node),
            "SpaceToDepth": lambda node: self.convert_space2depth_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Sum": lambda node: self.convert_sum_op(node),
            "Sqrt": lambda node: self.convert_sqrt_op(node),
            "Tanh": lambda node: self.convert_tanh_op(node),
            "Tile": lambda node: self.convert_tile_op(node),
            "TopK": lambda node: self.convert_topk_op(node),
            "Trilu": lambda node: self.convert_trilu_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "Upsample": lambda node: self.convert_upsample_op(node),
            "Where": lambda node: self.convert_where_op(node),
            "Xor": lambda node: self.convert_cmp_op(node),
            "If": lambda node: self.convert_if_op(node),
            "Loop": lambda node: self.convert_loop_op(node),
        }

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def cleanup(self):
        file_clean()

    def check_need(self, name):
        for node in self.converted_nodes:
            for i in node.inputs:
                if i == name:
                    return True
        if name in self.output_names:
            return True
        return False

    def select_unuse(self, names):
        for name in names:
            if name in self.all_weights:
                self.all_weights.pop(name)
            if name in self.all_values:
                self.all_values.pop(name)
            if name in self.all_inputs:
                self.all_inputs.pop(name)
            if name in self.all_nodes:
                cur_node = self.all_nodes.pop(name)
                for o in cur_node.output:
                    if o in self.all_nodes:
                        self.all_nodes.pop(o)
                self.select_unuse(cur_node.input)

    def select_output(self, output_names: list):
        # set new output
        self.all_outputs = []
        self.all_inputs = {}
        for x in self.model.graph.input:
            self.all_inputs[x.name] = x
        self.all_values = {}
        for x in self.model.graph.output:
            if x.name in output_names:
                self.all_outputs.append(x.name)
                output_names.remove(x.name)
                if len(output_names) == 0:
                    break
        for x in self.model.graph.value_info:
            self.all_values[x.name] = x
            if x.name not in output_names:
                continue
            self.model.graph.output.append(x)
            self.all_outputs.append(x.name)
            output_names.remove(x.name)
        # node map name
        self.all_nodes = {}
        for x in self.model.graph.node:
            for o in x.output:
                self.all_nodes[o] = x
                if o in output_names:
                    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                    intermediate_layer_value_info.name = o
                    self.model.graph.output.append(intermediate_layer_value_info)
                    output_names.remove(o)
                    self.all_outputs.append(o)
        if len(output_names) != 0:
            raise RuntimeError("Error, can't find {} in model".format(output_names))
        # weight map name
        self.all_weights = {}
        for w in self.model.graph.initializer:
            self.all_weights[w.name] = w
        # remove unused node
        self.select_unuse(self.all_outputs)
        for n in self.all_nodes.values():
            if n in self.model.graph.node:
                self.model.graph.node.remove(n)
        for w in self.all_weights.values():
            self.model.graph.initializer.remove(w)
        for i in self.all_inputs.values():
            self.model.graph.input.remove(i)
        for v in self.all_values.values():
            self.model.graph.value_info.remove(v)
        unuse_output = []
        for o in self.model.graph.output:
            if o.name not in self.all_outputs:
                unuse_output.append(o)
        for o in unuse_output:
            self.model.graph.output.remove(o)

    def get_value_info(self, name):
        for x in self.model.graph.value_info:
            if x.name == name:
                return x
        return None

    def get_outputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [opt for opt in model.graph.output if opt.name not in initializer_names]

    def get_inputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self, model: onnx.ModelProto):
        input_names = [ipt.name for ipt in self.get_inputs(model)]
        return input_names

    def get_input_types(self, model: onnx.ModelProto):
        input_types = []
        for input in self.get_inputs(model):
            if input.type.tensor_type.elem_type in [onnx.TensorProto.INT64, onnx.TensorProto.INT32]:
                input_types.append('INT32')
            else:
                input_types.append('F32')
        return input_types

    def get_output_types(self, model: onnx.ModelProto):
        output_types = []
        for output in self.get_outputs(model):
            if output.type.tensor_type.elem_type in [
                    onnx.TensorProto.INT64, onnx.TensorProto.INT32
            ]:
                output_types.append('INT32')
            else:
                output_types.append('F32')
        return output_types

    def get_shape_from_value_info_proto(self, v: onnx.ValueInfoProto):
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_input_shapes(self, model: onnx.ModelProto):
        inputs = self.get_inputs(model)
        return [self.get_shape_from_value_info_proto(i) for i in inputs]

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def is_dynamic(self):
        return self.run_mode == "DYNAMIC"

    def model_simplify(self, input_shapes=[]):
        # Do constantFolding before onnxsim to avoid onnxsim bug (such as run yolox)
        try:
            self.model = ConstantFolding(self.model, self.test_input, self.dynamic_shape_input_names).run()
        except:
            logger.warning("ConstantFolding failed.")
        logger.info("ConstantFolding finished")
        try:
            onnx_sim = self.onnx_sim.split(',')
            skip_fuse_bn = "skip_fuse_bn" in onnx_sim
            logger.info(f'skip_fuse_bn:{skip_fuse_bn}')
            self.model, _ = onnxsim.simplify(self.model,
                                             skip_fuse_bn=skip_fuse_bn,
                                             skip_constant_folding=True,
                                             skip_shape_inference=True)
        except:
            logger.warning("onnxsim opt failed.")
        logger.info("Onnxsim opt finished")
        if self.dynamic_shape_input_names:
            self.input_shape_assign(input_shapes)
            logger.info("Input_shape assigned")
        # Do constantFolding after onnxsim to avoid onnxsim bug (such as run ppyolo_tiny)
        try:
            self.model = ConstantFolding(self.model, self.test_input, self.dynamic_shape_input_names).run()
        except:
            logger.warning("ConstantFolding failed.")
        logger.info("ConstantFolding finished")

    def find_named_tensor(self, name):
        for tensor in self.model.graph.initializer:
            if name == tensor.name:
                return numpy_helper.to_array(tensor).astype(np.float32)
        if self.subgraph_initializer is not None:
            for tensor in self.subgraph_initializer:
                if name == tensor.name:
                    return numpy_helper.to_array(tensor).astype(np.float32)
        for node in self.converted_nodes:
            if node.op_type != "Constant":
                continue
            if node.name == name:
                onnx_tensor = node.attrs['value']
                return numpy_helper.to_array(onnx_tensor)

    def load_onnx_model(self, onnx_file, input_shapes: list, output_names: list, static_shape=True, dump_final_opt=True):
        if isinstance(onnx_file, str):
            self.model = onnx.load(onnx_file)
        else:
            self.model = onnx_file
        if output_names:
            self.select_output(output_names)
        self.input_names = self.get_input_names(self.model)
        self.num_input = len(self.input_names)
        self.dynamic_input_names_auto_assign()
        if not self.dynamic_shape_input_names:
            self.input_shape_assign(input_shapes)
            logger.info("Input_shape assigned")
            if static_shape:
                self.model_simplify()
        else:
            self.model_simplify(input_shapes)

        self.input_shapes = self.get_input_shapes(self.model)
        self.input_types = self.get_input_types(self.model)
        self.output_types = self.get_output_types(self.model)
        # add all weight
        for tensor in self.model.graph.initializer:
            name = tensor.name
            data = numpy_helper.to_array(tensor).astype(np.float32)
            self.addWeight(name, data)
            # TODO: for quantized onnx, keep the same type
        self.get_output_name(self.model.graph)
        # self.add_shape_info(self.model.graph)
        # if self.dynamic:
        #     self.get_dynamic_op_shape(self.model)
        self.onnx_file = "{}_opt.onnx".format(self.model_name)
        file_mark(self.onnx_file)
        try:
            onnx.save(self.model, self.onnx_file)
        except Exception as E:
            if "The proto size is larger than the 2 GB limit." in str(E):
                logger.info("LOG: Try to save {} by using save_as_external_data to save tensors separately from the model file.".format(self.onnx_file))
                onnx.save(self.model,
                          self.onnx_file,
                          save_as_external_data=True,
                          location=self.model_name + "_external_data",
                          convert_attribute=True)
            else:
                raise E
        strip_model = onnx.ModelProto()
        strip_model.CopyFrom(self.model)
        strip_model.graph.ClearField("initializer")
        with open(self.onnx_file + ".prototxt", "w") as f:
            f.write(str(strip_model))
        if static_shape:
            # fuse ops such as layernorm gelu...
            self.model, self.node_name_mapping = onnx_opt(self.model, dump_final_opt)

    def get_output_name(self, graph):
        for output in graph.output:
            if not self.isWeight(output.name):
                self.output_names.append(output.name)

    def addDynamicShape(self, name, shape):
        if len(shape) == 0:
            shape = [1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        if name in self.shapes:
            if self.shapes[name] != shape:
                raise KeyError("shape {} conflict {} vs {}".format(name, self.shapes[name], shape))
        self.dynamic_shapes[name] = shape

    def getDynamicShape(self, name):
        if name not in self.dynamic_shapes:
            logger.warning("shape {} not found in dynamic_shapes".format(name))
            return None
        return self.dynamic_shapes[name]

    def get_dynamic_op_shape(self, model):
        dynamic_op = ["RandomNormalLike", "Range"]
        ori_outputs = []
        ori_outputs.extend(model.graph.output)
        del self.model.graph.output[:]
        for node in model.graph.node:
            if node.op_type in dynamic_op:
                for output in node.output:
                    if not self.isWeight(output):
                        model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        if model.graph.output:
            ort_inputs = {}
            test_file = ''
            if isinstance(self.test_input, list):
                assert self.test_input[0].endswith('.npz')
                test_file = self.test_input[0]
            elif isinstance(self.test_input, str):
                assert self.test_input.endswith('.npz')
                test_file = self.test_input
            else:
                raise ValueError("test_input npz file is necessary when transform dynamic shape model")
            test_data = np.load(test_file)
            for i in test_data.files:
                ort_inputs[i] = test_data[i]
            try:
                try:
                    ort_session = rt.InferenceSession(model.SerializeToString())
                except Exception as E:
                    if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(E):
                        print("LOG: Try to convert through a temporary file when Constant Folding.")
                        # large models try to convert through a temporary file
                        import os
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            model_path = os.path.join(tmpdirname, 'dynamic_model.onnx')
                            onnx.save(model,
                                    model_path,
                                    save_as_external_data=True,
                                    location="temp_external_data",
                                    convert_attribute=True)
                            ort_session = rt.InferenceSession(model.SerializeToString())
                    else:
                        raise E
            except ValueError:
                logger.warning("onnxruntime.InferenceSession error when getting dynamic output shape.")

            # ort_session = rt.InferenceSession(model.SerializeToString())
            outputs = [x.name for x in ort_session.get_outputs()]
            ort_outs = ort_session.run(outputs, ort_inputs)
            ort_outs_shape = [x.shape for x in ort_outs]
            for i, output in enumerate(outputs):
                self.addDynamicShape(output, ort_outs_shape[i])
            del self.model.graph.output[:]
        model.graph.output.extend(ori_outputs)

    def dynamic_input_names_auto_assign(self):
        if self.dynamic != "auto":
            return
        inputs = self.get_inputs(self.model)
        for idx, input in enumerate(inputs):
            _dims = input.type.tensor_type.shape.dim
            if _dims:
                if len(_dims) == 1:
                    self.shape_influencing_input_names.append(input.name)
                for _i, _dim in enumerate(_dims):
                    if _dim.dim_value == 0 and _dim.dim_param:
                        self.dynamic_shape_input_names.append(input.name)
                        break

    def input_shape_assign(self, input_shapes):
        inputs = self.get_inputs(self.model)
        outputs = self.get_outputs(self.model)
        shape_changed = False
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input, len(input_shapes))
        for idx, input in enumerate(inputs):
            _dims = input.type.tensor_type.shape.dim
            # for 1-element scalars that has no shape, assign [1] as shape to convert to tensor
            if not _dims:
                _dims.append(onnx.TensorShapeProto.Dimension(dim_value=1))
            num_dims = len(_dims)
            if no_shape == False:
                check_shape(num_dims, len(input_shapes[idx]))
            _shape = []
            for _i, _dim in enumerate(_dims):
                if _dim.dim_value <= 0:
                    if no_shape:
                        assert 0, "Please check --input_shapes formula or check if there is any dynamic dim"
                    else:
                        _dim.dim_value = input_shapes[idx][_i]
                # elif not no_shape:
                #     check_shape(_dim_value, input_shapes)
                elif not no_shape and input_shapes[idx][_i] != _dim.dim_value:
                    _dim.dim_value = input_shapes[idx][_i]
                    shape_changed = True
                _shape.append(_dim.dim_value)
            self.addShape(input.name, _shape)
        idx = 0  # avoid confilict for multi dynamic axes
        for o in outputs:
            # for set arbitrary axes
            _odims = o.type.tensor_type.shape.dim
            for _odim in _odims:
                if _odim.dim_value <= 0 or shape_changed:
                    _odim.dim_param = '?_' + str(idx)
                    idx += 1

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        output_shapes = len(self.output_names) * [[]]
        for i, o in enumerate(self.output_names):
            if o in self.dynamic_shapes:
                output_shapes[i] = self.getDynamicShape(o)
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, Platform.ONNX,
                                 self.input_types, run_mode=self.run_mode)
        self.weight_file = self.mlir.weight_file

    def get_shape_for_node(self, input, output, value_info, name):
        for i in value_info:
            if i.name == name:
                return i.type.tensor_type.shape.dim
        for i in input:
            if i.name == name:
                return i.type.tensor_type.shape.dim
        for i in output:
            if i.name == name:
                return i.type.tensor_type.shape.dim

    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
        # add input op
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
        for idx, _name in enumerate(self.input_names):
            kwargs = copy.deepcopy(self.preprocess_args)
            if _name in self.shape_influencing_input_names:
                assert input_data[_name].ndim == 1, "input shape tensor should be 1D tensor"
                kwargs['shape_tensor'] = input_data[_name]
            input_ = self.mlir.create_input_op(self.get_loc(_name), idx, kwargs)
            self.addOperand(_name, input_)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.converted_nodes.clear()
        for n in self.model.graph.node:
            node = OnnxNode(n)
            if n.op_type in ["Gather"]:
                input_shape = dict()
                for input in n.input:
                    input_shape[input] = self.get_shape_for_node(self.model.graph.input, self.model.graph.output, self.model.graph.value_info, input)
                output_shape = dict()
                for output in n.output:
                    output_shape[output] = self.get_shape_for_node(self.model.graph.input, self.model.graph.output, self.model.graph.value_info, output)
                node.shape_info["input"] = input_shape
                node.shape_info["output"] = output_shape
            self.converted_nodes.append(node)
        # checkout all type is supported
        unsupported = set()
        for n in self.converted_nodes:
            if n.op_type not in self.onnxop_factory:
                unsupported.add(n.op_type)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        for n in self.converted_nodes:
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
        # add return op
        return_op = list()
        # Set output
        final_output_names = []
        if self.origin_output_names:
            final_output_names = self.origin_output_names
        else:
            final_output_names = self.output_names
        for idx, _name in enumerate(final_output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        logger.info("Save mlir file: {}".format(mlir_file))

    def convert_skip_op(self, onnx_node):
        op = self.getOperand(onnx_node.inputs[0])
        self.addOperand(onnx_node.name, op)

    def convert_add_op(self, onnx_node):
        assert (onnx_node.op_type == "Add")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        lhs_op = self.getOp(lhs)
        rhs_op = self.getOp(rhs)
        new_op = top.AddOp(self.unranked_type, [lhs_op, rhs_op],
                            loc=self.get_loc(name),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_sub_op(self, onnx_node):
        assert (onnx_node.op_type == "Sub")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        new_op = None
        if self.isScalar(lhs):
            # lhs_const + (-1 * rhs)
            rhs_op = self.getOp(rhs)
            new_op = top.SubConstOp(self.unranked_type,
                                    rhs_op,
                                    const_val=self.getScalar(lhs),
                                    loc=self.get_loc(name),
                                    is_reverse=True,
                                    ip=self.mlir.insert_point).output
        elif self.isScalar(rhs):
            # lhs + (-rhs_const)
            lhs_op = self.getOp(lhs)
            new_op = top.AddConstOp(self.unranked_type,
                                    lhs_op,
                                    const_val=-self.getScalar(rhs),
                                    loc=self.get_loc(name),
                                    ip=self.mlir.insert_point).output
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = top.SubOp(self.unranked_type, [lhs_op, rhs_op],
                               loc=self.get_loc(name),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_batchnorm_op(self, onnx_node):
        assert (onnx_node.op_type == "BatchNormalization")
        op = self.getOperand(onnx_node.inputs[0])
        gamma = self.getWeightOp(onnx_node.inputs[1])
        beta = self.getWeightOp(onnx_node.inputs[2])
        mean = self.getWeightOp(onnx_node.inputs[3])
        variance = self.getWeightOp(onnx_node.inputs[4])
        epsilon = onnx_node.attrs.get("epsilon")
        new_op = top.BatchNormOp(self.unranked_type,
                                 op,
                                 mean,
                                 variance,
                                 gamma,
                                 beta,
                                 epsilon=epsilon,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_cast_op(self, onnx_node):
        assert (onnx_node.op_type == "Cast")
        if self.isWeight(onnx_node.inputs[0]):
            data = self.getWeight(onnx_node.inputs[0])
            self.addWeight(onnx_node.name, data)
        else:
            op = self.getOperand(onnx_node.inputs[0])
            self.addOperand(onnx_node.name, op)

    def convert_ceil_op(self, onnx_node):
        assert (onnx_node.op_type == "Ceil")
        op = self.getOp(onnx_node.inputs[0])
        new_op = top.CeilOp(self.unranked_type,
                            op,
                            loc=self.get_loc(onnx_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_concat_op(self, onnx_node):
        assert (onnx_node.op_type == "Concat")
        axis = onnx_node.attrs['axis']
        operands = list()
        weight_data = None
        last_name = None
        for x in onnx_node.inputs:
            if self.isWeight(x):
                last_name = x
                data = self.getWeight(x)
                if len(data.shape) == 1 and data.shape[0] == 0:
                    continue
                if weight_data is not None:
                    weight_data = np.concatenate((weight_data, data), axis=axis)
                else:
                    weight_data = data
                continue
            else:
                if weight_data is not None:
                    w_name = last_name + "_weight"
                    self.addWeight(w_name, weight_data)
                    operands.append(self.getWeightOp(w_name))
                    weight_data = None
                operands.append(self.getOperand(x))
        if len(operands) == 0:
            # all weight
            self.addWeight(onnx_node.name, weight_data)
            return
        if weight_data is not None:
            w_name = onnx_node.name + "_weight"
            self.addWeight(w_name, weight_data)
            operands.append(self.getWeightOp(w_name))
        if len(operands) == 1:
            self.addOperand(onnx_node.name, operands[0])
            return
        new_op = top.ConcatOp(self.unranked_type,
                              operands,
                              axis=axis,
                              loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_constant_op(self, onnx_node):
        """
            Constant Op is tensor data at IR,
            we change it to load weight tensor, and store
        """
        assert (onnx_node.op_type == "Constant")
        onnx_tensor = onnx_node.attrs['value']
        np_tensor = numpy_helper.to_array(onnx_tensor)
        self.addWeight(onnx_node.name, np_tensor)

    def convert_constantofshape_op(self, onnx_node):
        """
            Constant Op is tensor data at IR,
            we change it to load weight tensor, and store
        """
        assert (onnx_node.op_type == "ConstantOfShape")
        value = 0
        if 'value' in onnx_node.attrs:
            onnx_tensor = onnx_node.attrs['value']
            np_tensor = numpy_helper.to_array(onnx_tensor)
            assert (np_tensor.size == 1)
            value = np_tensor[0]
        op = self.getOp(onnx_node.inputs[0])
        new_op = top.ConstantFillOp(self.unranked_type,
                                    op,
                                    value=value,
                                    loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                    onnx_node.op_type)),
                                    ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_op(self, onnx_node):
        assert (onnx_node.op_type == "Conv")
        op = self.getOp(onnx_node.inputs[0]) # input can be weight
        kernel_shape = onnx_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get("dilations", dim * [1])
        group = onnx_node.attrs.get("group", 1)
        strides = onnx_node.attrs.get("strides", dim * [1])
        auto_pad = onnx_node.attrs.get("auto_pad", "NOTSET")
        if not isinstance(auto_pad, str):
            auto_pad = auto_pad.decode('utf-8')
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        operands = list()
        operands.append(op)
        # filter may be dynamic weight
        filter_op = self.getOp(onnx_node.inputs[1])
        operands.append(filter_op)
        weight_is_coeff = 1 if self.isWeight(onnx_node.inputs[1]) else 0
        if len(onnx_node.inputs) > 2:
            bias_op = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        new_op = top.ConvOp(self.unranked_type,
                            *operands,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            auto_pad=StringAttr.get(auto_pad),
                            pads=pads,
                            group=group,
                            weight_is_coeff=weight_is_coeff,
                            do_relu=False,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_depth2space_op(self, onnx_node):
        assert (onnx_node.op_type == "DepthToSpace")
        op = self.getOperand(onnx_node.inputs[0])
        blocksize = onnx_node.attrs['blocksize']
        mode = onnx_node.attrs.get("mode", b"DCR")
        new_op = top.Depth2SpaceOp(self.unranked_type,
                                   op,
                                   block_h=blocksize,
                                   block_w=blocksize,
                                   is_CRD=(mode != b"DCR"),
                                   is_inversed=False,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_space2depth_op(self, onnx_node):
        assert (onnx_node.op_type == "SpaceToDepth")
        op = self.getOperand(onnx_node.inputs[0])
        blocksize = onnx_node.attrs['blocksize']
        mode = onnx_node.attrs.get("mode", b"DCR")
        new_op = top.Depth2SpaceOp(self.unranked_type,
                                   op,
                                   block_h=blocksize,
                                   block_w=blocksize,
                                   is_CRD=(mode != b"DCR"),
                                   is_inversed=True,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_flatten_op(self, onnx_node):
        assert (onnx_node.op_type == "Flatten")
        op = self.getOperand(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 1)
        new_op = top.FlattenOp(self.unranked_type,
                               op,
                               start_dim=axis,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_floor_op(self, onnx_node):
        assert (onnx_node.op_type == "Floor")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.FloorOp(self.unranked_type,
                             op,
                             loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                             ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gemm_op(self, onnx_node):
        assert (onnx_node.op_type == "Gemm" or onnx_node.op_type == 'MatMul')
        # (M, K) * (K, N) => (M, N)
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        # TODO:support more situations
        assert (trans_a == 0)
        operands = list()
        A = onnx_node.inputs[0]
        B = onnx_node.inputs[1]
        if self.isWeight(A):
            if trans_a == 1 or alpha != 1:
                _tensor = self.getWeight(A)
                _tensor = copy.deepcopy(_tensor)  #if change weight,should do deepcopy
                if trans_a == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                A += '_fix'
                self.addWeight(A, _tensor)
            operands.append(self.getWeightOp(A))
        else:
            operands.append(self.getOperand(A))

        if self.isWeight(B):
            if trans_b == 1 or alpha != 1:
                _tensor = self.getWeight(B)
                _tensor = copy.deepcopy(_tensor)  #if change weight,should do deepcopy
                if trans_b == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                B += '_fix'
                self.addWeight(B, _tensor)
            operands.append(self.getWeightOp(B))
        else:
            operands.append(self.getOperand(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if self.isWeight(C):
                if beta != 1:
                    _tensor = self.getWeight(C)
                    _tensor = copy.deepcopy(_tensor)  #if change weight,should do deepcopy
                    _tensor *= beta
                    C += '_fix'
                    self.addWeight(C, _tensor)
                operands.append(self.getWeightOp(C))
            else:
                operands.append(self.getOperand(C))
        else:
            operands.append(self.mlir.none_op)

        new_op = top.MatMulOp(self.unranked_type,
                              *operands,
                              do_relu=False,
                              loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_global_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalMaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.MaxPoolOp(self.unranked_type,
                               op,
                               kernel_shape=[],
                               strides=[],
                               pads=[],
                               count_include_pad=True,
                               do_relu=False,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_global_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalAveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        # check onnx define
        new_op = top.AvgPoolOp(self.unranked_type,
                               op,
                               kernel_shape=[],
                               strides=[],
                               pads=[],
                               count_include_pad=True,
                               do_relu=False,
                               keepdims=True,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "AveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        ceil_mode = onnx_node.attrs.get("ceil_mode", False)
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        strides = onnx_node.attrs.get("strides", kernel_shape)
        auto_pad = onnx_node.attrs.get("auto_pad", b"NOTSET")
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        if np.prod(kernel_shape) == 1 and np.sum(pads) == 0 and np.prod(strides) == 1:
            self.addOperand(onnx_node.name, op)
            return
        new_op = top.AvgPoolOp(self.unranked_type,
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads,
                               auto_pad=StringAttr.get(auto_pad),
                               ceil_mode=ceil_mode,
                               count_include_pad=count_include_pad,
                               do_relu=False,
                               keepdims=True,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "MaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        ceil_mode = onnx_node.attrs.get("ceil_mode", False)
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        strides = onnx_node.attrs.get("strides", kernel_shape)
        auto_pad = onnx_node.attrs.get("auto_pad", b"NOTSET")
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        new_op = top.MaxPoolOp(self.unranked_type,
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads,
                               auto_pad=StringAttr.get(auto_pad),
                               ceil_mode=ceil_mode,
                               count_include_pad=count_include_pad,
                               do_relu=False,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_mul_op(self, onnx_node):
        assert (onnx_node.op_type == "Mul")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = rhs, lhs
            self.convert_mul_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        op0 = self.getOp(lhs)
        op1 = self.getOp(rhs)
        mul_op = top.MulOp(self.unranked_type, [op0, op1],
                            loc=self.get_loc(name),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, mul_op)
        return

    def convert_dropout_op(self, onnx_node):
        assert (onnx_node.op_type == "Dropout")
        op = self.getOperand(onnx_node.inputs[0])
        self.addOperand(onnx_node.name, op)

    def convert_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "Relu")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.ReluOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_leaky_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "LeakyRelu")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.LeakyReluOp(self.unranked_type,
                                 op,
                                 alpha=onnx_node.attrs.get("alpha", 0.),
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_reshape_op(self, onnx_node):
        assert (onnx_node.op_type == "Reshape")
        op = self.getOperand(onnx_node.inputs[0])
        if self.isWeight(onnx_node.inputs[1]):
            shape = self.getWeight(onnx_node.inputs[1])
            new_op = top.ReshapeOp(self.unranked_type,
                                op,
                                shape=shape,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        else:
            shape = self.getOperand(onnx_node.inputs[1])
            new_op = top.ReshapeOp(self.unranked_type,
                                   op,
                                   shapeT=shape,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_reverse_sequence_op(self, onnx_node):
        assert (onnx_node.op_type == 'ReverseSequence')
        op = self.getOperand(onnx_node.inputs[0])
        batch_axis = onnx_node.attrs['batch_axis']
        time_axis = onnx_node.attrs['time_axis']
        axis_dict = {'batch_axis': batch_axis, 'time_axis': time_axis}
        assert(sorted(axis_dict.values()) == [0,1])
        # not suppport length of the sequences each batch equals to time temporarily
        new_op = top.ReverseOp(self.unranked_type,
                               op,
                               axis = time_axis,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    # when resize by linear or nearst, with float scale_h or float scale_w
    def resize_to_interp(self, onnx_node, op, scale_h, scale_w, mode,
                         coordinate_transformation_mode, target_shape):
        new_op = top.InterpOp(self.unranked_type,
                              op,
                              target_shape,
                              scale_h=float(scale_h),
                              scale_w=float(scale_w),
                              mode=StringAttr.get(mode),
                              coord_mode=StringAttr.get(coordinate_transformation_mode),
                              loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_upsample_op(self, onnx_node):
        assert (onnx_node.op_type == "Upsample")
        mode = onnx_node.attrs.get("mode", "nearest")
        op = self.getOperand(onnx_node.inputs[0])
        scale_factor = []
        scale_factor = self.getWeight(onnx_node.inputs[1])
        scale_factor = copy.deepcopy(scale_factor)  #if change it,should do deepcopy first
        if (type(scale_factor) == np.ndarray and len(scale_factor.shape) == 2
                and scale_factor.shape[1] == 1):
            scale_factor = scale_factor.reshape(-1)
        scale_h = scale_factor[2]  # scale [n, c, h, w]
        scale_w = scale_factor[3]
        coord_mode = onnx_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        self.resize_to_interp(onnx_node,
                              op,
                              scale_h,
                              scale_w,
                              mode,
                              coord_mode,
                              target_shape=self.mlir.none_op)
        return

    def convert_resize_op(self, onnx_node):
        assert (onnx_node.op_type == "Resize")
        mode = onnx_node.attrs.get("mode", "nearest")

        op = self.getOp(onnx_node.inputs[0])
        scale_factor = []
        sizes = []
        use_size = False
        target_shape = self.mlir.none_op
        if len(onnx_node.inputs) > 2:
            # onnx opset 11
            try:
                scale_factor = self.getWeight(onnx_node.inputs[2])
                scale_factor = copy.deepcopy(scale_factor)  #if change it,should do deepcopy first
            except KeyError:
                scale_factor = []
            if (type(scale_factor) == np.ndarray and len(scale_factor.shape) == 2
                    and scale_factor.shape[1] == 1):
                dims = scale_factor.shape[0]
                scale_factor = scale_factor.reshape(dims)
            if len(scale_factor) == 0:
                try:
                    sizes = self.getWeight(onnx_node.inputs[3])
                    assert(len(sizes) >= 2)
                    scale_factor = sizes
                    use_size = True
                except KeyError:
                    sizes = self.getOp(onnx_node.inputs[3])
                    use_size = True
        else:
            # opset 10
            scale_factor = self.getWeight(onnx_node.inputs[1])

        if (use_size):
            scale_d, scale_h, scale_w = -1, -1, -1
            if len(scale_factor) == 0:
                target_shape = sizes
            else:
                self.addWeight(onnx_node.name + "_target_shape",
                            np.array(scale_factor[2:], dtype=np.int64))
                target_shape = self.getWeightOp(onnx_node.name + "_target_shape")
        else:
            scale_d = -1 if len(scale_factor) <= 4 else scale_factor[-3]
            scale_h = -1 if len(scale_factor) <= 3 else scale_factor[-2]
            scale_w = scale_factor[-1]
            if scale_h == 1.0 and scale_w == 1.0:
                self.addOperand(onnx_node.name, op)
                return

        coord_mode = onnx_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        if coord_mode == b'tf_half_pixel_for_nn':       # different mode name in tf and pyt
            coord_mode = 'half_pixel'
        if mode == b'cubic':
            logging.warning("Not Support Resize Cubic !!!! Using Linear Instead !!!!!")
            time.sleep(3)
            mode = b'linear'
        self.resize_to_interp(onnx_node,
                              op,
                              scale_h,
                              scale_w,
                              mode,
                              coord_mode,
                              target_shape=target_shape)

    def convert_shape_op(self, onnx_node):
        assert (onnx_node.op_type == "Shape")
        input = onnx_node.inputs[0]
        start = onnx_node.attrs.get("start", 0)
        end = onnx_node.attrs.get("end", sys.maxsize)
        op = self.getOp(input)
        final_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        new_op = top.ShapeOp(self.unranked_type,
                             op,
                             start=start,
                             end=end,
                             loc=self.get_loc(final_name),
                             ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_random_normal_op(self, onnx_node):
        assert (onnx_node.op_type == "RandomNormalLike")
        input =  self.getOp(onnx_node.inputs[0])
        weight_shape = self.getDynamicShape(onnx_node.name)
        weight = np.random.randn(*weight_shape).astype(np.float32)
        weight_name = onnx_node.name + "_weight"
        self.addWeight(weight_name, weight)
        randn_data = self.getWeightOp(weight_name)

        randn_like_op = top.RandnLikeOp(self.unranked_type,
                                        input,
                                        randn_data,
                                        loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                        ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, randn_like_op)

    def convert_range_op(self, onnx_node):
        assert (onnx_node.op_type == "Range")
        start = self.getOp(onnx_node.inputs[0])
        limit = self.getOp(onnx_node.inputs[1])
        delta = self.getOp(onnx_node.inputs[2])
        range_op = top.RangeOp(self.unranked_type,
                            start,
                            limit,
                            delta,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, range_op)

    def convert_sigmoid_op(self, onnx_node):
        assert (onnx_node.op_type == "Sigmoid")
        op = self.getOperand(onnx_node.inputs[0])
        scale = onnx_node.attrs.get('scale', 1)
        bias = onnx_node.attrs.get('bias', 0)
        new_op = top.SigmoidOp(self.unranked_type,
                               op,
                               scale=scale,
                               bias=bias,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_sign_op(self, onnx_node):
        assert (onnx_node.op_type == "Sign")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.SignOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_sin_op(self, onnx_node):
        assert (onnx_node.op_type == "Sin")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.SinOp(self.unranked_type,
                           op,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_cos_op(self, onnx_node):
        assert (onnx_node.op_type == "Cos")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.CosOp(self.unranked_type,
                           op,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_slice_op(self, onnx_node):
        def try_get_slice_input(node, i, attr):
            is_const = self.isWeight(node.inputs[i])
            ret_list = []
            ret_op = self.mlir.none_op
            if is_const:
                ret_list = self.getWeight(node.inputs[i])
                ret_op = self.getWeightOp(node.inputs[i])
            elif attr in node.attrs:
                ret_list = node.attrs.get(attr)
                is_const = True
            else:
                ret_op = self.getOperand(node.inputs[i])
            return ret_list, ret_op, is_const

        assert (onnx_node.op_type == "Slice")
        starts = []
        ends = []
        axes = []
        steps = [1]
        num_input = len(onnx_node.inputs)
        if num_input > 1:
            op = self.getOperand(onnx_node.inputs[0]) if not self.isWeight(onnx_node.inputs[0]) \
                else self.getWeightOp(onnx_node.inputs[0])
            starts, start_op, starts_is_const = try_get_slice_input(onnx_node, 1, 'starts')
            ends, end_op, ends_is_const = try_get_slice_input(onnx_node, 2, 'ends') \
                if num_input > 2 else (ends, self.mlir.none_op, True)
            axes, axis_op, axes_is_const = try_get_slice_input(onnx_node, 3, 'axes') \
                if num_input > 3 else (list(np.arange(len(ends))), self.mlir.none_op, True)
            steps, step_op, steps_is_const = try_get_slice_input(onnx_node, 4, 'steps') \
                if num_input > 4 else ([1] * len(axes), self.mlir.none_op, True)
            if steps[0] == -1 and starts[0] == -1 and ends == -np.iinfo(np.int64).max and self.isWeight(op):
                in0 = op
                indices_op_name = 'indices_'+onnx_node.inputs[0]
                extra_attr = {}
                dim_length = self.getWeight(onnx_node.inputs[axes])
                np_tensor = np.arange(dim_length, -1, -1, dtype=np.int64)
                self.addWeight(indices_op_name, np_tensor)
                indices_op = self.getWeightOp(indices_op_name)
                extra_attr.update({"keepdims": True})
                indices = indices_op
                new_op = top.GatherOp(self.unranked_type,
                        in0,
                        indices,
                        axis=axes,
                        **extra_attr,
                        loc=self.get_loc("{}_{}".format(onnx_node.name, 'Gather')),
                        ip=self.mlir.insert_point).output
                self.addOperand(onnx_node.name, new_op)
                return

            ends = list(map(lambda x: np.iinfo(np.int64).max if x >= np.iinfo(np.int64).max else x, ends))
            if not (starts_is_const * ends_is_const * axes_is_const * steps_is_const):
                new_op = top.SliceAxisOp(self.unranked_type,
                                         op,
                                         axis_op,
                                         start_op,
                                         step_op,
                                         end_op,
                                         loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                         ip=self.mlir.insert_point).output
                self.addOperand(onnx_node.name, new_op)
                return
        else:
            starts = onnx_node.attrs.get('starts')
            ends = onnx_node.attrs.get('ends')
            axes = onnx_node.attrs.get('axes')
            if axes == None:
                axes_len = len(ends)
                axes = [i for i in range(axes_len)]
            steps = [1] * len(axes)
        assert (len(starts) == len(ends))
        assert (len(axes) == len(ends))
        if self.isWeight(onnx_node.inputs[0]):
            tensor_data = self.getWeight(onnx_node.inputs[0])
            num_dims = len(tensor_data.shape)
            for start, end, axis, step in zip(starts, ends, axes, steps):
                start, end, axis, step = int(start), int(end), int(axis), int(step)
                if axis < 0:
                    axis = axis + num_dims
                s = slice(start, end, step)
                tensor_data = tensor_data[(slice(None), ) * axis + (s, )]
            self.addWeight(onnx_node.name, tensor_data)
            return
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.SliceOp(self.unranked_type,
                             op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             offset=list(starts),
                             steps=list(steps),
                             ends=list(ends),
                             axes=list(axes),
                             loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                             ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_transpose_op(self, onnx_node):
        assert (onnx_node.op_type == "Transpose")
        op = self.getOp(onnx_node.inputs[0])
        transpose_perm = onnx_node.attrs.get('perm', [])
        new_op = top.PermuteOp(self.unranked_type,
                               op,
                               order=transpose_perm,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_softmax_op(self, onnx_node):
        assert (onnx_node.op_type in ("Softmax", "LogSoftmax"))
        op = self.getOperand(onnx_node.inputs[0])
        axis_default = -1 if self.opset >= 13 else 1
        axis = onnx_node.attrs.get('axis', axis_default)
        new_op = top.SoftmaxOp(self.unranked_type,
                               op,
                               axis=axis,
                               log=onnx_node.op_type == "LogSoftmax",
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_softplus_op(self, onnx_node):
        assert (onnx_node.op_type == "Softplus")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.SoftplusOp(self.unranked_type,
                                op,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_log_op(self, onnx_node):
        assert (onnx_node.op_type == "Log")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.LogOp(self.unranked_type,
                           op,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    # https://pytorch.org/docs/1.13/generated/torch.einsum.html?highlight=einsum#torch.einsum
    def convert_einsum_op(self, onnx_node):
        assert (onnx_node.op_type == "Einsum")
        equation = onnx_node.attrs.get("equation").decode()

        def normalize_equation(equation_c):
            equation = equation_c
            new_equation = ''
            start = 'a'
            translate_map = {}
            for s in equation:
                if s == ' ':
                    continue
                elif not ((s >= 'a' and s <= 'z') or (s >= 'A' and s <= 'Z')):
                    translate_map[s] = s
                elif s not in translate_map:
                    translate_map[s] = start
                    start = chr(ord(start) + 1)
                new_equation += translate_map[s]
            return new_equation

        equation = normalize_equation(equation)
        lhs = self.getOp(onnx_node.inputs[0])
        rhs = self.getOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) == 2:
            new_op = top.EinsumOp(self.unranked_type,
                                [lhs, rhs],
                                mode=StringAttr.get(equation),
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        elif len(onnx_node.inputs) == 3:
            dhs = self.getOp(onnx_node.inputs[2])
            new_op = top.EinsumOp(self.unranked_type,
                                [lhs, rhs, dhs],
                                mode=StringAttr.get(equation),
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        else:
            raise RuntimeError("This mode not support yet: {}".format(mode))
        self.addOperand(onnx_node.name, new_op)

    def convert_exp_op(self, onnx_node):
        assert (onnx_node.op_type == "Exp")
        op = self.getOp(onnx_node.inputs[0])
        new_op = top.ExpOp(self.unranked_type,
                           op,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_elu_op(self, onnx_node):
        assert (onnx_node.op_type == "Elu")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.EluOp(self.unranked_type,
                           op,
                           alpha=onnx_node.attrs.get("alpha", 0.),
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_erf_op(self, onnx_node):
        assert (onnx_node.op_type == "Erf")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.ErfOp(self.unranked_type,
                           op,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_pad_op(self, onnx_node):
        assert (onnx_node.op_type == "Pad")
        op = self.getOperand(onnx_node.inputs[0])
        # get pad mode
        mode = onnx_node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        assert (mode in ("constant", "reflect", "edge"))
        if len(onnx_node.inputs) > 1:
            pads = list(self.getWeight(onnx_node.inputs[1]))
        else:
            pads = onnx_node.attrs.get("pads")
        if pads == None:
            raise RuntimeError("No paddings value")
        # opset 11, value from second input
        val = 0.0
        if len(onnx_node.inputs) > 2 and onnx_node.inputs[2]:
            val = self.getWeight(onnx_node.inputs[2])
        else:
            val = onnx_node.attrs.get("value", 0.0)

        new_op = top.PadOp(self.unranked_type,
                           op,
                           paddings=pads,
                           val=val,
                           mode=StringAttr.get(mode),
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_div_op(self, onnx_node):
        assert (onnx_node.op_type == "Div")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if self.isScalar(lhs):
            # lhs_const * (1 / rhs)
            rhs_op = self.getOp(rhs)
            new_op = top.ReciprocalOp(self.unranked_type,
                                      rhs_op,
                                      const_val=self.getScalar(lhs),
                                      loc=self.get_loc(name),
                                      ip=self.mlir.insert_point).output
        elif self.isScalar(rhs):
            # lhs * (1 / rhs_const)
            lhs_op = self.getOp(lhs)
            lhs_type = None
            output_type = None
            if self.get_value_info(lhs) != None:
                lhs_type = self.get_value_info(lhs).type.tensor_type.elem_type
            if self.get_value_info(onnx_node.name) != None:
                output_type = self.get_value_info(onnx_node.name).type.tensor_type.elem_type
            need_floor = (output_type in [onnx.TensorProto.INT32, onnx.TensorProto.INT64]) \
                        or (lhs_type in [onnx.TensorProto.INT32, onnx.TensorProto.INT64])

            if self.is_dynamic() and need_floor:
                new_op=top.DivConstOp(self.unranked_type,
                                      lhs_op,
                                      const_val=self.getScalar(rhs),
                                      loc=self.get_loc(name),
                                      ip=self.mlir.insert_point).output
            else:
                new_op = top.MulConstOp(self.unranked_type,
                                        lhs_op,
                                        const_val=1 / self.getScalar(rhs),
                                        loc=self.get_loc(name),
                                        ip=self.mlir.insert_point).output
                if (need_floor):
                    new_op = top.FloorOp(self.unranked_type,
                                        new_op,
                                        loc=self.get_loc(name + '_floor'),
                                        ip=self.mlir.insert_point).output
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = top.DivOp(self.unranked_type, [lhs_op, rhs_op],
                               loc=self.get_loc(name),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_reciprocal_op(self, onnx_node):
        assert (onnx_node.op_type == "Reciprocal")
        assert len(onnx_node.inputs) == 1
        op0 = self.getOperand(onnx_node.inputs[0])
        div_op = top.ReciprocalOp(self.unranked_type,
                                  op0,
                                  const_val=1,
                                  loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                  onnx_node.op_type)),
                                  ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, div_op)

    def convert_squeeze_op(self, onnx_node):
        assert (onnx_node.op_type == "Squeeze")
        op = self.getOperand(onnx_node.inputs[0])
        axes = []
        if 'axes' in onnx_node.attrs or len(onnx_node.inputs) > 1:
            if self.opset < 13:
                axes = onnx_node.attrs.get('axes')
            else:
                if len(onnx_node.inputs) != 1:
                    axes = self.getWeight(onnx_node.inputs[1]).astype(int)
        new_op = top.SqueezeOp(self.unranked_type,
                               op,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point,
                               axes=axes).output
        self.addOperand(onnx_node.name, new_op)

    def convert_unsqueeze_op(self, onnx_node):
        assert (onnx_node.op_type == "Unsqueeze")
        if self.isWeight(onnx_node.inputs[0]):
            op = self.getWeightOp(onnx_node.inputs[0])
        else:
            op = self.getOperand(onnx_node.inputs[0])
        if self.opset < 13:
            axes = onnx_node.attrs.get('axes')
        else:
            if len(onnx_node.inputs) == 1:
                axes = []
            else:
                axes = self.getWeight(onnx_node.inputs[1]).astype(int)
        new_op = top.UnsqueezeOp(self.unranked_type,
                                 op,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point,
                                 axes=axes).output
        self.addOperand(onnx_node.name, new_op)

    def convert_clip_op(self, onnx_node):
        assert (onnx_node.op_type == "Clip")
        input = self.getOperand(onnx_node.inputs[0])
        if len(onnx_node.inputs) == 3:
            try:
                min = self.getWeight(onnx_node.inputs[1])
            except:
                min = onnx_node.attrs.get('min', -np.inf)
            try:
                max = self.getWeight(onnx_node.inputs[2])
            except:
                max = onnx_node.attrs.get('max', np.inf)
        else:
            min = onnx_node.attrs.get('min', -np.inf)
            max = onnx_node.attrs.get('max', np.inf)
        if min == 0.0 and max > min:
            new_op = top.ReluOp(self.unranked_type,
                                input,
                                relu_limit=max if max != np.inf else 0.0,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        else:
            new_op = top.ClipOp(self.unranked_type,
                                input,
                                min=min,
                                max=max,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_transpose_op(self, onnx_node):
        assert (onnx_node.op_type == "ConvTranspose")
        kernel_shape = onnx_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get('dilations', dim * [1])
        group = onnx_node.attrs.get('group', 1)
        strides = onnx_node.attrs.get('strides', dim * [1])
        pads = onnx_node.attrs.get('pads', dim * 2 * [0])
        output_padding = onnx_node.attrs.get('output_padding', dim * [0])

        operands = list()
        input_opd = self.getOperand(onnx_node.inputs[0])
        weight_name = onnx_node.inputs[1]
        # weight can be dynamic
        if weight_name in self.tensors:
            old_weight = np.ascontiguousarray(self.tensors[weight_name])
            if weight_name not in self.mlir.load_weight:
                if group != 1:
                    # (ic, oc / g, kh, kw) --> (g, oc/g, ic / g, kh, kw) --> (oc / g, ic, kh, kw)
                    _shape = list(old_weight.shape)
                    old_shape = [group, int(_shape[0] / group), _shape[1]] + _shape[2:]
                    new_shape = [_shape[1], _shape[0]] + _shape[2:]
                    old_weight = old_weight.reshape(old_shape)
                    order = [0, 2, 1] + list(range(len(_shape) + 1)[3:])
                    new_weight = np.transpose(old_weight, order).reshape(new_shape)
                    self.tensors[weight_name] = new_weight
                else:
                    # (ic, oc, kh, kw) --> (oc, ic, kh, kw)
                    order = [1, 0] + list(range(len(old_weight.shape))[2:])
                    self.tensors[weight_name] = np.transpose(old_weight, order)

                self.shapes[weight_name] = self.tensors[weight_name].shape

        filter_opd = self.getOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            bias_opd = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_opd = self.mlir.none_op
        operands.append(input_opd)
        operands.append(filter_opd)
        operands.append(bias_opd)

        new_op = top.DeconvOp(self.unranked_type,
                              *operands,
                              kernel_shape=kernel_shape,
                              strides=strides,
                              dilations=dilations,
                              pads=pads,
                              output_padding=output_padding,
                              group=group,
                              do_relu=False,
                              loc=self.get_loc('{}_{}'.format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_split_op(self, onnx_node):
        assert (onnx_node.op_type == "Split")
        op = self.getOperand(onnx_node.inputs[0])
        num_output = len(onnx_node.outputs)
        axis = onnx_node.attrs['axis']
        split_size = onnx_node.attrs.get('split', None)
        if len(onnx_node.inputs) > 1:
            split_size = self.getWeight(onnx_node.inputs[1]).astype(int)

        loc_names = [n +  "_" + onnx_node.op_type for n in onnx_node.outputs]
        new_op = top.SplitOp([self.unranked_type] * num_output,
                             op,
                             axis=axis,
                             num=num_output,
                             split_size=split_size,
                             loc=self.get_loc(loc_names),
                             ip=self.mlir.insert_point).outputs
        for i in range(num_output):
            self.addOperand(onnx_node.outputs[i], new_op[i])

    # support max ndims to 6
    def convert_reduce_op(self, onnx_node):
        assert (onnx_node.op_type in [
            "ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1",
            "ReduceSum"
        ])
        op = self.getOperand(onnx_node.inputs[0])
        axes = onnx_node.attrs.get('axes', list()) \
            if len(onnx_node.inputs) == 1 else self.getWeight(onnx_node.inputs[1])
        axes = copy.deepcopy(axes)  #if change it, should do deepcopy
        keepdims = onnx_node.attrs.get('keepdims', 1) != 0
        axes.sort()
        new_op = top.ReduceOp(self.unranked_type,
                              op,
                              axes=axes,
                              keepdims=keepdims,
                              mode=StringAttr.get(onnx_node.op_type),
                              loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_arg_op(self, onnx_node):
        assert (onnx_node.op_type in ["ArgMin", "ArgMax"])
        op = self.getOperand(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        keepdims = onnx_node.attrs.get('keepdims', 1) != 0
        select_last_index = onnx_node.attrs.get('select_last_index', 0) != 0
        loc_names = []
        out_shapes = [None, None]
        out_needs = [False, False]
        for idx, out in enumerate(onnx_node.outputs):
            if len(out) > 0 and self.check_need(out):
                loc_names.append("{}_{}".format(out, onnx_node.op_type))
                out_needs[idx] = True
                out_shapes[idx] = []
        out_op = top.ArgOp(*self.mlir.get_tensor_type(out_shapes),
                           op,
                           axis=axis,
                           keepdims=keepdims,
                           mode=StringAttr.get(onnx_node.op_type),
                           select_last_index=select_last_index,
                           loc=self.get_loc(loc_names),
                           ip=self.mlir.insert_point)
        out_ops = [out_op.indices, out_op.values]
        for idx, need in enumerate(out_needs):
            if not need: continue
            self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_lrn_op(self, onnx_node):
        assert onnx_node.op_type == "LRN"
        op = self.getOperand(onnx_node.inputs[0])

        size = onnx_node.attrs.get("size")
        alpha = onnx_node.attrs.get("alpha", None)
        beta = onnx_node.attrs.get("beta", None)
        bias = onnx_node.attrs.get("bias", None)
        new_op = top.LRNOp(self.unranked_type,
                           op,
                           size=size,
                           alpha=alpha,
                           beta=beta,
                           bias=bias,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gru_op(self, onnx_node):
        assert (onnx_node.op_type == "GRU")
        direction = onnx_node.attrs.get("direction", 'forward')
        layout = onnx_node.attrs.get("layout", 0)
        hidden_size = onnx_node.attrs.get("hidden_size")
        batch_first = True if layout == 1 else False
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))  # in
        operands.append(self.getWeightOp(onnx_node.inputs[1]))  # W
        operands.append(self.getWeightOp(onnx_node.inputs[2]))  # R
        num_inputs = len(onnx_node.inputs)
        bias_op, init_h_op = self.mlir.none_op, self.mlir.none_op
        if num_inputs > 3:
            bias_op = self.getWeightOp(onnx_node.inputs[3])
        if num_inputs > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("GRU does not test the case of specify the sequence_lens.")
        if num_inputs > 5 and len(onnx_node.inputs[5]) != 0:
            init_h_op = self.getOp(onnx_node.inputs[5])
        operands.extend([bias_op, init_h_op])
        loc_names = [onnx_node.name + '_GRU', onnx_node.name + '_H']
        out_shapes = [None, None]
        out_needs = [False, False]
        for idx, out in enumerate(onnx_node.outputs):
            if len(out) > 0 and self.check_need(out):
                loc_names[idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = []

        out_op = top.GRUOp(*self.mlir.get_tensor_type(out_shapes),
                           *operands,
                           hidden_size=hidden_size,
                           bidirectional=direction == b'bidirectional',
                           batch_first=batch_first,
                           loc=self.get_loc(loc_names),
                           ip=self.mlir.insert_point)
        out_ops = [out_op.Y, out_op.Y_h]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_lstm_op(self, onnx_node):
        assert (onnx_node.op_type == "LSTM")
        direction = onnx_node.attrs.get("direction", 'forward')
        layout = onnx_node.attrs.get("layout", 0)
        hidden_size = onnx_node.attrs.get("hidden_size")
        batch_first = True if layout == 1 else False
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))  # in
        operands.append(self.getWeightOp(onnx_node.inputs[1]))  # W
        operands.append(self.getWeightOp(onnx_node.inputs[2]))  # R
        num_inputs = len(onnx_node.inputs)
        bias_op, init_h_op, init_c_op = self.mlir.none_op, self.mlir.none_op, self.mlir.none_op
        if num_inputs > 3 and len(onnx_node.inputs[3]) != 0:
            bias_op = self.getWeightOp(onnx_node.inputs[3])
        if num_inputs > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("LSTM does not test the case of specify the sequence_lens.")
        if num_inputs > 5 and len(onnx_node.inputs[5]) != 0:
            init_h_op = self.getOp(onnx_node.inputs[5])
        if num_inputs > 6 and len(onnx_node.inputs[5]) != 0:
            init_c_op = self.getOp(onnx_node.inputs[6])
        operands.extend([bias_op, init_h_op, init_c_op])
        loc_names = [onnx_node.name + '_LSTM', onnx_node.name + '_H', onnx_node.name + '_C']
        operands.append(self.mlir.none_op)
        out_shapes = [None, None, None]
        out_needs = [False, False, False]
        for idx, out in enumerate(onnx_node.outputs):
            if len(out) > 0 and self.check_need(out):
                loc_names[idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = []
        out_op = top.LSTMOp(*self.mlir.get_tensor_type(out_shapes),
                            *operands,
                            hidden_size=hidden_size,
                            bidirectional=direction == b'bidirectional',
                            batch_first=batch_first,
                            loc=self.get_loc(loc_names),
                            ip=self.mlir.insert_point)
        out_ops = [out_op.Y, out_op.Y_h, out_op.Y_c]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_gather_op(self, onnx_node):
        assert (onnx_node.op_type == "Gather")
        in0 = self.getOp(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        extra_attr = {}
        if self.isScalar(onnx_node.inputs[1]):
            extra_attr.update({"keepdims": True})
            idx = self.find_named_tensor(onnx_node.inputs[1])
            if idx is not None and len(idx.shape) == 0:
                extra_attr["keepdims"] = False
        elif onnx_node.shape_info['input'][onnx_node.inputs[1]] is not None \
            and not onnx_node.shape_info['input'][onnx_node.inputs[1]]:
            extra_attr.update({"keepdims": False})
        indices = self.getOp(onnx_node.inputs[1])
        new_op = top.GatherOp(self.unranked_type,
                              in0,
                              indices,
                              axis=axis,
                              **extra_attr,
                              loc=self.get_loc(name),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gather_elements_op(self, onnx_node):
        assert (onnx_node.op_type == "GatherElements")
        in0 = self.getOp(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        indices = self.getOp(onnx_node.inputs[1])
        new_op = top.GatherElementsOp(self.unranked_type,
                                      in0,
                                      indices,
                                      axis=axis,
                                      loc=self.get_loc(name),
                                      ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gathernd_op(self, onnx_node):
        assert (onnx_node.op_type == "GatherND")
        input = self.getOp(onnx_node.inputs[0])
        indices = self.getOp(onnx_node.inputs[1])
        batch_dims = onnx_node.attrs.get('batch_dims', 0)
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        new_op = top.GatherNDOp(self.unranked_type,
                                input,
                                indices,
                                batch_dims=batch_dims,
                                loc=self.get_loc(name),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_expand_op(self, onnx_node):
        assert (onnx_node.op_type == 'Expand')
        in0 = self.getOp(onnx_node.inputs[0])
        if self.isWeight(onnx_node.inputs[1]):
            shape = self.getWeight(onnx_node.inputs[1])
            new_op = top.ExpandOp(self.unranked_type,
                                   in0,
                                   shape=shape,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        else:
            shape = self.getOperand(onnx_node.inputs[1])
            new_op = top.ExpandOp(self.unranked_type,
                                in0,
                                shapeT=shape,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)
        return

    def convert_tile_op(self, onnx_node):
        assert (onnx_node.op_type == "Tile")
        in0_op = self.getOp(onnx_node.inputs[0])
        if self.isWeight(onnx_node.inputs[1]):
            tile_data = self.getWeight(onnx_node.inputs[1])
            if np.prod(tile_data) == 1:
                self.addOperand(onnx_node.name, in0_op)
                return
            else:
                new_op = top.TileOp(self.unranked_type,
                        in0_op,
                        tile = tile_data,
                        loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                        ip=self.mlir.insert_point).output
        else:
            tile_op = self.getOperand(onnx_node.inputs[1])
            new_op = top.TileOp(self.unranked_type,
                        in0_op,
                        tileT = tile_op,
                        loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                        ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_topk_op(self, onnx_node):
        assert (onnx_node.op_type == "TopK")
        in_op = self.getOperand(onnx_node.inputs[0])
        K = onnx_node.attrs.get('k', -1) # opset 10
        k_op = None
        if (len(onnx_node.inputs) > 1):
            if self.isWeight(onnx_node.inputs[1]):
                K = self.getScalar(onnx_node.inputs[1])
            else:
                k_op = self.getOperand(onnx_node.inputs[1])
        axis = onnx_node.attrs.get('axis', -1)
        largest = onnx_node.attrs.get('largest', True)
        sorted = onnx_node.attrs.get('sorted', True)
        loc_names = [onnx_node.name + '_TopK_indices', onnx_node.name + "_TopK_values"]
        out_shapes = [None, None]
        out_needs = [False, False]
        for idx, out in enumerate(onnx_node.outputs):
            #topk at the hw need two output
            if len(out) > 0:
                loc_names[idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = []
        out_op = top.TopKOp(*self.mlir.get_tensor_type(out_shapes),
                            in_op,
                            axis=axis,
                            K=K,
                            kT=k_op,
                            largest=largest,
                            sorted=sorted,
                            loc=self.get_loc(loc_names),
                            ip=self.mlir.insert_point)
        out_ops = [out_op.values, out_op.indices]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_max_op(self, onnx_node):
        assert (onnx_node.op_type == 'Max')
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        lhs_op = self.getWeightOp(lhs) if self.isWeight(lhs) else self.getOp(lhs)
        rhs_op = self.getWeightOp(rhs) if self.isWeight(rhs) else self.getOp(rhs)
        new_op = top.MaxOp(self.unranked_type, [lhs_op, rhs_op],
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_trilu_op(self, onnx_node):
        assert (onnx_node.op_type == 'Trilu')
        upper = onnx_node.attrs.get('upper', 1)
        diagonal = 0
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.TriluOp(self.unranked_type,
                           operand,
                           upper,
                           diagonal,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_min_op(self, onnx_node):
        assert (onnx_node.op_type == "Min")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        lhs_op = self.getWeightOp(lhs) if self.isWeight(lhs) else self.getOp(lhs)
        rhs_op = self.getWeightOp(rhs) if self.isWeight(rhs) else self.getOp(rhs)
        new_op = top.MinOp(self.unranked_type, [lhs_op, rhs_op],
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_mod_op(self, onnx_node):
        assert (onnx_node.op_type == "Mod")
        assert (len(onnx_node.inputs) == 2)
        inp1 = onnx_node.inputs[0]
        inp2 = onnx_node.inputs[1]
        inp1_op = self.getOp(inp1)
        inp2_op = self.getOp(inp2)
        new_op = top.ModOp(self.unranked_type,
                           [inp1_op, inp2_op],
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_abs_op(self, onnx_node):
        assert (onnx_node.op_type == "Abs")
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.AbsOp(self.unranked_type,
                           operand,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_neg_op(self, onnx_node):
        assert (onnx_node.op_type == "Neg")
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        operand = self.getOperand(onnx_node.inputs[0])
        mul_const_op = top.MulConstOp(self.unranked_type,
                                      operand,
                                      const_val=-1.0,
                                      loc=self.get_loc(name),
                                      ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, mul_const_op)

    def convert_nms_op(self, onnx_node):
        assert (onnx_node.op_type == "NonMaxSuppression")
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        operands = []
        optional_weight_name = ['max_output_boxes_per_class', 'iou_threshold', 'score_threshold']
        # for i, x in enumerate(onnx_node.inputs):
        inputs_len = len(onnx_node.inputs)
        assert(inputs_len >= 2 and inputs_len <= 5)
        for i in range(5):
            if i < inputs_len:
                x = onnx_node.inputs[i]
                if self.isWeight(x):
                    data = self.getWeight(x)
                    # self.addWeight(x, data)
                    operands.append(self.getWeightOp(x))
                    if i == 2:
                        # not strictly equal to 2**63 -1, (case:9.223372e+18) still can be cast to negative because of overflow
                        if(data > 2**63 -1000):
                            max_output_size = 10000
                        else:
                            max_output_size = data.astype(np.int64)
                else:
                    operands.append(self.getOperand(x))
            else:
                w_name = "{}_{}_default".format(name, optional_weight_name[i - 2])
                wtype = np.float32 if i > 2 else np.int64
                self.addWeight(w_name, np.array([0], dtype=wtype))
                operands.append(self.getWeightOp(w_name))

        max_output_size = 0
        if (len(onnx_node.inputs) > 2):
            if self.isWeight(onnx_node.inputs[2]):
                if(self.getWeight(onnx_node.inputs[2]) > 2**63 -1000):
                    max_output_size = 10000
                else:
                    max_output_size = self.getWeight(onnx_node.inputs[2]).astype(np.int64)
        nms_op = top.NmsOp(self.unranked_type,
                           operands,
                           center_point_box=0,
                           max_output_size=max_output_size,
                           loc=self.get_loc(name),
                           ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, nms_op)

    def convert_prelu_op(self, onnx_node):
        assert (onnx_node.op_type == "PRelu")
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        in_op = self.getOperand(lhs)
        if self.isScalar(rhs):
            new_op = top.LeakyReluOp(self.unranked_type,
                                     in_op,
                                     alpha=self.getScalar(rhs),
                                     loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                     onnx_node.op_type)),
                                     ip=self.mlir.insert_point).output
            self.addOperand(onnx_node.name, new_op)
            return
        slope = self.getOp(rhs)
        prelu_op = top.PReluOp(self.unranked_type,
                               in_op,
                               slope,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, prelu_op)

    def convert_sum_op(self, onnx_node):
        assert (onnx_node.op_type == "Sum")
        opd0 = self.getOperand(onnx_node.inputs[0])
        num_inputs = len(onnx_node.inputs)
        for i in range(1, num_inputs):
            opd1 = self.getOperand(onnx_node.inputs[i])
            last_name = onnx_node.name
            if i != num_inputs - 1:
                last_name += "_{}".format(str(i))
            opd0 = top.AddOp(self.unranked_type, [opd0, opd1],
                             do_relu=False,
                             loc=self.get_loc("{}_{}".format(last_name, onnx_node.op_type)),
                             ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, opd0)

    def convert_sqrt_op(self, onnx_node):
        assert (onnx_node.op_type == "Sqrt")
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.SqrtOp(self.unranked_type,
                            operand,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_tanh_op(self, onnx_node):
        assert (onnx_node.op_type == "Tanh")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.TanhOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_arctan_op(self, onnx_node):
        assert (onnx_node.op_type == "Atan")
        # arctan(x) = aign(x) * acos(1 / (sqrt(1 + mul(abs(x), abs(x)))))
        op0 = self.getOperand(onnx_node.inputs[0])
        op_name = onnx_node.name + "_sign"
        sign_op = top.SignOp(self.unranked_type,
                                op0,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output
        op_name = onnx_node.name + "_abs"
        abs_op = top.AbsOp(self.unranked_type,
                                op0,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output
        op_name = onnx_node.name + "_mul"
        mul_op = top.MulOp(self.unranked_type, [abs_op, abs_op],
                           do_relu=False,
                           loc=self.get_loc(op_name),
                           ip=self.mlir.insert_point).output

        op_name = onnx_node.name + "_ml_mulscale"
        add_op = top.AddConstOp(self.unranked_type,
                                mul_op,
                                const_val=1,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output

        op_name = onnx_node.name + "_sqrt"
        sqrt_op = top.SqrtOp(self.unranked_type,
                             add_op,
                             loc=self.get_loc(op_name),
                             ip=self.mlir.insert_point).output

        op_name = onnx_node.name + "_reciprocal"
        reciprocal_op = top.ReciprocalOp(self.unranked_type,
                                         sqrt_op,
                                         loc=self.get_loc(op_name),
                                         ip=self.mlir.insert_point).output

        op_name = onnx_node.name + "_arccos"
        arccos_op = top.ArccosOp(self.unranked_type,
                                 reciprocal_op,
                                 loc=self.get_loc(op_name),
                                 ip=self.mlir.insert_point).output
        arctan_op = top.MulOp(self.unranked_type, [sign_op, arccos_op],
                           do_relu=False,
                           loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                           ip=self.mlir.insert_point).output

        self.addOperand(onnx_node.name, arctan_op)

    def convert_arctanh_op(self, onnx_node):
        assert (onnx_node.op_type == "Atanh")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.ArctanhOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_arccos_op(self, onnx_node):
        assert (onnx_node.op_type == "Acos")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = top.ArccosOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_pow_op(self, onnx_node):
        assert (onnx_node.op_type == "Pow")
        assert (len(onnx_node.inputs) == 2)
        base = onnx_node.inputs[0]
        expn = onnx_node.inputs[1]
        if self.isScalar(expn):
            base_op = self.getOp(base)
            expn_const = self.getScalar(expn)
            if expn_const == 1.0:
                self.addOperand(onnx_node.name, base_op)
                return
            if expn_const == 2.0:
                mul_op = top.MulOp(self.unranked_type, [base_op, base_op],
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
                self.addOperand(onnx_node.name, mul_op)
                return
            else:
                pow_op = top.PowOp(self.unranked_type,
                                   base_op,
                                   exponent=expn_const,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
                self.addOperand(onnx_node.name, pow_op)
        elif self.isScalar(base):
            expn_op = self.getOp(expn)
            base_const = self.getScalar(base)
            pow_op = top.Pow2Op(self.unranked_type,
                                base_const,
                                expn_op,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
            self.addOperand(onnx_node.name, pow_op)
        else:
            base_op = self.getOp(base)
            expn_op = self.getOp(expn)
            pow_op = top.Pow3Op(self.unranked_type,
                                [base_op, expn_op],
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
            self.addOperand(onnx_node.name, pow_op)


    def convert_where_op(self, onnx_node):
        assert (onnx_node.op_type == "Where")
        assert (len(onnx_node.inputs) == 3)
        cond = onnx_node.inputs[0]
        tbrn = onnx_node.inputs[1]
        fbrn = onnx_node.inputs[2]
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
        if num_const == 0:
            new_op = top.WhereOp(self.unranked_type,
                                 cond_opd,
                                 tbrn_opd,
                                 fbrn_opd,
                                 x_is_const=False,
                                 y_is_const=False,
                                 x_const_val=0,
                                 y_const_val=0,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        elif num_const >= 1:
            x_is_const = False
            y_is_const = False
            if self.isScalar(tbrn):
                x_is_const = True
                x_const_val = self.getScalar(tbrn)
                t_opd = self.mlir.none_op
            else:
                t_opd = tbrn_opd
                x_const_val = 0
            if self.isScalar(fbrn):
                y_is_const = True
                y_const_val = self.getScalar(fbrn)
                f_opd = self.mlir.none_op
            else:
                f_opd = fbrn_opd
                y_const_val = 0
            new_op = top.WhereOp(self.unranked_type,
                                 cond_opd,
                                 t_opd,
                                 f_opd,
                                 x_is_const=x_is_const,
                                 y_is_const=y_is_const,
                                 x_const_val=x_const_val,
                                 y_const_val=y_const_val,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        else:
            assert (0)
        self.addOperand(onnx_node.name, new_op)

    def convert_not_op(self, onnx_node):
        assert (onnx_node.op_type == "Not")
        opd = onnx_node.inputs[0]
        not_op = top.CompareConstOp(self.unranked_type,
                                    self.getOp(opd),
                                    mode=StringAttr.get(onnx_node.op_type),
                                    const_val=np.array([0]).astype(np.bool_),
                                    inversed=False,
                                    loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                    onnx_node.op_type)),
                                    ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, not_op)

    def convert_cmp_op(self, onnx_node):
        supports = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Xor"}
        assert (onnx_node.op_type in supports)
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isScalar(lhs) and len(self.getShape(lhs)) == 1:
            rhs_opd = self.getOp(rhs)
            cmp_op = top.CompareConstOp(self.unranked_type,
                                        rhs_opd,
                                        mode=StringAttr.get(onnx_node.op_type),
                                        const_val=self.getScalar(lhs),
                                        inversed=True,
                                        loc=self.get_loc("{}_{}".format(
                                            onnx_node.name, onnx_node.op_type)),
                                        ip=self.mlir.insert_point).output
        elif self.isScalar(rhs) and len(self.getShape(rhs)) == 1:
            lhs_opd = self.getOp(lhs)
            cmp_op = top.CompareConstOp(self.unranked_type,
                                        lhs_opd,
                                        mode=StringAttr.get(onnx_node.op_type),
                                        const_val=self.getScalar(rhs),
                                        inversed=False,
                                        loc=self.get_loc("{}_{}".format(
                                            onnx_node.name, onnx_node.op_type)),
                                        ip=self.mlir.insert_point).output
        else:
            rhs_opd = self.getOp(rhs)
            lhs_opd = self.getOp(lhs)
            cmp_op = top.CompareOp(self.unranked_type,
                                   lhs_opd,
                                   rhs_opd,
                                   mode=StringAttr.get(onnx_node.op_type),
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, cmp_op)

    def convert_hsigmoid_op(self, onnx_node):
        # hardsigmoid(x; alpha, beta) := min(max(alpha*x + beta, 0), 1)
        assert (onnx_node.op_type == "HardSigmoid")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        alpha = onnx_node.attrs.get("alpha", 1. / 6)
        beta = onnx_node.attrs.get("beta", 0.5)
        new_op = top.HardSigmoidOp(self.unranked_type,
                                   operand,
                                   alpha=alpha,
                                   beta=beta,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_hswish_op(self, onnx_node):
        # hardswish(x) := x * hardsigmoid(x; 1/6, 0.5)
        assert (onnx_node.op_type == "HardSwish")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.HardSwishOp(self.unranked_type,
                                 operand,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gelu_op(self, onnx_node):
        # 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
        assert (onnx_node.op_type == "GELU")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.GELUOp(self.unranked_type,
                            operand,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_qlinear_op(self, onnx_node):
        assert (onnx_node.op_type == "QuantizeLinear")
        assert (len(onnx_node.inputs) == 3)
        operand = self.getOperand(onnx_node.inputs[0])
        y_scale = self.getWeight(onnx_node.inputs[1]).tolist()
        y_zero_point = self.getWeight(onnx_node.inputs[2]).tolist()
        if hasattr(onnx_node, 'attrs'):
            axis = onnx_node.attrs.get('axis', None)

        new_op = top.QuantizeLinearOp(self.unranked_type,
                                      operand,
                                      y_scale=y_scale,
                                      y_zero_point=y_zero_point,
                                      axis=axis,
                                      loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                      onnx_node.op_type)),
                                      ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_deqlinear_op(self, onnx_node):
        assert (onnx_node.op_type == "DequantizeLinear")
        assert (len(onnx_node.inputs) == 3)
        try:
            operand = self.getOperand(onnx_node.inputs[0])
        except:
            operand = self.getWeightOp(onnx_node.inputs[0])
        x_scale = self.getWeight(onnx_node.inputs[1])
        x_zero_point = self.getWeight(onnx_node.inputs[2])
        if hasattr(onnx_node, 'attrs'):
            axis = onnx_node.attrs.get('axis', None)
        new_op = top.DequantizeLinearOp(self.unranked_type,
                                        operand,
                                        x_scale=x_scale,
                                        x_zero_point=x_zero_point,
                                        axis=axis,
                                        loc=self.get_loc("{}_{}".format(
                                            onnx_node.name, onnx_node.op_type)),
                                        ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_layer_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "LayerNormalization")
        assert (len(onnx_node.inputs) <= 3)
        axis = onnx_node.attrs.get("axis", -1)
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        if type(eps) == list and len(eps) == 1:
            eps = eps[0]
        input_opd = self.getOp(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not self.isScalar_(onnx_node.inputs[1], 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            if not self.isScalar_(onnx_node.inputs[2], 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2])
        out_op = top.LayerNormOp(self.unranked_type,
                                 input_opd,
                                 scale_opd,
                                 bias_opd,
                                 normalized_shape=[],
                                 axis=axis,
                                 eps=eps,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, out_op)

    def convert_pixel_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "PixelNormalization")
        assert (len(onnx_node.inputs) in (1, 2, 3))
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not self.isScalar_(onnx_node.inputs[1], 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            if not self.isScalar_(onnx_node.inputs[2], 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2])
        new_op = top.PixelNormOp(self.unranked_type,
                                 input_opd,
                                 scale_opd,
                                 bias_opd,
                                 eps=eps,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_instance_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "InstanceNormalization")
        assert (len(onnx_node.inputs) in (1, 2, 3))
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not np.all(self.getWeight(onnx_node.inputs[1]) == 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            if not np.all(self.getWeight(onnx_node.inputs[2]) == 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2])
        new_op = top.InstanceNormOp(self.unranked_type,
                                    input_opd,
                                    scale_opd,
                                    bias_opd,
                                    eps=eps,
                                    loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                    onnx_node.op_type)),
                                    ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_group_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "GroupNormalization")
        assert (len(onnx_node.inputs) in (1, 2, 3))
        num_groups = onnx_node.attrs.get("num_groups")  # required
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not self.isScalar_(onnx_node.inputs[1], 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            if not self.isScalar_(onnx_node.inputs[2], 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2])
        new_op = top.GroupNormOp(self.unranked_type,
                                 input_opd,
                                 scale_opd,
                                 bias_opd,
                                 num_groups=num_groups,
                                 eps=eps,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_scatter_elements_op(self, onnx_node):
        assert (onnx_node.op_type == "ScatterElements")
        assert (len(onnx_node.inputs) == 3)
        input = self.getOp(onnx_node.inputs[0])
        indices = self.getOp(onnx_node.inputs[1])
        updates = self.getOp(onnx_node.inputs[2])
        axis = onnx_node.attrs.get("axis", 0)
        reduction = onnx_node.attrs.get("reduction", None)
        assert not reduction
        new_op = top.ScatterElementsOp(
            self.unranked_type,
            input,
            indices,
            updates,
            axis=axis,
            # reduction=reduction, # ??????????? no such param
            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_scatternd_op(self, onnx_node):
        assert (onnx_node.op_type == "ScatterND")
        assert (len(onnx_node.inputs) == 3)
        input_data = self.getOp(onnx_node.inputs[0])
        indices = self.getOp(onnx_node.inputs[1])
        updates = self.getOp(onnx_node.inputs[2])
        reduction = onnx_node.attrs.get("reduction", None)
        assert not reduction
        scatternd_op = top.ScatterNDOp(
            self.unranked_type,
            input_data,
            indices,
            updates,
            # reduction=reduction, # ??????????? no such param
            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, scatternd_op)

    def convert_roi_align_op(self, onnx_node: OnnxNode):
        assert (onnx_node.op_type == "RoiAlign")
        assert (len(onnx_node.inputs) == 3)
        input = self.getOp(onnx_node.inputs[0])
        rois = self.getOp(onnx_node.inputs[1])
        batch_indices = self.getOp(onnx_node.inputs[2])
        output_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        mode = onnx_node.attrs.get("mode", "Avg")
        if  isinstance(mode, bytes):
            mode_str = str(mode,'utf-8')
            if  mode_str == "avg" or mode_str == "max":
                mode = mode_str.capitalize()
        output_height = onnx_node.attrs.get("output_height", 1)
        output_width = onnx_node.attrs.get("output_width", 1)
        sampling_ratio = onnx_node.attrs.get("sampling_ratio", 0)
        spatial_scale = onnx_node.attrs.get("spatial_scale", 1.0)
        if self.opset < 16:
            coord_transf_mode = "output_half_pixel"
        else:
            coord_transf_mode = onnx_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        align_corners = coord_transf_mode == "half_pixel"
        batch_indices_xpd = top.UnsqueezeOp(self.unranked_type,
                                            batch_indices,
                                            axes=[-1],
                                            loc=self.get_loc(output_name + "_unsqueeze"),
                                            ip=self.mlir.insert_point).output
        rois_xpd = top.ConcatOp(self.unranked_type, [batch_indices_xpd, rois],
                                axis=1,
                                loc=self.get_loc(output_name + "_concat"),
                                ip=self.mlir.insert_point).output
        new_op = top.RoiAlignOp(self.unranked_type,
                                input,
                                rois_xpd,
                                mode=StringAttr.get(mode),
                                output_height=output_height,
                                output_width=output_width,
                                sampling_ratio=sampling_ratio,
                                spatial_scale=spatial_scale,
                                align_corners=align_corners,
                                loc=self.get_loc(output_name),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_nonzero_op(self, onnx_node):
        assert (onnx_node.op_type == "NonZero")
        assert (len(onnx_node.inputs) == 1)
        input_data = self.getOp(onnx_node.inputs[0])
        new_op = top.NonZeroOp(self.unranked_type,
                               input_data,
                               order=StringAttr.get("RowMajor"),
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_onehot_op(self, onnx_node):
        assert (onnx_node.op_type == "OneHot")
        assert (len(onnx_node.inputs) == 3)
        assert (len(onnx_node.outputs) == 1)
        indices = self.getOp(onnx_node.inputs[0])
        depth = self.getOp(onnx_node.inputs[1])      #  -depth <= indeces[i] <= depth-1
        values = self.getOp(onnx_node.inputs[2])
        axis = onnx_node.attrs.get("axis",-1)
        assert (axis == -1)
        output_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        min_value = 0
        max_value = 0
        if (self.isWeight(values)):
            min_value = top.ReduceOp(self.unranked_type,
                                    self.getWeightOp(values),
                                    axes=[0],
                                    keepdims=1,
                                    mode=StringAttr.get("ReduceMin"),
                                    loc=self.get_loc(output_name + "_min"),
                                    ip=self.mlir.insert_point).output
            max_value = top.ReduceOp(self.unranked_type,
                                    self.getWeightOp(values),
                                    axes=[0],
                                    keepdims=1,
                                    mode=StringAttr.get("ReduceMax"),
                                    loc=self.get_loc(output_name + "_max"),
                                    ip=self.mlir.insert_point).output
        else:
            min_value = top.ReduceOp(self.unranked_type,
                                    values,
                                    axes=[0],
                                    keepdims=1,
                                    mode=StringAttr.get("ReduceMin"),
                                    loc=self.get_loc(output_name + "_min"),
                                    ip=self.mlir.insert_point).output
            max_value = top.ReduceOp(self.unranked_type,
                                    values,
                                    axes=[0],
                                    keepdims=1,
                                    mode=StringAttr.get("ReduceMax"),
                                    loc=self.get_loc(output_name + "_max"),
                                    ip=self.mlir.insert_point).output
        if (self.isWeight(indices)):
            ind_dims = list(self.getWeight(onnx_node.inputs[0]).shape)
            ind_unsq = list(self.getWeight(onnx_node.inputs[0]).shape)
            ind_dims.extend(self.getWeight(onnx_node.inputs[1]))
            ind_unsq.extend([1])
            padding_shape = np.array(ind_dims).astype(np.int64)
            input_data = top.ExpandOp(self.unranked_type,
                                min_value,
                                shape=padding_shape,
                                loc=self.get_loc(output_name + "_expandmin"),
                                ip=self.mlir.insert_point).output
            updates = top.ExpandOp(self.unranked_type,
                                max_value,
                                shape=ind_unsq,
                                loc=self.get_loc(output_name + "_expandmax"),
                                ip=self.mlir.insert_point).output
        else:
            depth_max = self.getWeightOp(onnx_node.inputs[1])
            depth_min = np.array([1], dtype=np.int64)
            depth_min_name = output_name + "_undates_concat_depth_min"
            self.addWeight(depth_min_name, depth_min)
            shape_op = top.ShapeOp(self.unranked_type,
                             indices,
                             start=0,
                             loc=self.get_loc(output_name + "_shape"),
                             ip=self.mlir.insert_point).output
            padding_shape = top.ConcatOp(self.unranked_type, [shape_op, depth_max],
                                axis=-1,
                                loc=self.get_loc(output_name + "_indata_concat"),
                                ip=self.mlir.insert_point).output
            ind_unsq = top.ConcatOp(self.unranked_type, [shape_op, self.getWeightOp(depth_min_name)],
                                axis=-1,
                                loc=self.get_loc(output_name + "_undates_concat"),
                                ip=self.mlir.insert_point).output
            input_data = top.ExpandOp(self.unranked_type,
                                min_value,
                                shapeT=padding_shape,
                                loc=self.get_loc(output_name + "_expandmin"),
                                ip=self.mlir.insert_point).output
            updates = top.ExpandOp(self.unranked_type,
                                max_value,
                                shapeT=ind_unsq,
                                loc=self.get_loc(output_name + "_expandmax"),
                                ip=self.mlir.insert_point).output
        indices_unsq = top.UnsqueezeOp(self.unranked_type,
                                indices,
                                loc=self.get_loc(output_name + "_indices_unsqeeze"),
                                ip=self.mlir.insert_point,
                                axes=[-1]).output
        new_op = top.ScatterElementsOp(self.unranked_type,
                                input_data,
                                indices_unsq,
                                updates,
                                axis=-1,
                                loc=self.get_loc(output_name + "_scatter_elements"),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def parse_subgraph(self, op, region_idx, graph_node):
        converted_nodes = list()
        for n in graph_node.node:
            node = OnnxNode(n)
            if n.op_type in ["Gather"]:
                input_shape = dict()
                for input in n.input:
                    input_shape[input] = self.get_shape_for_node(graph_node.input, graph_node.output, graph_node.value_info, input)
                output_shape = dict()
                for output in n.output:
                    output_shape[output] = self.get_shape_for_node(graph_node.input, graph_node.output, graph_node.value_info, output)
                node.shape_info["input"] = input_shape
                node.shape_info["output"] = output_shape
            converted_nodes.append(node)

        unsupported = set()
        for n in converted_nodes:
            if n.op_type not in self.onnxop_factory:
                unsupported.add(n.op_type)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))
        initializer_names = [x.name for x in graph_node.initializer]
        subgraph_input_names = list()

        region = op.regions[region_idx]
        arg_types = list()
        #add block argument to entry block
        for input in graph_node.input:
            if input.name not in initializer_names:
                shape = self.get_shape_from_value_info_proto(input)
                #if int64/int32/bool, replace it with int32
                if input.type.tensor_type.elem_type in [
                        onnx.TensorProto.INT64, onnx.TensorProto.INT32, onnx.TensorProto.BOOL
                ]:
                    dtype = "INT32"
                else:
                    dtype = "F32"
                arg_types.append(
                    self.mlir.get_tensor_type(shape if len(shape) > 0 else [1],
                                              self.mlir.mlir_type[dtype]))
                self.input_names.append(input.name)
                subgraph_input_names.append(input.name)
        self.mlir.buildBlock(region, arg_types)
        self.mlir.reconfig_insert_point(region.blocks[0])

        entry_block_args = list()
        for i in region.blocks[0].arguments:
            entry_block_args.append(i)
        #create subgraph's input op
        for idx, input in enumerate(graph_node.input):
            if input.name not in initializer_names:
                input_op = self.mlir.create_subgraph_input_op(input.name, arg_types[idx],
                                                              entry_block_args[idx], **{})
                self.addOperand(input.name, input_op)
        # add all weight
        self.subgraph_initializer = graph_node.initializer
        for tensor in graph_node.initializer:
            name = tensor.name
            data = numpy_helper.to_array(tensor).astype(np.float32)
            self.addWeight(name, data)
        self.get_output_name(graph_node)
        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        for n in converted_nodes:
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
        self.subgraph_initializer = None

        yield_op = list()
        #remove the input tensor from self.input_names
        for n in subgraph_input_names:
            self.input_names.remove(n)

        #Todo: remove the shape/tensor from self.shapes/self.tensors
        for output in graph_node.output:
            if not self.isWeight(output.name):
                self.output_names.remove(output.name)
                op = self.getOperand(output.name)
                yield_op.append(op)
            else:
                yield_op.append(self.getWeightOp(output.name))
        self.mlir.create_yield_op(yield_op)

    def convert_if_op(self, onnx_node):
        assert (onnx_node.op_type == "If")
        assert (len(onnx_node.inputs) == 1)
        input_data = self.getOp(onnx_node.inputs[0])
        p = {
            "name": ["{}_{}_{}".format(onnx_node.name, onnx_node.op_type, id) for id in range(len(onnx_node.outputs))],
            "region": 2,
        }
        new_op = self.mlir.create_if_op([input_data], [], **p)
        self.addOperand(onnx_node.name, new_op)
        for attr in onnx_node.node_proto.attribute:
            #attr.type == 5 : graph
            region_idx = 0 if attr.name == "then_branch" else 1
            if attr.type == 5:
                self.parse_subgraph(new_op.owner, region_idx, attr.g)
        #restore the insert_point
        self.mlir.restore_insert_point()

    def convert_loop_op(self, onnx_node):
        assert (onnx_node.op_type == "Loop")
        assert (len(onnx_node.inputs) >= 2)
        assert (len(onnx_node.outputs) >= 1)
        operands = list()
        out_shapes = list()
        for input in onnx_node.inputs:
            op = self.getOp(input)
            operands.append(op)
        for output in onnx_node.outputs:
            out_shapes.append([])
        p = {
            "name": ["{}_{}_{}".format(onnx_node.name, onnx_node.op_type, id) for id in range(len(out_shapes))],
            "region": 1,
        }
        new_op = self.mlir.create_loop_op(operands, out_shapes, **p)
        for idx, output in enumerate(onnx_node.outputs):
            self.addOperand(output, new_op[idx])
        for attr in onnx_node.node_proto.attribute:
            #attr.type: Graph
            if attr.type == 5:
                self.parse_subgraph(new_op[0].owner, 0, attr.g)
        #restore the insert_point
        self.mlir.restore_insert_point()

    def convert_grid_sampler_op(self, onnx_node):
        assert (onnx_node.op_type == "GridSample")
        assert (len(onnx_node.inputs) == 2)
        input_data = self.getOp(onnx_node.inputs[0])
        grid_data = self.getOp(onnx_node.inputs[1])
        align_corners = onnx_node.attrs.get("align_corners", 0)
        mode = onnx_node.attrs.get("mode", "bilinear")
        if mode == b"bilinear":
            mode = 0
        elif mode == b"nearest":
            mode = 1
        else:
            assert ("Unsupported interpolation mode of {}.".format(mode) and 0)
        padding_mode = onnx_node.attrs.get("padding_mode", "zeros")
        if padding_mode == b"zeros":
            padding_mode = 0
        elif padding_mode == b"border":
            padding_mode = 1
        elif padding_mode == b"reflection":
            padding_mode = 2
        else:
            assert ("Unsupported padding_mode of {}.".format(padding_mode) and 0)
        new_op = top.GridSamplerOp(self.unranked_type,
                                   input_data,
                                   grid_data,
                                   mode=mode,
                                   padding_mode=padding_mode,
                                   align_corners=align_corners,
                                   loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                   onnx_node.op_type)),
                                   ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_cumsum_op(self, onnx_node):
        assert onnx_node.op_type == "CumSum"
        if not self.isWeight(onnx_node.inputs[1]):
            raise ValueError("Currently, only constant axis is supported")
        axis = self.getWeight(onnx_node.inputs[1])
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))
        operands.append(self.getWeightOp(onnx_node.inputs[1]))
        new_op = top.CumSumOp(self.unranked_type, *operands,
                            axis=axis,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_round_op(self, onnx_node):
        assert (onnx_node.op_type == "Round")
        operand = self.getOperand(onnx_node.inputs[0])
        new_op = top.RoundOp(self.unranked_type,
                            operand,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)
