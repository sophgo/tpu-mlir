from collections import Counter, defaultdict, OrderedDict
import onnx
import onnx.numpy_helper
import copy
import numpy as np
import onnxruntime as rt
from transform.OnnxOpOptionalAttrs import OnnxOpOptionalAttrGetter

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "keepdims": lambda x: bool(x),
}

optional_attr_getter = OnnxOpOptionalAttrGetter()


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


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


def dump_model(model, name="opt.onnx"):
    data = model.SerializeToString()
    with open(name, "wb") as file:
        file.write(data)


def get_node_attrs(node) -> dict:
    attrs = dict((attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
                 for attr in node.attribute)
    attrs_full = optional_attr_getter.get(node.op_type)
    for k, v in attrs_full.items():
        if k not in attrs:
            attrs[k] = v
    return attrs


class ConstantFolding(object):
    def __init__(self, model, test_input, dynamic_shape_input_names):
        self.test_input = test_input
        self.model = copy.deepcopy(model)
        self.dynamic_shape_input_names = dynamic_shape_input_names
        if self.model.graph.value_info:
            n = len(self.model.graph.value_info)
            for _ in range(n):
                v = self.model.graph.value_info[0]
                self.model.graph.value_info.remove(v)
        try:
            onnx.checker.check_model(self.model)
        except:
            print("WARNING: onnx model check failed")
        self.const_tensors = []

    def get_inputs(self):
        initializer_names = [x.name for x in self.model.graph.initializer]
        return [ipt for ipt in self.model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self):
        input_names = [ipt.name for ipt in self.get_inputs()]
        return input_names

    def generate_specific_rand_input(self, input_shapes):
        inputs = {}
        for key, shape in input_shapes.items():
            if not np.all(np.array(shape) > 0):
                raise RuntimeError("The shape of input '{}' has dynamic size '{}', "
                                   "please determine the input size when export "
                                   "onnx".format(key, shape))
            elem_type = self.get_elem_type(key)
            elem_type = self.get_np_type_from_elem_type(elem_type)
            if elem_type == np.bool_ :  # for mask
                inputs.update({key: np.random.randint(0, 2, shape, dtype=elem_type)})
            elif elem_type == np.int64:
                inputs.update({key: np.random.randint(1, 10, size=shape, dtype=elem_type)})
            elif len(shape) == 0: # for idx
                inputs.update({key: np.array(0, dtype=elem_type)})
            else:
                inputs.update({key: np.random.rand(*shape).astype(elem_type)})
        return inputs

    def get_value_info_all(self, name):
        for v in self.model.graph.value_info:
            if v.name == name:
                return v
        for v in self.model.graph.input:
            if v.name == name:
                return v
        for v in self.model.graph.output:
            if v.name == name:
                return v
        return None

    @staticmethod
    def insert_elem(nodes, idx, element):
        nodes.extend([nodes[-1]])
        for i in reversed(range(idx + 1, len(nodes) - 1)):
            nodes[i].CopyFrom(nodes[i - 1])
        nodes[idx].CopyFrom(element)

    @staticmethod
    def get_shape_from_value_info_proto(vinfo):
        return [dim.dim_value for dim in vinfo.type.tensor_type.shape.dim]

    @staticmethod
    def get_np_type_from_elem_type(elem_type):
        types = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32,
                np.int64, str, np.bool_, np.float16, np.double, np.uint32, np.uint64,
                np.complex64, np.complex128, np.float16)
        assert len(types) == 17
        _type = types[elem_type]
        assert _type is not None
        return _type

    def get_shape(self, name):
        vinfo = self.get_value_info_all(name)
        if vinfo is None:
            raise RuntimeError("Can't get shape of '{}'".format(name))
        return self.get_shape_from_value_info_proto(vinfo)

    def get_elem_type(self, name):
        vinfo = self.get_value_info_all(name)
        if vinfo is None:
            raise RuntimeError("Can't get dtype of '{}'".format(name))
        return vinfo.type.tensor_type.elem_type

    def is_dynamic(self, node):
        # for i in node.input:
        #     if i in self.dynamic_shape_input_names:
        #         return True
        if node.op_type in ["NonMaxSuppression", "NonZero", "Unique"] \
                and node.input[0] not in self.const_tensors:
            return True
        if node.op_type in ["Reshape", "Expand", "Upsample", "ConstantOfShape"] \
                and len(node.input) > 1 and node.input[1] not in self.const_tensors:
            return True
        if node.op_type in ["Resize"] \
                and ((len(node.input) > 2 and node.input[2] not in self.const_tensors) \
                    or (len(node.input) > 3 and node.input[3] not in self.const_tensors)):
            return True
        if node.op_type in ["Slice"] \
                and ((len(node.input) > 1 and node.input[1] not in self.const_tensors) \
                    or (len(node.input) > 2 and node.input[2] not in self.const_tensors) \
                    or (len(node.input) > 3 and node.input[3] not in self.const_tensors) \
                    or (len(node.input) > 4 and node.input[4] not in self.const_tensors)):
            return True
        return False

    def has_subgraph_in_node(self, node):
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return True
        return False

    def is_quantizeLinear(self, node):
        return node.op_type in ["DequantizeLinear", "QuantizeLinear"]

    def is_non_determinstic_node(self, node):
        return node.op_type in ["RandomNormal", "RandomNormalLike", "RandomUniformLike"]

    def get_constant_nodes(self):
        const_nodes = []
        dynamic_tensors = []
        dynamic_tensors.extend(self.dynamic_shape_input_names)
        self.const_tensors = [x.name for x in self.model.graph.initializer]
        self.const_tensors.extend([node.output[0] for node in self.model.graph.node if node.op_type == "Constant"])
        self.const_tensors.extend([''])
        for node in self.model.graph.node:
            if node.op_type == "Shape" and node.input[0] not in dynamic_tensors:
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
            elif node.op_type == "Resize" and all([x in self.const_tensors for x in node.input]):
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
            elif any(x in dynamic_tensors for x in node.input):
                dynamic_tensors.extend(node.output)
            elif self.is_dynamic(node):
                dynamic_tensors.extend(node.output)
            elif self.is_quantizeLinear(node):
                pass
            elif self.has_subgraph_in_node(node):
                if all([x in self.const_tensors for x in node.input]):
                    if (node.op_type == "If"):
                        const_nodes.append(node)
            elif len(node.input) > 0 and all([x in self.const_tensors for x in node.input]) \
                    and not self.is_non_determinstic_node(node):
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
            elif node.op_type == "Transpose" and all([x in self.const_tensors for x in node.input]):
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
        return copy.deepcopy(const_nodes)

    def forward(self, model, test_input):
        input_shapes = {}
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        try:
            try:
                sess = rt.InferenceSession(model.SerializeToString(), sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])
            except Exception as E:
                if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(E):
                    print("LOG: Try to convert through a temporary file when Constant Folding.")
                    # large models try to convert through a temporary file
                    import os
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        model_path = os.path.join(tmpdirname, 'model.onnx')
                        onnx.save(model,
                                  model_path,
                                  save_as_external_data=True,
                                  location="temp_external_data",
                                  convert_attribute=True)
                        sess = rt.InferenceSession(model_path, sess_options=sess_options,
                                                   providers=["CPUExecutionProvider"])
                else:
                    raise E
        except ValueError:
            print("WARNING: onnxruntime.InferenceSession error.")

        input_names = self.get_input_names()
        inputs = {}
        for name in input_names:
            shape = self.get_shape(name)
            input_shapes.update({name: shape})

        if len(test_input) == 1 and test_input[0].endswith('.npz'):
            inputs_npz = np.load(test_input[0])
            for name in inputs_npz.files:
                elem_type = self.get_elem_type(name)
                elem_type = self.get_np_type_from_elem_type(elem_type)
                inputs[name] = inputs_npz[name].astype(elem_type)
        else:
            inputs.update(self.generate_specific_rand_input(input_shapes))


        outputs = [x.name for x in sess.get_outputs()]
        run_options = rt.RunOptions()
        run_options.log_severity_level = 3
        return OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))

    def forward_for_node_outputs(self, const_nodes):
        model = copy.deepcopy(self.model)
        test_input = self.test_input
        for node in const_nodes:
            for output in node.output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        return self.forward(model, test_input)

    def eliminate_const_nodes(self, const_node, res):
        do_eliminate = False
        for i, node in enumerate(self.model.graph.node):
            if node in const_node:
                if node.op_type == "If":
                    sub_graph = {}
                    for attr in node.attribute:
                        sub_graph[attr.name] = attr.g.node
                    if res[node.input[0]]:
                        sub_nodes = sub_graph['then_branch']
                    else:
                        sub_nodes = sub_graph['else_branch']
                    if len(node.output) != len(sub_nodes[-1].output):
                        raise RuntimeError("If op not support multi output now, fix me.")
                    sub_nodes[-1].output[:] = []
                    sub_nodes[-1].output.extend(node.output)
                    sub_nodes = sub_nodes[::-1]
                    for n in sub_nodes:
                        self.insert_elem(self.model.graph.node, i + 1, n)
                    self.model.graph.node.remove(node)
                    do_eliminate = True
                    continue
                for output in node.output:
                    new_node = copy.deepcopy(node)
                    new_node.name = "node_" + output
                    new_node.op_type = "Constant"
                    new_attr = onnx.helper.make_attribute(
                        "value",
                        onnx.numpy_helper.from_array(res[output], name=output)
                    )
                    del new_node.input[:]
                    del new_node.attribute[:]
                    del new_node.output[:]
                    new_node.output.extend([output])
                    new_node.attribute.extend([new_attr])
                    self.insert_elem(self.model.graph.node, i + 1, new_node)
                del self.model.graph.node[i]
                do_eliminate = True
        return do_eliminate

    def remove_unused_nodes(self):
        node_inputs = []
        unused_node = []
        for n in self.model.graph.node:
            node_inputs.extend(n.input)
        node_inputs.extend([out.name for out in self.model.graph.output])
        node_inputs = set(node_inputs)

        for n in self.model.graph.node:
            if len(set(n.output).intersection(node_inputs)) == 0:
                unused_node.append(n)
        for n in unused_node:
            self.model.graph.node.remove(n)

    def infer_shapes(self):
        try:
            self.model = onnx.shape_inference.infer_shapes(self.model)
        except:
            pass
        # self.model = onnx.shape_inference.infer_shapes(self.model, strict_mode =True, data_prop=True)


    def folding(self, infer_shapes=True):
        const_nodes = self.get_constant_nodes()
        res = self.forward_for_node_outputs(const_nodes)
        const_node = [node for node in const_nodes if node.output[0] in res]
        do_eliminate = self.eliminate_const_nodes(const_node, res)
        if infer_shapes:
            self.infer_shapes()
        return do_eliminate

    def run(self):
        def fixed_point(fun):
            flag = fun()
            while True:
                if flag:
                    flag = fun()
                    continue
                break
        fixed_point(self.folding)
        self.remove_unused_nodes()
        # dump_model(self.model, "constant_opt.onnx")
        return self.model


class OuterNode(object):

    def __init__(self, is_tensor=False, tensor_value=None, attr_name=None):
        '''out of pattern chain. eg. pattern[0]'s input / tensor'''
        self.output = []  # when do input match we get name direct from onnx_node
        self.is_tensor = is_tensor  # also check if tensor with same value
        # will be checked when pattern match, will be set when insert new node in replace
        self.tensor_value = tensor_value
        self.attr_name = attr_name
        self.attr_value = None
        if is_tensor == False:
            if tensor_value is not None:
                self.tensor_value = np.array(tensor_value)
                if self.tensor_value.shape == ():
                    self.tensor_value = np.expand_dims(self.tensor_value, 0)
                self.is_tensor = True
        if attr_name:
            # for some case tensor is a part of new onnx_node's attr
            self.is_tensor = True

    def get_attr(self):
        attr_value = self.attr_value
        if len(self.attr_value.shape) == 0:
            attr_value = float(attr_value)
        return {self.attr_name: translate_onnx(self.attr_name, attr_value)}


class AttrCheck(object):

    def __init__(self, attrs: list = [], func=(lambda x: x)):
        self.attrs = attrs
        self.func = func


class AttrFunctor(object):

    def __init__(self, inputs: list = [], attrs: list = [], func=(lambda x: x)):
        assert (len(inputs) == len(attrs))
        self.inputs = inputs
        self.attrs = attrs
        self.func = func


class PatternNode(object):

    def __init__(self,
                 op_type,
                 input=[],
                 cur_attr_name=[],
                 new_attr_name=[],
                 new_attr={},
                 attrmap={},
                 constraint='',
                 attrcheck=None):
        self.op_type = op_type
        self.input = input
        self.output = []
        self.attr = {}
        # get attr form current node and renamed with new_attr_name
        self.cur_attr_name = cur_attr_name
        self.new_attr_name = new_attr_name
        # add new attr in curent node
        self.new_attr = new_attr
        self.attrmap = attrmap
        # check current node's cal manner
        self.constraint = constraint
        # check attr, should be AttrFunctor or None
        self.attrcheck = attrcheck
        assert (isinstance(self.input, list))
        assert (isinstance(self.attr, dict))
        assert (isinstance(self.cur_attr_name, list))
        assert (isinstance(self.new_attr_name, list))
        assert (isinstance(self.constraint, str))

        if cur_attr_name and len(new_attr_name) == 0:
            # if cur_attr_name and new_attr_name are all the same leave new_attr_name blank is ok
            # otherwise you should explicit assign all the key in both cur_attr_name and new_attr_name
            self.new_attr_name = cur_attr_name
        assert (len(self.cur_attr_name) == len(self.new_attr_name))

    def update(self, output, attr_value):
        # attr: from both inp / node / new, output
        self.output.clear()
        self.attr.clear()
        self.output.extend(output)
        self.attr.update(zip(self.new_attr_name, attr_value))

    def get_attr(self):
        for new_attr, attr_func in self.attrmap.items():
            args = [
                t.get_attr()[old_attr] for t, old_attr in zip(attr_func.inputs, attr_func.attrs)
            ]
            self.attr.update({new_attr: attr_func.func(*args)})
        return self.attr


class ReformInfo(object):

    def __init__(self, name: str, src_nodes, dst_nodes):
        self.name = name
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes


class ReForm(object):
    # current just support form/deform single output op
    def __init__(self, model, rigorous):
        self.rigorous = rigorous
        self.reform_info_list = []
        self.nodes = model.graph.node
        self.weight = model.graph.initializer
        self.gout = model.graph.output
        self.ginfo = model.graph.value_info
        # store node shape
        self.shape_info = [info for info in model.graph.value_info]
        self.shape_info.extend(model.graph.output)
        self.shape_info = {
            info.name: [i.dim_value for i in info.type.tensor_type.shape.dim if i.dim_value > 0]
            for info in self.shape_info
        }
        self.weight_tensor = [x.name for x in self.weight]
        self.node_tensor = [node.output[0] for node in self.nodes if node.op_type == "Constant"]
        # stores output node name mapping from src to dst of replace subgraphs
        self.node_name_mapping = {}

    def get_tensor_value(self, name):
        for n in self.nodes:
            if name == n.output[0] and n.op_type == 'Constant':
                return onnx.numpy_helper.to_array(n.attribute[0].t)
        for w in self.weight:
            if name == w.name:
                return onnx.numpy_helper.to_array(w).astype(np.float32)

    def find_tensor(self, name):
        if name in self.node_tensor or name in self.weight_tensor:
            return True
        return False

    def get_node(self, name):
        for idx, n in enumerate(self.nodes):
            if name in n.output:
                return idx, n

    def get_input_shape(self, name):
        for n in self.nodes:
            if name == n.output[0]:
                return self.shape_info[name]
        for w in self.weight:
            if name == w.name:
                return list(w.dims)

    def constraint(self, node, mode):
        if mode == 'broadcast' and len(node.input) == 2:
            inp_0, inp_1 = node.input
            inp0Shape = self.get_input_shape(inp_0)
            inp1Shape = self.get_input_shape(inp_1)
            if len(inp0Shape) == 1 or len(inp1Shape) == 1:
                # normal case
                if inp0Shape[-1] == inp1Shape[-1] \
                    or inp0Shape[-1] == 1 or inp1Shape[-1] == 1:
                    return True
            elif ((inp0Shape[-2] == 1 or inp1Shape[-2] == 1) \
                  and inp0Shape[:-2] == inp1Shape[:-2]):
                # for group fc
                return True
        else:
            raise ValueError("constrain mode: {} not support now.".format(mode))
        return False

    def check_attrs(self, node, attrcheck: AttrCheck):
        attrs = get_node_attrs(node)
        for key in attrcheck.attrs:
            if key not in attrs:
                return False
        args = tuple(attrs[key] for key in attrcheck.attrs)
        return attrcheck.func(*args)

    def attr_to_tensor(self, node, pindices, op_type):
        '''
           high opset node's input is low opset node's attr
           here map pattern node input's idx to attr key
        '''
        tensor_value = []
        attrs = get_node_attrs(node)
        for i in pindices:
            key = "None"
            if op_type == "Clip":
                if i == 1:
                    key = "min"
                elif i == 2:
                    key = "max"
            # add other op here
            try:
                tensor_value.append(np.array(attrs[key]))
            except KeyError:
                pass
        return tensor_value

    def process_low_opset(self, node, pninp, nofdiff):
        tensor_value = []
        if nofdiff == 0:
            return tensor_value
        flag = True  # this mean maybe tensor in pinp is onnx node's attr
        start_idx = len(pninp) - nofdiff
        for pnode in pninp[start_idx:]:
            if not pnode.is_tensor:
                flag = False
                break
        if flag:
            tensor_value = self.attr_to_tensor(node, range(start_idx, len(pninp)), node.op_type)
        if not flag or not len(tensor_value) == nofdiff:
            if self.rigorous:
                raise RuntimeError("Unsupport opset for {}".format(node.op_type))
            else:
                print("Warning unsupport opset for {} skipped.".format(node.op_type))
        return tensor_value

    def match_input(self, node, ninp, pninp):
        nofdiff = len(pninp) - len(ninp)
        if nofdiff >= 0:
            ex_tensor_value = self.process_low_opset(node, pninp, nofdiff)
            ninp = ninp[:] + ex_tensor_value
        else:
            if self.rigorous:
                raise RuntimeError("Unsupport opset for {}".format(node.op_type))
            else:
                print("Warning unsupport opset for {} skipped.".format(node.op_type))
                return False

        for pnode, node_name in zip(pninp, ninp):
            if isinstance(pnode, OuterNode):
                if pnode.is_tensor:
                    tensor_value = node_name
                    if not type(node_name) == np.ndarray:
                        # check if tensor exist
                        if not self.find_tensor(node_name):
                            return False
                        if pnode.attr_name or pnode.tensor_value is not None:
                            tensor_value = self.get_tensor_value(node_name)

                    if pnode.tensor_value is not None:
                        # check tensor value
                        # tensor_value = self.get_tensor_value(node_name)
                        _tensor_value = copy.deepcopy(tensor_value)
                        if _tensor_value.shape == ():
                            _tensor_value = np.expand_dims(_tensor_value, 0)
                        if pnode.tensor_value.shape != _tensor_value.shape \
                           or (pnode.tensor_value != _tensor_value).any():
                            return False
                    if pnode.attr_name:
                        # tensor_value = self.get_tensor_value(node_name)
                        pnode.attr_value = tensor_value
                if not pnode.output or pnode.is_tensor:
                    pnode.output.clear()
                    pnode.output.append(node_name)
            if not node_name in pnode.output:
                return False
        return True

    def match_node(self, node, pnode):
        matched = self.match_input(node, node.input, pnode.input)
        if not matched and (node.op_type == 'Mul' or node.op_type == 'Add'):
            # naive method, need to be discussed
            matched = self.match_input(node, node.input[::-1], pnode.input)
        if matched:
            # process constraint and check attrs
            if pnode.constraint:
                matched = self.constraint(node, pnode.constraint)
            if matched and pnode.attrcheck:
                matched = self.check_attrs(node, pnode.attrcheck)
                if not matched:
                    return matched
            # update output and needed attr
            attr_value = []
            if pnode.cur_attr_name:
                attrs = get_node_attrs(node)
                for key in pnode.cur_attr_name:
                    attr_value.append(attrs[key])
            pnode.update(node.output, attr_value)
        return matched

    def match_pattern(self, reform_info):
        name = reform_info.name
        pnodeIdx = 0
        matched_patterns = []
        unused_nodes = []
        pattern = reform_info.src_nodes
        patternLens = len(pattern)
        for node in self.nodes:
            matched = False
            if node.op_type == 'Constant':
                continue
            if node.op_type == pattern[pnodeIdx].op_type:
                matched = self.match_node(node, pattern[pnodeIdx])
            if matched:
                pnodeIdx += 1
                unused_nodes.append(node)
                if pnodeIdx == patternLens:
                    newNodes = copy.deepcopy(reform_info.dst_nodes)
                    matched_patterns.append(ReformInfo(name, unused_nodes, newNodes))
                    pnodeIdx = 0
                    unused_nodes = []
                    self.reset_outer_node(pattern)
            else:
                pnodeIdx = 0
                unused_nodes = []
                self.reset_outer_node(pattern)
                if node.op_type == pattern[0].op_type:
                    matched = self.match_node(node, pattern[0])
                if matched:
                    pnodeIdx += 1
                    unused_nodes.append(node)
                else:
                    self.reset_outer_node(pattern)
        return matched_patterns

    def reset_outer_node(self, pattern):
        # reset outer node
        for p in pattern:
            for pinp in p.input:
                if isinstance(pinp, OuterNode):
                    pinp.output.clear()

    def replace_pattern(self, matched_pattern):
        # Recently we assume that subgraph to be replace has only one output
        # TODO: implement for multi-output cases
        for reform_info in matched_pattern:
            src_nodes = reform_info.src_nodes
            dst_nodes = reform_info.dst_nodes
            last_node = src_nodes[-1]
            insert_idx, _ = self.get_node(last_node.output[0])
            out = last_node.output
            for i, new_node in enumerate(dst_nodes):
                if i == len(dst_nodes) - 1:
                    _output = out
                else:
                    _output = ["{}_{}".format(last_node.name, i)]
                new_node.output.clear()
                new_node.output.extend(_output)
                _input = []
                for j, inode in enumerate(new_node.input):
                    if isinstance(inode, OuterNode) and len(inode.output) == 0:
                        # insert new tensor node
                        if inode.tensor_value is None:
                            raise ValueError("New tensor node must with tensor_value.")
                        tensor_value = np.array(inode.tensor_value)
                        tensor_name = _output[0] + "_in_{}".format(j)
                        new_onnx_node = onnx.helper.make_node("Constant",
                                                              name=tensor_name,
                                                              inputs=[],
                                                              outputs=[tensor_name],
                                                              value=onnx.helper.make_tensor(
                                                                  "value", onnx.TensorProto.FLOAT,
                                                                  tensor_value.shape, tensor_value))
                        self.nodes.insert(insert_idx, new_onnx_node)
                        insert_idx += 1
                        inode.output.extend(new_onnx_node.output)
                    _input.append(inode.output[0])
                # insert new pattern node
                new_node = onnx.helper.make_node(new_node.op_type,
                                                 name=_output[0],
                                                 inputs=_input,
                                                 outputs=_output,
                                                 **new_node.get_attr())
                self.nodes.insert(insert_idx, new_node)
                insert_idx += 1
            node_name = _output[0]
            src_oname = "{}_{}".format(node_name, src_nodes[-1].op_type)
            dst_oname = "{}_{}".format(node_name, dst_nodes[-1].op_type)
            assert (src_oname not in self.node_name_mapping)
            self.node_name_mapping[src_oname] = dst_oname
            # clear up
            for node in src_nodes:
                self.nodes.remove(node)
            self.remove_unused_tensor()
            # print("[ONNX OPT] RULE <<{}>> applied \n".format(reform_info.name))

    def remove_unused_tensor(self):
        # purging redundancy tensor
        all_input = []
        all_node = [n for n in self.nodes]
        for n in all_node:
            all_input.extend(n.input)
        unused_weight = []
        unused_node = []
        for w in self.weight:
            if w.name in all_input:
                continue
            unused_weight.append(w)
        for n in self.nodes:
            if n.op_type != "Constant" or n.output[0] in all_input:
                continue
            unused_node.append(n)
        for w in unused_weight:
            self.weight.remove(w)
        for n in unused_node:
            self.nodes.remove(n)
        # update
        self.weight_tensor = [x.name for x in self.weight]
        self.node_tensor = [node.output[0] for node in self.nodes if node.op_type == "Constant"]

    def remove_duplicate(self):
        # same op_type and inputs different output_name
        nodes_info = {}
        duplicate_op = {}
        kept_info = {}
        oname_map = {}
        rm_node = []
        # find duplicate node's {op_type: str(inputs)}
        for node in self.nodes:
            if len(node.attribute) > 0:  # FIXME consider node's attr
                continue
            if node.op_type not in nodes_info:
                nodes_info[node.op_type] = []
            nodes_info[node.op_type].append(" ".join(node.input))
        for k, v in nodes_info.items():
            if not len(set(v)) == len(v):
                inputs = dict(Counter(v))
                duplicate_op[k] = [i.split(" ") for i, c in inputs.items() if c > 1]
        nodes_info.clear()
        # find duplicate node's str(input) output_name
        duplicate_op_type = duplicate_op.keys()
        for node in self.nodes:
            if node.op_type not in duplicate_op_type:
                continue
            if node.input in duplicate_op[node.op_type]:
                tinp = node.op_type + " " + " ".join(node.input)
                if tinp not in kept_info:
                    kept_info[tinp] = node.output
                else:
                    okept = kept_info[tinp]
                    oremove = node.output
                    assert (len(okept) == len(oremove))
                    for i in range(len(okept)):
                        oname_map[oremove[i]] = okept[i]
                    rm_node.append(node)
        # remove duplicat node
        for n in rm_node:
            self.nodes.remove(n)
        # verify inputs for each node
        removed_input = oname_map.keys()
        for node in self.nodes:
            for i, inp in enumerate(node.input):
                if inp in removed_input:
                    node.input[i] = oname_map[inp]
        # verify graph output
        for o in self.gout:
            if o.name in removed_input:
                o.name = oname_map[o.name]

    def remove_cast(self):
        cast_ops = []
        cast_in_dict = defaultdict(str)
        cast_out_dict = defaultdict(str)
        net_out_names = set()
        for gout in self.gout:
            net_out_names.add(gout.name)
        reverse_search = False

        def find_cast(node, cast_dict):
            if node not in cast_dict:
                return node
            else:
                return find_cast(cast_dict[node], cast_dict)

        def insert_identity(cur_node_out, out_name):
            identity_node = onnx.helper.make_node(
                                "Identity",
                                name=cur_node_out + "_insert_Identity",
                                inputs=[cur_node_out],
                                outputs=[out_name])
            insert_idx, _ = self.get_node(out_name)
            self.nodes.insert(insert_idx, identity_node)

        for node in self.nodes:
            if node.op_type == "Cast":
                cast_ops.append(node)
                cast_in_dict[node.output[0]] = node.input[0]
                if node.output[0] in net_out_names: reverse_search = True
                continue
            if node.op_type == "Constant":
                continue
            for i in range(len(node.input)):
                if node.input[i] in cast_in_dict:
                    node.input[i] = find_cast(cast_in_dict[node.input[i]], cast_in_dict)

        if reverse_search:
            for node in reversed(self.nodes):
                if node.op_type == "Cast":
                    cast_out_dict[node.input[0]] = node.output[0]
                    continue
                if node.op_type == "Constant":
                    continue
                for i in range(len(node.output)):
                    if node.output[i] in cast_out_dict:
                        out_name = find_cast(cast_out_dict[node.output[i]], cast_out_dict)
                        if out_name in net_out_names:
                            insert_identity(node.output[i], out_name)

        for op in cast_ops:
            self.nodes.remove(op)

    # remove invalid slice node which shape is 0
    def remove_invalid_slice(self):
        node_slice_name = []
        node_invalid_slice_name = []
        delete_info_ops = []
        delte_node_ops = []

        for node in self.nodes:
            if node.op_type == "Slice":
                node_slice_name.append(node.output[0])

        if len(node_slice_name) > 0:
            for info in self.ginfo:
                if info.name in node_slice_name:
                    for info_dim in info.type.tensor_type.shape.dim:
                        if info_dim.HasField("dim_value") and info_dim.dim_value == 0:
                            node_invalid_slice_name.append(info.name)
                            delete_info_ops.append(info)

        if len(node_invalid_slice_name) > 0:
            for node in self.nodes:
                if node.output[0] in node_invalid_slice_name:
                    delte_node_ops.append(node)
                for i, input_name in enumerate(node.input):
                    if input_name in node_invalid_slice_name:
                        del node.input[i]

        for op in delete_info_ops:
            self.ginfo.remove(op)

        for op in delte_node_ops:
            self.nodes.remove(op)


    def graph_opt(self):
        replaced = False
        for reform_info in self.reform_info_list:
            matched_pattern = self.match_pattern(reform_info)
            if len(matched_pattern) > 0:
                replaced = True
            self.replace_pattern(matched_pattern)
        if replaced:
            self.graph_opt()

    def __call__(self, reform_info_list):
        self.reform_info_list = reform_info_list
        self.remove_cast()
        self.remove_invalid_slice()
        self.remove_duplicate()
        self.graph_opt()
        return self.node_name_mapping, self.nodes, self.weight


###====================== Declare your patterns here ======================###


############ torch.LayerNorm ############
def TorchLayerNormPattern(patterns: list):
    def is_last_dims(x: list):
        return np.all(np.diff(x) == 1) and x[-1] == -1

    reducemean_input = OuterNode()

    pow_tensor = OuterNode(tensor_value=2)
    add_0_tensor = OuterNode(attr_name="eps")
    mul_tensor = OuterNode(is_tensor=True)
    add_1_tensor = OuterNode(is_tensor=True)

    _reducemean_0 = PatternNode(
        "ReduceMean",
        [reducemean_input],
        ["axes"],
        attrcheck=AttrCheck(attrs=["axes"], func=is_last_dims),
    )
    _sub = PatternNode("Sub", [reducemean_input, _reducemean_0])
    _pow = PatternNode("Pow", [_sub, pow_tensor])
    _reducemean_1 = PatternNode(
        "ReduceMean",
        [_pow],
        attrcheck=AttrCheck(attrs=["axes"], func=is_last_dims),
    )
    _add_0 = PatternNode("Add", [_reducemean_1, add_0_tensor])
    _sqrt = PatternNode("Sqrt", [_add_0])
    _div = PatternNode("Div", [_sub, _sqrt])
    mul = PatternNode("Mul", [_div, mul_tensor])
    _add_1 = PatternNode("Add", [mul, add_1_tensor])

    epsilon_attrfunc = AttrFunctor([add_0_tensor], ["eps"])
    axis_attrfunc = AttrFunctor([_reducemean_0], ["axes"], lambda x: x[0])

    # affine (have both weight and bias)
    layernorm_aff = PatternNode("LayerNormalization", [reducemean_input, mul_tensor, add_1_tensor],
                                attrmap={
                                    "epsilon": epsilon_attrfunc,
                                    "axis": axis_attrfunc
                                })
    patterns.append(
        ReformInfo(
            name="layernorm_aff",
            src_nodes=[_reducemean_0, _sub, _pow, _reducemean_1, _add_0, _sqrt, _div, mul, _add_1],
            dst_nodes=[layernorm_aff]))
    # without affine (do not have both weight and bias)
    layernorm = PatternNode("LayerNormalization", [reducemean_input],
                            attrmap={
                                "epsilon": epsilon_attrfunc,
                                "axis": axis_attrfunc
                            })
    patterns.append(
        ReformInfo(name="layernorm",
                   src_nodes=[_reducemean_0, _sub, _pow, _reducemean_1, _add_0, _sqrt, _div],
                   dst_nodes=[layernorm]))

############ torch.PixelNorm ############
def TorchPixelNormPattern(patterns: list):
    def is_c_dim(x: list):
        return len(x) == 1 and x[0] == 1

    reducemean_input = OuterNode()

    pow_tensor = OuterNode(tensor_value=2)
    add_0_tensor = OuterNode(attr_name="eps")
    mul_tensor = OuterNode(is_tensor=True)
    add_1_tensor = OuterNode(is_tensor=True)

    _reducemean_0 = PatternNode(
        "ReduceMean",
        [reducemean_input],
        ["axes"],
        attrcheck=AttrCheck(attrs=["axes"], func=is_c_dim),
    )
    _sub = PatternNode("Sub", [reducemean_input, _reducemean_0])
    _pow = PatternNode("Pow", [_sub, pow_tensor])
    _reducemean_1 = PatternNode(
        "ReduceMean",
        [_pow],
        attrcheck=AttrCheck(attrs=["axes"], func=is_c_dim),
    )
    _add_0 = PatternNode("Add", [_reducemean_1, add_0_tensor])
    _sqrt = PatternNode("Sqrt", [_add_0])
    _div = PatternNode("Div", [_sub, _sqrt])
    mul = PatternNode("Mul", [_div, mul_tensor])
    _add_1 = PatternNode("Add", [mul, add_1_tensor])

    epsilon_attrfunc = AttrFunctor([add_0_tensor], ["eps"])

    # affine (have both weight and bias)
    layernorm_aff = PatternNode("PixelNormalization", [reducemean_input, mul_tensor, add_1_tensor],
                                attrmap={"epsilon": epsilon_attrfunc})
    patterns.append(
        ReformInfo(
            name="pixelnorm_aff",
            src_nodes=[_reducemean_0, _sub, _pow, _reducemean_1, _add_0, _sqrt, _div, mul, _add_1],
            dst_nodes=[layernorm_aff]))
    # without affine (do not have both weight and bias)
    layernorm = PatternNode("PixelNormalization", [reducemean_input],
                            attrmap={"epsilon": epsilon_attrfunc})
    patterns.append(
        ReformInfo(name="pixelnorm",
                   src_nodes=[_reducemean_0, _sub, _pow, _reducemean_1, _add_0, _sqrt, _div],
                   dst_nodes=[layernorm]))

############ torch.GELU ############
def TorchGELUPattern(patterns: list):
    gelu_input = OuterNode()
    div_tensor = OuterNode(is_tensor=True)
    add_tensor = OuterNode(tensor_value=1)
    mul_tensor = OuterNode(tensor_value=0.5)

    _div = PatternNode("Div", [gelu_input, div_tensor])
    _erf = PatternNode("Erf", [_div])
    _add = PatternNode("Add", [_erf, add_tensor])
    _mu_0 = PatternNode("Mul", [gelu_input, _add])
    _mul_1 = PatternNode("Mul", [_mu_0, mul_tensor])
    gelu = PatternNode("GELU", [gelu_input])
    patterns.append(
        ReformInfo(name="GELU", src_nodes=[_div, _erf, _add, _mu_0, _mul_1], dst_nodes=[gelu]))


def TorchGELUPattern2(patterns: list):
    gelu_input = OuterNode()
    add_tensor = OuterNode(tensor_value=1)
    mul_tensor = OuterNode(tensor_value=0.5)
    power_tensor = OuterNode(tensor_value=3)
    add_tensor = OuterNode(tensor_value=1)
    mul_tensor_1 = OuterNode(is_tensor=True)
    mul_tensor_2 = OuterNode(is_tensor=True)
    _mul_0 = PatternNode("Mul", [gelu_input, mul_tensor])
    _power_1 = PatternNode("Pow", [gelu_input, power_tensor])
    _mul_2 = PatternNode("Mul", [_power_1, mul_tensor_1])
    _add_3 = PatternNode("Add", [gelu_input, _mul_2])
    _mul_4 = PatternNode("Mul", [_add_3, mul_tensor_2])
    _tanh_5 = PatternNode("Tanh", [_mul_4])
    _add_6 = PatternNode("Add", [_tanh_5, add_tensor])
    _mul_7 = PatternNode("Mul", [_mul_0, _add_6])
    gelu = PatternNode("GELU", [gelu_input])
    patterns.append(
        ReformInfo(name="GELU",
                   src_nodes=[_mul_0, _power_1, _mul_2, _add_3, _mul_4, _tanh_5, _add_6, _mul_7],
                   dst_nodes=[gelu]))


############ torch.HardSigmodid ############
def TorchHardSigmoidPattern(patterns: list):
    # nomal case
    add_input = OuterNode()
    add_tensor = OuterNode(tensor_value=3)
    clip_min = OuterNode(tensor_value=0)
    clip_max = OuterNode(tensor_value=6)
    div_tensor = OuterNode(tensor_value=6)
    add = PatternNode("Add", [add_input, add_tensor])
    clip = PatternNode("Clip", [add, clip_min, clip_max])
    div = PatternNode("Div", [clip, div_tensor])
    hard_sigmoid = PatternNode("HardSigmoid", [add_input])
    patterns.append(
        ReformInfo(name="HardSigmoid", src_nodes=[add, clip, div], dst_nodes=[hard_sigmoid]))


############ torch.HardSwish ############
def TorchHardSwishPattern(patterns: list):
    input = OuterNode()
    attrcheck = AttrCheck(attrs=['alpha', 'beta'],
                          func=lambda x, y: x == 0.1666666716337204 and y == 0.5)
    hard_sigmoid = PatternNode("HardSigmoid", [input], attrcheck=attrcheck)
    mul = PatternNode("Mul", [input, hard_sigmoid])
    hard_swish = PatternNode("HardSwish", [input])
    patterns.append(
        ReformInfo(name="hardswish", src_nodes=[hard_sigmoid, mul], dst_nodes=[hard_swish]))


def TorchHardSwishPattern2(patterns: list):
    add_input = OuterNode()
    add_tensor = OuterNode(tensor_value=3)
    clip_min = OuterNode(tensor_value=0)
    clip_max = OuterNode(tensor_value=6)
    div_tensor = OuterNode(tensor_value=6)

    add = PatternNode("Add", [add_input, add_tensor])
    clip = PatternNode("Clip", [add, clip_min, clip_max])
    mul = PatternNode("Mul", [add_input, clip])
    div = PatternNode("Div", [mul, div_tensor])
    hard_swish = PatternNode("HardSwish", [add_input])
    patterns.append(
        ReformInfo(name="hardswish", src_nodes=[add, clip, mul, div], dst_nodes=[hard_swish]))


def remove_tensor_from_input(model):
    tensor_names = [x.name for x in model.graph.initializer]
    tensor_names.extend([x.name for x in model.graph.node if x.op_type == "Constant"])
    inputs = model.graph.input
    tensors = []
    for i in inputs:
        if i.name in tensor_names:
            tensors.append(i)
    for t in tensors:
        model.graph.input.remove(t)


def onnx_opt(model, dump=False, rigorous=True):
    remove_tensor_from_input(model)
    # add your patterns here if you expect that your patterns actually works
    pattern_functions = [
        TorchLayerNormPattern,
        TorchPixelNormPattern,
        TorchHardSigmoidPattern,
        TorchHardSwishPattern,
        TorchHardSwishPattern2,
        TorchGELUPattern,
        TorchGELUPattern2,
    ]

    patterns = []
    for pf in pattern_functions:
        pf(patterns)

    reform = ReForm(model, rigorous)
    node_name_mapping, _, _ = reform(patterns)
    if dump:
        dump_model(model, "final_opt.onnx")
    return model, node_name_mapping
