import onnx
import numpy as np
from onnx import numpy_helper
import os
import copy
from numpy_helper.npz_compare import align_type_and_shape
from numpy_helper.tensor_compare import TensorCompare, TensorCompareStats


class FakeQuantNodelProcessor:
    PERCHANNEL_FAKEQUANTIZER = [
        'FakeQuantizeLearnablePerchannelAffine', 'FixedPerChannelAffine',
        'FakeQuantizeDSQPerchannel', 'FPEmuOp_per_channel'
    ]
    PERTENSOR_FAKEQUANTIZER = [
        'LearnablePerTensorAffine', 'FixedPerTensorAffine', 'FakeQuantizeDSQPertensor',
        'FakeQuantizeTqtAffine', 'FPEmuOp_per_tensor'
    ]
    ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER

    def __init__(self, input_model_path, input_model_name):
        self.input_model_name = input_model_name
        self.input_model_path = input_model_path
        self.output_model_path = input_model_path.replace('.onnx', '_qat.onnx')
        self.fakequant_model = self.check_onnx_for_fakequant_nodes(input_model_path)
        self.calitable_name = input_model_name + "_calitable_qat"
        self.qtable_name = input_model_name + "_qtable_qat"
        self.nodes_to_be_removed = []
        self.cali_table = {}
        self.weight_table = {}
        self.q_table = {}

    def check_onnx_for_fakequant_nodes(self, onnx_model_path):
        try:
            self.model = onnx.load(onnx_model_path)
            self.graph = self.model.graph
            valid_ops = set(self.ALL_FAKEQUANTIZER)
            for node in self.graph.node:
                if node.op_type in valid_ops:
                    return True
            return False
        except Exception as e:
            print(f"load fakequant ONNX fail: {e}")
            return False

    def prepare_params(self):

        self.out2node, self.inp2node = self.update_inp2node_out2node(self.graph)
        self.name2data = self.prepare_data(self.graph)
        self.named_initializer = self.prepare_initializer(self.graph)

    def parse_attrs(self, node_attrs):
        attrs = {}
        for attr in node_attrs:
            if attr.type == onnx.AttributeProto.AttributeType.INTS:
                attrs[attr.name] = tuple(attr.ints)
                attrs['dtype'] = 'ints'
            elif attr.type == onnx.AttributeProto.AttributeType.INT:
                attrs[attr.name] = attr.i
                attrs['dtype'] = 'int'
            elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
                attrs[attr.name] = tuple(attr.floats)
                attrs['dtype'] = 'floats'
            elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
                attrs[attr.name] = attr.f
                attrs['dtype'] = 'float'
            elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
                attrs[attr.name] = numpy_helper.to_array(attr.t)
                attrs['dtype'] = 't'
            elif attr.type == onnx.AttributeProto.AttributeType.STRING:
                attrs[attr.name] = str(attr.s)
                attrs['dtype'] = 'st'
            elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
                attrs[attr.name] = tuple([str(x) for x in attr.strings])
                attrs['dtype'] = 'none'
            else:
                raise Exception("ATTR Type [{}] Not Supported!".format(attr.type))
        return attrs

    def update_inp2node_out2node(self, graph):
        out2node = {}
        inp2node = {}
        for node in graph.node:
            for out in node.output:
                out2node[out] = node
            for idx, inp in enumerate(node.input):
                if inp not in inp2node:
                    inp2node[inp] = []
                inp2node[inp].append([node, idx])
        return out2node, inp2node

    def prepare_data(self, graph):
        params = {}
        for init in graph.initializer:
            params[init.name] = numpy_helper.to_array(init)
        for node in graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        params[node.output[0]] = numpy_helper.to_array(attr.t)
        return params

    def prepare_initializer(self, graph):
        named_initializer = {}
        for init in graph.initializer:
            named_initializer[init.name] = init
        return named_initializer

    def get_constant_inputs(self, node):
        node_list = []
        for inp in node.input:
            if inp in self.out2node and self.out2node[inp].op_type == 'Constant':
                node_list.append(self.out2node[inp])
        return node_list

    def remove_fake_pad_op(self):
        nodes_to_be_removed = []
        for idx, node in enumerate(self.graph.node):
            if node.op_type == 'Pad':
                pads = self.name2data[node.input[1]]
                if all([x == 0 for x in pads]):
                    print(f"Remove pad op: <{node.name}>.")
                    next_nodes = self.inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]
                    nodes_to_be_removed.append(node)
                    nodes_to_be_removed.extend(self.get_constant_inputs(node))
        for node in nodes_to_be_removed:
            self.graph.node.remove(node)

    def deal_with_weight_fakequant(self, node):
        next_nodes = self.inp2node[node.output[0]]
        assert len(next_nodes) == 1
        next_node, idx = next_nodes[0]
        assert next_node.op_type in ['Conv', 'Gemm', 'ConvTranspose']
        redundant_nodes = []

        if node.input[0] not in self.named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0])

        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def weight_preprocess(self, target_tensor):

        def find_weight(tensor):
            if tensor not in self.named_initializer:
                _node = self.out2node[tensor]
                for inp in _node.input:
                    return find_weight(inp)
            return tensor

        weight = find_weight(target_tensor)
        data = numpy_helper.to_array(self.named_initializer[weight])
        data = np.tanh(data)
        data = data / (np.max(np.abs(data)) + 1e-5)
        data = numpy_helper.from_array(data)
        self.named_initializer[weight].raw_data = data.raw_data

        redundant_nodes = []

        def find_redundant_nodes(tensor):
            if tensor == target_tensor:
                return
            nodes = self.inp2node[tensor]
            for node, idx in nodes:
                if node not in redundant_nodes:
                    redundant_nodes.append(node)
                    redundant_nodes.extend(self.get_constant_inputs(node))
                find_redundant_nodes(node.output[0])

        find_redundant_nodes(weight)
        return weight, redundant_nodes

    def clip_weight(self, node):
        tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = self.parse_qparams(node)
        data = self.name2data[tensor_name]
        clip_range_min = ((qmin - zero_point) * scale).astype(data.dtype)
        clip_range_max = ((qmax - zero_point) * scale).astype(data.dtype)
        if len(scale.shape) > 0 and scale.shape[0] > 1:
            new_data = []
            transposed = False
            next_node = self.inp2node[node.output[0]]
            if len(next_node) == 1 and next_node[0][0].op_type == 'ConvTranspose':
                transposed = True
                data = data.transpose(1, 0, 2, 3)
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            if transposed:
                new_data = new_data.transpose(1, 0, 2, 3)
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
        new_data = numpy_helper.from_array(new_data)
        self.named_initializer[tensor_name].raw_data = new_data.raw_data

    def parse_qparams(self, node):
        tensor_name, scale, zero_point = node.input[:3]
        dtype = 'None'
        scale, zero_point = self.name2data[scale], self.name2data[zero_point]

        scale_name = node.input[1]
        module_name = scale_name.rsplit('.scale', 1)[0]
        quant_type = 'None'

        if len(node.input) > 3:
            qmin, qmax = node.input[-2:]
            qmin, qmax = self.name2data[qmin], self.name2data[qmax]
            if len(node.attribute) > 0:
                qparams = self.parse_attrs(node.attribute)
                dtype = qparams['dtype']
            else:
                dtype = 'None'
        elif len(node.attribute) > 0:
            qparams = self.parse_attrs(node.attribute)
            qmin = qparams['quant_min']
            qmax = qparams['quant_max']
            dtype = qparams['dtype']
        else:
            print.info(f'qmin and qmax are not found for <{node.name}>!')

        if qmax == float(448.0) or qmax == float(57344.0):
            quant_type = 'FP8'

        bit = int(np.log2(qmax - qmin + 1))
        if (bit == 8 and qmin < 0):
            dtype = 'int'
            quant_type = 'INT8'
        elif (bit == 8 and qmin == 0):
            dtype = 'uint'
            quant_type = 'UINT8'
        elif (bit == 4 and qmin < 0):
            dtype = 'int'
            quant_type = 'INT4'
        return tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type

    def post_process_clip_ranges(self):
        op_type_inAndOutShouldSameClipRange = ['Flatten', 'Resize', 'Reshape', 'Transpose']
        for node in self.graph.node:
            tensor_name = f'{node.output[0]}_{node.op_type}'
            if tensor_name not in self.cali_table:
                pre_op = node
                finded = False
                while pre_op.op_type in op_type_inAndOutShouldSameClipRange:
                    input_0 = pre_op.input[0]
                    op_type = ''
                    if input_0 in self.out2node:
                        op_type = self.out2node[input_0].op_type
                        tensor_name2 = '{}_{}'.format(input_0, op_type)
                    else:
                        tensor_name2 = input_0

                    if tensor_name2 in self.cali_table:
                        finded = True
                        self.cali_table[tensor_name] = [
                            tensor_name, *self.cali_table[tensor_name2][1:4]
                        ]
                        self.q_table[tensor_name] = [tensor_name, self.cali_table[tensor_name2][1]]
                        print(f'pre_op finded, transfer {tensor_name2} to {tensor_name}')
                        break
                    if pre_op.input[0] in self.out2node:
                        pre_op = self.out2node[pre_op.input[0]]
                    else:
                        print(f'{pre_op.name}\'s pre_node not exist')
                        break
                if not finded:
                    if node.output[0] in self.inp2node:
                        next_op = self.inp2node[node.output[0]][0][0]
                        while next_op.op_type in op_type_inAndOutShouldSameClipRange:
                            tensor_name2 = f'{next_op.output[0]}_{next_op.op_type}'
                            if tensor_name2 in self.cali_table:
                                finded = True
                                self.cali_table[tensor_name] = [
                                    tensor_name, *self.cali_table[tensor_name2][1:4]
                                ]
                                self.q_table[tensor_name] = [
                                    tensor_name, self.cali_table[tensor_name2][1]
                                ]
                                print(f'next_op finded, transfer {tensor_name2} to {tensor_name}')
                                break
                            if next_op.output[0] in self.inp2node:
                                next_op = self.inp2node[next_op.output[0]][0][0]
                            else:
                                print(f'{next_op.name}\'s next_op not exist')
                                break
                    else:
                        print(f'{node.name}\'s next_op not exist')

    def deal_with_activation_fakequant(self, node):
        next_nodes = self.inp2node[node.output[0]]
        for next_node, idx in next_nodes:
            next_node.input[idx] = node.input[0]

    def process_model(self):
        self.prepare_params()
        self.remove_fake_pad_op()
        self.update_inp2node_out2node(self.graph)
        for node in self.graph.node:
            print(f'process_node :{node.name}, type:{node.op_type}')
            if node.op_type in self.ALL_FAKEQUANTIZER:
                self.nodes_to_be_removed.append(node)
                self.nodes_to_be_removed.extend(self.get_constant_inputs(node))
            if node.output[0] not in self.inp2node:
                assert node.output[0] in [l.name for l in self.graph.output]
                self.inp2node[node.output[0]] = []
            next_nodes = self.inp2node[node.output[0]]
            if node.op_type in self.PERCHANNEL_FAKEQUANTIZER:
                redundant_nodes = self.deal_with_weight_fakequant(node)
                self.nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node)
                tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = self.parse_qparams(
                    node)
                if len(next_nodes) == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    next_node_output = next_nodes[0][0].output[0]
                    assert next_nodes[0][0].op_type in ['Gemm', 'Conv']
                    if self.inp2node[next_node_output][0][0].op_type == 'Relu':
                        tensor_name = '{}_{}_weight'.format(
                            self.inp2node[next_node_output][0][0].output[0], 'Relu')
                    else:
                        tensor_name = '{}_{}_weight'.format(next_node_output,
                                                            next_nodes[0][0].op_type)
                    self.weight_table[tensor_name] = [
                        tensor_name,
                        len(scale), *[float(f"{float(x):.7f}") for x in scale],
                        len(zero_point), *[int(x) for x in zero_point]
                    ]
                    self.q_table[tensor_name] = [tensor_name, quant_type]

            elif node.op_type in self.PERTENSOR_FAKEQUANTIZER:
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in [
                        'Gemm', 'Conv'
                ]:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node)
                    self.nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node)
                    tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = self.parse_qparams(
                        node)
                    assert next_nodes[0][0].op_type in ['Gemm', 'Conv']
                    tensor_name_new = '{}_{}_weight'.format(next_nodes[0][0].output[0],
                                                            next_nodes[0][0].op_type)
                    self.weight_table[tensor_name_new] = [
                        tensor_name_new,
                        len(scale), *[float(f"{float(x):.7f}") for x in scale],
                        len(zero_point), *[int(x) for x in zero_point]
                    ]
                    self.q_table[tensor_name_new] = [tensor_name_new, quant_type]
                else:
                    tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = self.parse_qparams(
                        node)
                    tensor_name = node.input[0]
                    if node.input[0] in self.out2node:
                        pre_node = self.out2node[tensor_name]
                        pre_type = pre_node.op_type
                        tensor_name = '{}_{}'.format(tensor_name, pre_type)
                    self.cali_table[tensor_name] = [
                        tensor_name,
                        float(f"{float(scale * max(-qmin, qmax)):.7f}"),
                        float(f"{float(scale * (qmin - zero_point)):.7f}"),
                        float(f"{float(scale * (qmax - zero_point)):.7f}")
                    ]
                    self.q_table[tensor_name] = [tensor_name, quant_type]
                    self.deal_with_activation_fakequant(node)
                    output_name = node.output[0]
                    for out in self.graph.output:
                        if out.name == output_name:
                            out.name = node.input[0]

        for node in self.nodes_to_be_removed:
            self.graph.node.remove(node)
        model_onnx = onnx.shape_inference.infer_shapes(self.model)
        onnx.save(model_onnx, self.output_model_path)

        # delete initializer
        self.out2node, self.inp2node = self.update_inp2node_out2node(self.graph)
        self.named_initializer = self.prepare_initializer(self.graph)
        for name, initial_data in self.named_initializer.items():
            if name in (self.out2node.keys() | self.inp2node.keys()):
                continue
            self.graph.initializer.remove(initial_data)

        self.post_process_clip_ranges()

    def export_tables(self):
        print("导出calitable")
        with open(self.calitable_name, 'w') as f:
            f.write(
                f"#Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n"
            )
            f.write("# op_name    threshold    min    max\n")
            for entry in self.cali_table.values():
                line = ' '.join(map(str, entry))
                f.write(line + '\n')
            # f.write('#weight_scale\n')
            # for entry in self.weight_table.values():
            #     line = ' '.join(map(str, entry))
            #     f.write(line + '\n')

        print("导出qtable")
        with open(self.qtable_name, 'w') as f:
            f.write("# qtable from sophgo_mq\n")
            f.write("# op_name  quantize_mode\n")
            for entry in self.q_table.values():
                line = ' '.join(map(str, entry))
                f.write(line + '\n')

    def get_activation_mappings(self, model):
        cast_dict = ['Constant']
        output_to_next_outputs = {}
        output_to_inputs = {}
        for node in model.graph.node:
            if node.op_type not in cast_dict:
                for output in node.output:
                    output_to_inputs[output] = []
                    output_to_next_outputs[output] = []
        for node in model.graph.node:
            for input_name in node.input:
                if input_name in output_to_next_outputs:
                    output_to_next_outputs[input_name].extend(node.output)
        for node in model.graph.node:
            for output in node.output:
                for input_name in node.input:
                    if input_name in output_to_inputs.keys():
                        output_to_inputs[output].append(input_name)

        return output_to_next_outputs, output_to_inputs

    def align_final_opt(self, node_name_mapping, onnx_sim=""):
        self.opt_node_mapping = node_name_mapping
        # get onnx simplify activation name
        simplify_file = "{}_opt.onnx".format(self.input_model_name)
        simplify_model = None
        if not os.path.exists(simplify_file):
            import onnxsim.onnx_simplifier as onnxsim
            print(f"simplify onnx {simplify_file} is not exit, mybe deleted by auto_remove")
            onnx_sim_str = onnx_sim.split(',')
            skip_fuse_bn = "skip_fuse_bn" in onnx_sim_str
            simplify_model = onnxsim.simplify(self.model,
                                              skip_fuse_bn=skip_fuse_bn,
                                              skip_constant_folding=True,
                                              skip_shape_inference=True)
        else:
            simplify_model = onnx.load(simplify_file)
        sim_out2node, sim_inp2node = self.update_inp2node_out2node(simplify_model.graph)
        ori_outputs, ori_inputs = self.get_activation_mappings(self.model)
        sim_outputs, sim_inputs = self.get_activation_mappings(simplify_model)

        sim_node_mapping = {}
        not_find_dict = []
        for sim_output, out_list in sim_outputs.items():
            sim_node = sim_out2node[sim_output]
            # If the op activation isnot in the original activation dictionary, the op is simplified
            if sim_output not in ori_outputs.keys():
                not_find_dict.append(sim_output)
                # case: 1->2->3->4->5->6->7, after onnxsim: 1->3.1->5.1->6->7
                # only when the next node is the original node can aid in inference
                # example, if 6 is the ori node, only 5.1 = 5 can be introduced, and 3.1 may be node 2~4
                for out_name in out_list:
                    # when the output of the successor op is in the original model, then judge the inputs to this op
                    # if agrees, proving that this op is not the fused operator
                    # if not agrees, op is fused operator, can not get the original name
                    if out_name in ori_outputs.keys():
                        ori_nextop = self.out2node[out_name]
                        sim_nextop = sim_out2node[out_name]
                        # next op type and name are same, not the fused op
                        if ori_nextop.op_type == sim_nextop.op_type and ori_nextop.name == sim_nextop.name:
                            sim_nextop_inpus = sim_inputs[out_name]
                            ori_nextop_inpus = ori_inputs[out_name]
                            new_node_inpus_copy = copy.deepcopy(ori_nextop_inpus)
                            # Remove the input of the same part of the successor node
                            for inp_name in sim_nextop_inpus:
                                if inp_name in ori_nextop_inpus:
                                    new_node_inpus_copy.remove(inp_name)
                            # inputs of next node can only one different name in list, otherwise namemap cannot get
                            if len(new_node_inpus_copy) == 1:
                                ori_node = self.out2node[new_node_inpus_copy[0]]
                                ori_name = f"{new_node_inpus_copy[0]}_{ori_node.op_type}"
                                sim_name = f"{sim_output}_{sim_node.op_type}"
                                sim_node_mapping[ori_name] = sim_name
                                # doesnt insert fakequant node by fuse mode, exsample conv + relu; avagepool + flatten
                                if ori_name in self.cali_table:
                                    self.cali_table[ori_name][0] = sim_name
                                    self.q_table[ori_name][0] = sim_name
                                not_find_dict.remove(sim_output)
            # slice + concat + reshape = reshape, activation name is not change
            else:
                ori_node = self.out2node[sim_output]
                # op type mybe change
                if sim_node.op_type != ori_node.op_type:
                    sim_name = f"{sim_output}_{sim_node.op_type}"
                    ori_name = f"{sim_output}_{ori_node.op_type}"
                    # doesnt insert fakequant node by fuse mode, exsample conv + relu; avagepool + flatten
                    if ori_name in self.cali_table:
                        sim_node_mapping[ori_name] = sim_name
                        self.creat_new_qat_item(ori_name, sim_name)

        print("simplify onnx cant compare activation name list:", not_find_dict)
        print("simplify onnx rename pair:", sim_node_mapping)
        print("final opt onnx rename pair:", node_name_mapping)
        self.sim_node_mapping = sim_node_mapping
        # created by onnx_opt: example, hardsigmoid + mul = hardswish
        for qat_act_name in node_name_mapping:
            finalopt_name = node_name_mapping[qat_act_name]
            if qat_act_name in self.cali_table:
                self.creat_new_qat_item(qat_act_name, finalopt_name)

        # self.export_tables()

    def creat_new_qat_item(self, oir_name, new_name):
        self.cali_table[new_name] = copy.deepcopy(self.cali_table[oir_name])
        self.cali_table[new_name][0] = new_name
        self.q_table[new_name] = copy.deepcopy(self.q_table[oir_name])
        self.q_table[new_name][0] = new_name

    def align_canonicalize(self, mlir_file, test_result):
        self.act_list = self.extract_activation_names_mlir(mlir_file)
        keys_qat = set(self.cali_table.keys())
        keys_mlir = set(self.act_list)
        unpair_keys = keys_mlir - keys_qat
        complete_match = True
        for key in unpair_keys:
            if "_r_" in key:
                before_canonicalize = key[:key.index("_r_")]
                if before_canonicalize in keys_qat:
                    self.creat_new_qat_item(before_canonicalize, key)
                else:
                    complete_match = False
            else:
                complete_match = False
        if complete_match:
            print("successfully get all mlir op_name by align_canonicalize.")
        else:
            _, _, not_match = self.compare_npz_name_mapping(test_result)
            keys_qat_after_align = set(self.cali_table.keys())
            unpair_keys = keys_mlir - keys_qat_after_align
            if unpair_keys:
                print("remain this op name cant find in QAT calitable:", unpair_keys)
                print(
                    "QAT calitable cont compare to mlir, please run run_calitable to get the available calitable!!"
                )
            else:
                print("successfully get all mlir op_name.")
        self.export_tables()

    def extract_activation_names_mlir(self, mlir_path):
        import re
        with open(mlir_path, 'r') as file:
            mlir_code = file.read()
        loc_dict = {}
        loc_pattern = re.compile(r'#loc(?P<id>\d+)\s*=\s*loc\((?P<content>.*?)\)', re.MULTILINE)
        loc_matches = loc_pattern.findall(mlir_code)
        loc_dict = {f"#loc{num}": name for num, name in loc_matches}

        op_pattern = re.compile(
            r'^\s*((?:%\w+\s*,\s*)*%[\w\.]+)\s*=\s*"([^"]+)"\(([^)]*)\)'
            r'.*?loc\((?P<loc>[^\)]+)\)', re.MULTILINE | re.DOTALL)
        op_matches = op_pattern.findall(mlir_code)
        act_list = []
        for _, _type, inputs, loc in op_matches:
            if inputs.strip():
                if loc_dict[loc].startswith('fused['):
                    nested_locs = [f"{x.strip()}" for x in re.findall(r'#loc\d+', loc_dict[loc])]
                    for node in nested_locs:
                        act_list.append(loc_dict[node].strip('"'))
                else:
                    act_list.append(loc_dict[loc].strip('"'))
        return act_list

    def compare_npz_name_mapping(self, test_result):
        data_a_name = f"{self.input_model_name}_ref_outputs.npz"
        if not os.path.exists(data_a_name):
            from tools.model_runner import onnx_inference
            print(f"onnx reference data: {data_a_name} is not exit, mybe deleted by auto_remove")
            self.in_f32_npz = self.input_model_name + '_in_f32.npz'
            inputs = np.load(self.in_f32_npz)
            simplify_file = "{}_opt.onnx".format(self.input_model_name)
            data_a = onnx_inference(inputs, simplify_file)
            print("Saving {}".format(data_a_name))
            np.savez(data_a_name, **data_a)
        else:
            data_a = np.load(data_a_name)
        data_b = np.load(test_result)
        tc = TensorCompare(
            close_order_tol=3,
            cosine_similarity_tol=0.98,
            euclidean_similarity_tol=0.95,
            signal_to_quantization_noise_tol=float("-inf"),
            per_axis_compare=-1,
        )
        keys_a = set(data_a.keys())
        keys_b = set(data_b.keys())
        remaining_keys_a = keys_a - keys_b
        unique_keys_b = keys_b - keys_a
        paired_keys = {}
        double_match = []
        not_match = copy.deepcopy(unique_keys_b)
        for key_b in unique_keys_b:
            val_b = data_b[key_b]
            for key_a in remaining_keys_a:
                val_a = data_a[key_a]
                if val_a.shape == val_b.shape:
                    val_a, val_b = align_type_and_shape(val_a, val_b)
                    result = tc.compare(val_a, val_b, 0, 1, -1)
                    if result[1] == "EQUAL" or result[1] == "SIMILAR":
                        if key_a not in paired_keys.keys():
                            paired_keys[key_a] = key_b
                            not_match.remove(key_b)
                        else:
                            double_match.append(key_b)
        print("final mlir rename pair:", paired_keys)
        print("final mlir double match:", double_match)
        for qat_act_name in paired_keys:
            mlir_name = paired_keys[qat_act_name]
            if qat_act_name in self.cali_table and qat_act_name not in double_match:
                if mlir_name not in self.cali_table:
                    self.creat_new_qat_item(qat_act_name, mlir_name)
        return paired_keys, double_match, not_match
