import os
import pymlir
import numpy as np
import mlir.dialects.top as top
from tqdm import tqdm
from calibration.model_modifier import ModelModifier
from utils.mlir_parser import Operation, MlirParser
from utils.mlir_shell import _os_system
from transform.MLIRImporter import MLIRImporter
from mlir.ir import Location


def linear(x, W, b=None):
    y = x @ W
    if b is not None:
        y += b
    return y


def qdq_weight(W, perchannel=True):
    if perchannel:
        mxw = np.abs(W).max(axis=1, keepdims=True)
    else:
        mxw = np.abs(W).max()
    s = mxw / 127
    qdqW = np.clip(np.round(W / s), -127, 127) * s
    return qdqW


def qdq_act(X, mxa):
    s = mxa / 127
    qdqX = np.clip(np.round(X / s), -127, 127) * s
    return qdqX


def mse(x, y):
    return np.mean((x - y)**2)


class SmoothQuant(ModelModifier):

    def insert_mul_after_ln(self):
        need_mul_lns = []  # layernorms that need to insert mul for smooth
        matmul2ln = {}  # matmul to layernorm mapping, used to replace matmul input
        for op in self.parser.get_op_name_list():
            op_type = self.parser.get_op_type_by_op_name(op)
            if op_type == 'top.LayerNorm':
                users = self.skip_reshape_like_users(op)
                users_type = [self.parser.get_op_type_by_op_name(user) for user in users]
                if set(users_type) == {'top.Add', 'top.MatMul'}:
                    need_mul_lns.append(op)
                    for ut, u in zip(users_type, users):
                        if ut == 'top.MatMul':
                            matmul2ln[u] = op
        if len(need_mul_lns) == 0: return

        input_shapes = [list(self.input_tensors[0][n].shape) for n in self.module.input_names]
        output_shapes = [[] for n in self.module.output_names]
        input_types = []
        for n in self.module.input_names:
            if self.input_tensors[0][n].dtype == np.float32:
                input_types.append('F32')
            elif self.input_tensors[0][n].dtype == np.int32:
                input_types.append('INT32')
            else:
                raise TypeError(
                    f'input_name: {n}, sunknown input data type: {self.input_tensors[0][n]}')

        new_mlir = MLIRImporter(input_shapes,
                                output_shapes,
                                self.parser.module_name,
                                "ONNX",
                                input_types,
                                run_mode='STATIC',
                                weight_file=self.parser.module_weight_file)
        self.unranked_type = new_mlir.get_tensor_type([])

        idx2operand = {0: new_mlir.none_op}
        idx2name = {0: ''}
        # create input ops
        for i, name in enumerate(self.module.input_names):
            op = self.parser.body.operations[i + 1]
            attrs = op.attributes
            name = Operation.name(op)
            kwargs = {}
            for a in attrs:
                if hasattr(a.attr, 'value'):
                    kwargs[a.name] = a.attr.value
                else:
                    kwargs[a.name] = [_.value for _ in a.attr]
            input_ = new_mlir.create_input_op(
                Location.fused([Location.name(name)], context=new_mlir.ctx), i, kwargs)
            idx2operand[i + 1] = input_
            idx2name[i + 1] = name

        ln2mulout = {}  # layernorm to mul output mapping, used to replace matmul input
        return_idx = []
        # insert other ops
        for i in range(len(self.module.input_names) + 1, len(self.parser.body.operations) - 1):
            op = self.parser.body.operations[i]
            op_type = Operation.type(op)
            attrs = op.attributes
            name = Operation.name(op)
            kwargs = {a.name: a.attr for a in attrs}
            kwargs['ip'] = new_mlir.insert_point
            kwargs['loc'] = Location.fused([Location.name(name)], context=new_mlir.ctx)
            pre_op_ids = [int(_.get_name().strip('%')) for _ in op.operands]
            args = [idx2operand[j] for j in pre_op_ids]
            if name in matmul2ln:  # replace matmul input with inserted mul output
                args[0] = ln2mulout[matmul2ln[name]]

            if op_type == 'top.Weight':
                weight_shape = Operation.shape(op)
                new_out = new_mlir.create_weight_op(name, weight_shape)
                self.module_weights[name] = self.module_weights[name].reshape(weight_shape)
            else:
                new_out = self.insert_origin_op(op_type, args, kwargs)
            if new_out is None: return  # fail to insert, stop inserting mul

            if name in need_mul_lns:  # insert mul after layernorm
                mul_shape = self.module_weights[idx2name[pre_op_ids[1]]].shape
                mulW_name = name + "_scaled_weight"
                self.module_weights[mulW_name] = np.ones(mul_shape).astype(np.float32)
                mulW_out = new_mlir.create_weight_op(mulW_name, mul_shape)
                mul_name = "Mul_after_" + name
                mul_out = top.MulOp(self.unranked_type, [new_out, mulW_out],
                                    loc=Location.fused([Location.name(mul_name)],
                                                       context=new_mlir.ctx),
                                    ip=new_mlir.insert_point).output
                ln2mulout[name] = mul_out

            idx2operand[i] = new_out
            idx2name[i] = name
            if name in self.parser.get_output_op_names_n_shapes():
                return_idx.append(i)
        # create return op
        return_op = []
        for i in return_idx:
            return_op.append(idx2operand[i])
        new_mlir.create_return_op(return_op)
        # update mlir file
        new_mlir_txt = new_mlir.print_module()
        with open('tmp.mlir', 'w') as f:
            f.write(new_mlir_txt)
        np.savez(self.parser.module_weight_file, **self.module_weights)
        cmd = ['tpuc-opt', 'tmp.mlir', '--shape-infer', '-o', self.args.mlir_file]
        _os_system(cmd, log_level="normal")
        os.remove('tmp.mlir')
        # update module and parser
        self.module = pymlir.module()
        self.module.load(self.args.mlir_file)
        self.parser = MlirParser(self.args.mlir_file)

    def get_smooth_op_names(self):
        self.smooth_op_names = []
        self.all_op_names = self.parser.get_op_name_list()
        for op in self.all_op_names:
            op_type = self.parser.get_op_type_by_op_name(op)
            if op_type == 'top.LayerNorm':
                users = self.skip_reshape_like_users(op)
                for user in users:
                    if self.parser.get_op_type_by_op_name(user) != 'top.MatMul':
                        break
                else:  # all user is top.MatMul
                    self.smooth_op_names.append(op)
            elif op_type == 'top.Mul':
                mul_op = self.parser.get_op_by_op_name(op)
                if mul_op.opds[1] not in self.module_weights:
                    continue  # opds[1] is not weight op
                if self.module_weights[mul_op.opds[1]].shape[-1] != mul_op.shape[-1]:
                    continue  # weight shape is different from act shape
                users = self.skip_reshape_like_users(op)
                for user in users:
                    if self.parser.get_op_type_by_op_name(user) != 'top.MatMul':
                        break
                else:  # all user is top.MatMul
                    self.smooth_op_names.append(op)
            elif self.is_vproj_matmul(
                    op):  # smooth between v proj and o proj, now only support bert model
                self.smooth_op_names.append(op)
        for i, op_name in enumerate(self.smooth_op_names):
            print(f"smooth op {i} {op_name}")

    def skip_reshape_like_users(self, op):
        candidates = self.parser.get_next_op_by_op_name(op)
        candidates = sorted(candidates,
                            key=lambda x: self.parser.get_op_type_by_op_name(x) != 'top.Concat')
        seen = set()
        users = []

        def add_candidates(users):
            for user in users:
                if user in candidates: continue
                if self.parser.get_op_type_by_op_name(user) == 'top.Concat':
                    candidates.insert(0, user)
                else:
                    candidates.append(user)

        while candidates:
            user = candidates.pop()
            user_op = self.parser.get_op_by_op_name(user)
            user_type = self.parser.get_op_type_by_op_name(user)
            if user_type in ['top.Slice', 'top.Squeeze', 'top.Reshape', 'top.Reduce']:
                input_shape = self.module.get_tensor(user_op.opds[0]).shape
                output_shape = self.module.get_tensor(user).shape
                if input_shape[-1] == output_shape[-1]:
                    users2 = self.parser.get_next_op_by_op_name(user)
                    add_candidates(users2)
                else:
                    users.append(user)
            elif user_type == 'top.Permute':
                order = eval(user_op.attrs['order'])
                if len(order) == order[-1] + 1:
                    users2 = self.parser.get_next_op_by_op_name(user)
                    add_candidates(users2)
                else:
                    users.append(user)
            elif user_type == 'top.Concat':
                input_shape = self.module.get_tensor(user_op.opds[0]).shape
                output_shape = self.module.get_tensor(user).shape
                if all(opd in seen for opd in user_op.opds) and input_shape[-1] == output_shape[-1]:
                    users2 = self.parser.get_next_op_by_op_name(user)
                    add_candidates(users2)
                else:
                    users.append(user)
            else:
                users.append(user)
            seen.add(user)
        users = list(set(users))
        return users

    def is_vproj_matmul(self, op_name):
        if self.parser.get_op_type_by_op_name(op_name) != 'top.MatMul':
            return False

        first_matmul_users = []
        users = self.parser.get_next_op_by_op_name(op_name)
        while True:
            if len(users) != 1:
                return False
            if self.parser.get_op_type_by_op_name(users[0]) == 'top.MatMul':
                break
            first_matmul_users.append(users[0])
            users = self.parser.get_next_op_by_op_name(users[0])
        for user_op in first_matmul_users:
            if self.parser.get_op_type_by_op_name(user_op) not in ['top.Reshape', 'top.Permute']:
                return False

        second_matmul_users = []
        users = self.parser.get_next_op_by_op_name(users[0])
        while True:
            if len(users) != 1:
                return False
            if self.parser.get_op_type_by_op_name(users[0]) == 'top.MatMul':
                break
            second_matmul_users.append(users[0])
            users = self.parser.get_next_op_by_op_name(users[0])
        for user_op in second_matmul_users:
            if self.parser.get_op_type_by_op_name(user_op) not in ['top.Reshape', 'top.Permute']:
                return False

        matmul_op = self.parser.get_op_by_op_name(op_name)
        first_matmul_user_ops = [self.parser.get_op_by_op_name(_) for _ in first_matmul_users]
        second_matmul_user_ops = [self.parser.get_op_by_op_name(_) for _ in second_matmul_users]
        if matmul_op.shape != second_matmul_user_ops[-1].shape:
            return False
        for first_op in first_matmul_user_ops:
            second_op = second_matmul_user_ops.pop()
            if first_op.type != second_op.type:
                return False
            second_input_shape = self.parser.get_op_by_op_name(second_op.opds[0]).shape
            if second_input_shape != first_op.shape:
                return False
        return True

    def get_next_matmul_ops(self, op_name):
        next_ops = self.parser.get_next_op_by_op_name(op_name)
        next_matmuls = []
        while next_ops:
            next_op = next_ops.pop()
            if self.parser.get_op_type_by_op_name(next_op) == 'top.MatMul':
                next_matmuls.append(next_op)
            else:
                next_ops.extend(self.parser.get_next_op_by_op_name(next_op))
        return next_matmuls

    def collect_activation_scales_by_op_name(self, op_name):
        outputs = self.parser.get_outputs_by_op_name(op_name)
        for output in outputs:
            if output not in self.all_op_names:
                continue
            activation = self.module.get_tensor(output).copy()
            if op_name not in self.activation_collection:
                self.activation_collection[op_name] = []
            self.activation_collection[op_name].append(activation)
            if activation is None:
                continue
            activation = activation.reshape(-1, activation.shape[-1])
            scales = np.abs(activation).max(axis=0)
            if output not in self.activation_scales:
                self.activation_scales[output] = scales
            else:
                self.activation_scales[output] = np.maximum(scales, self.activation_scales[output])

    def collect_activation_scales(self):
        self.activation_scales = {}
        self.activation_collection = {}
        pbar = tqdm([i for i in range(self.args.input_num)],
                    total=self.args.input_num,
                    position=0,
                    leave=True)
        for idx in range(self.args.input_num):
            pbar.set_description(f"SmoothQuant for sample {idx}")
            for name in self.module.input_names:
                input_tensor = self.input_tensors[idx][name]
                self.module.set_tensor(name, input_tensor, input_tensor.shape)
            self.module.invoke()
            for op_name in self.smooth_op_names:
                self.collect_activation_scales_by_op_name(op_name)
            pbar.update(1)
        pbar.close()

    def collect_weight_scales(self):
        self.weight_scales = {}
        for op_name in self.smooth_op_names:
            op_type = self.parser.get_op_type_by_op_name(op_name)
            if op_type in ['top.LayerNorm', 'top.Mul']:
                users = self.skip_reshape_like_users(op_name)
                for user in users:
                    assert self.parser.get_op_type_by_op_name(user) == 'top.MatMul'
                    user_op = self.parser.get_op_by_op_name(user)
                    user_weight = self.module_weights[user_op.opds[1]]
                    if user_weight.ndim == 1:
                        user_weight = user_weight.reshape(-1, user_op.shape[-1])
                    user_weight_scale = np.abs(user_weight).max(axis=1)
                    if op_name not in self.weight_scales:
                        self.weight_scales[op_name] = user_weight_scale
                    else:
                        self.weight_scales[op_name] = np.maximum(user_weight_scale,
                                                                 self.weight_scales[op_name])

            elif op_type == 'top.MatMul':
                users = self.get_next_matmul_ops(op_name)
                oproj = self.get_next_matmul_ops(users[0])[0]
                oproj_op = self.parser.get_op_by_op_name(oproj)
                oproj_weight = self.module_weights[oproj_op.opds[1]]
                if oproj_weight.ndim == 1:
                    oproj_weight = oproj_weight.reshape(-1, oproj_op.shape[-1])
                oproj_weight_scale = np.abs(oproj_weight).max(axis=1)
                if op_name not in self.weight_scales:
                    self.weight_scales[op_name] = oproj_weight_scale
                else:
                    self.weight_scales[op_name] = np.maximum(oproj_weight_scale,
                                                             self.weight_scales[op_name])

    def search_opt_scale(self, op_name):
        # search the best smooth scale by minimizing the mse loss
        op_type = self.parser.get_op_type_by_op_name(op_name)
        weight_scale = self.weight_scales[op_name]
        activation_scale = self.activation_scales[op_name]
        inputs = self.activation_collection[op_name]
        opt_loss = float('inf')
        opt_scale = None

        if op_type in ['top.LayerNorm', 'top.Mul']:
            users = self.skip_reshape_like_users(op_name)
        elif op_type == 'top.MatMul':
            users = self.get_next_matmul_ops(op_name)
            users = self.get_next_matmul_ops(users[0])

        fp32_outputs = {user: [] for user in users}
        user_weights = {}
        for user in users:
            user_opds = self.parser.get_opds_by_op_name(user)
            ori_weight = self.module_weights[user_opds[1]].copy()
            if ori_weight.ndim == 1:
                ori_weight = ori_weight.reshape(inputs[0].shape[-1], -1)
            user_weights[user] = ori_weight
            for x in inputs:
                fp32_outputs[user].append(linear(x, ori_weight))

        # smoothquant search space
        for alpha in np.arange(0, 1.05, 0.05):
            loss = 0
            smooth_scale = activation_scale**alpha / weight_scale**(1 - alpha)
            smooth_scale[np.isinf(smooth_scale)] = 1
            smooth_scale[np.isnan(smooth_scale)] = 1
            scaled_mxa = (activation_scale / smooth_scale).max()
            scaled_inputs = []
            for x in inputs:
                scaled_inputs.append(qdq_act(x / smooth_scale, scaled_mxa))
            for user in users:
                scaled_weight = user_weights[user] * smooth_scale.reshape(-1, 1)
                scaled_weight = qdq_weight(scaled_weight, )
                for i, scaled_x in enumerate(scaled_inputs):
                    scaled_output = linear(scaled_x, scaled_weight)
                    loss += mse(scaled_output, fp32_outputs[user][i])
            if loss < opt_loss:
                opt_loss = loss
                opt_scale = smooth_scale

        # os+ search space
        max_step = 20
        amx = activation_scale.max()
        for step in range(max_step):
            loss = 0
            mx_range = amx * (1 + step) / max_step
            smooth_scale = np.maximum(1.0, activation_scale / mx_range)
            smooth_scale[np.isinf(smooth_scale)] = 1
            smooth_scale[np.isnan(smooth_scale)] = 1
            scaled_mxa = (activation_scale / smooth_scale).max()
            scaled_inputs = []
            for x in inputs:
                scaled_inputs.append(qdq_act(x / smooth_scale, scaled_mxa))
            for user in users:
                scaled_weight = user_weights[user] * smooth_scale.reshape(-1, 1)
                scaled_weight = qdq_weight(scaled_weight, )
                for i, scaled_x in enumerate(scaled_inputs):
                    scaled_output = linear(scaled_x, scaled_weight)
                    loss += mse(scaled_output, fp32_outputs[user][i])
            if loss < opt_loss:
                opt_loss = loss
                opt_scale = smooth_scale

        return opt_scale

    def smoothquant(self):
        for op_name in tqdm(self.smooth_op_names):
            if op_name in self.weight_scales and op_name in self.activation_scales:
                op_type = self.parser.get_op_type_by_op_name(op_name)
                opds = self.parser.get_opds_by_op_name(op_name)
                smooth_scale = self.search_opt_scale(op_name)

                ori_weight_shape = self.module_weights[opds[1]].shape
                self.module_weights[opds[1]] /= smooth_scale
                new_weight_shape = self.module_weights[opds[1]].shape
                assert np.all(ori_weight_shape == new_weight_shape
                              ), f"{op_name} weight {ori_weight_shape} {new_weight_shape}"
                if len(opds) > 2:
                    ori_weight_shape = self.module_weights[opds[2]].shape
                    self.module_weights[opds[2]] /= smooth_scale
                    new_weight_shape = self.module_weights[opds[2]].shape
                    assert np.all(ori_weight_shape == new_weight_shape
                                  ), f"{op_name} bias {ori_weight_shape} {new_weight_shape}"
                smooth_scale = smooth_scale.reshape(-1, 1)

                if op_type in ['top.LayerNorm', 'top.Mul']:
                    users = self.skip_reshape_like_users(op_name)
                elif op_type == 'top.MatMul':
                    users = self.get_next_matmul_ops(op_name)
                    users = self.get_next_matmul_ops(users[0])
                else:
                    users = []

                for user in users:
                    assert self.parser.get_op_type_by_op_name(user) == 'top.MatMul'
                    user_opds = self.parser.get_opds_by_op_name(user)
                    ori_weight_shape = self.module_weights[user_opds[1]].shape
                    if self.module_weights[user_opds[1]].ndim == 1:
                        repeat_num = self.module_weights[
                            user_opds[1]].shape[0] // smooth_scale.shape[0]
                        tmp_scale = np.repeat(smooth_scale, repeat_num, axis=1).flatten()
                        self.module_weights[user_opds[1]] *= tmp_scale
                    else:
                        self.module_weights[user_opds[1]] *= smooth_scale
                    new_weight_shape = self.module_weights[user_opds[1]].shape
                    assert np.all(ori_weight_shape == new_weight_shape
                                  ), f"{op_name} {user} {ori_weight_shape} {new_weight_shape}"
        print(f"save weights to {self.parser.module_weight_file}")
        os.system(f'rm -f {self.parser.module_weight_file}')
        np.savez(self.parser.module_weight_file, **self.module_weights)

    def run(self):
        if self.data_selector.all_image:
            self.init_ppa()
        self.load_net_weights()
        self.load_net_inputs()
        self.insert_mul_after_ln()
        self.get_smooth_op_names()
        if len(self.smooth_op_names):
            self.collect_weight_scales()
            self.collect_activation_scales()
            self.smoothquant()