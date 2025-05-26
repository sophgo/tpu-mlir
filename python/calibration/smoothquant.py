import os
import pymlir
import numpy as np
from tqdm import tqdm
from calibration.data_selector import DataSelector
from utils.mlir_parser import MlirParser
from utils.preprocess import preprocess

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
    return np.mean((x - y) ** 2)

class SmoothQuant:
    def __init__(self, args, data_selector: DataSelector):
        self.args = args
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.data_selector = data_selector
        self.data_list = data_selector.data_list
        self.args.input_num = len(self.data_list)
        if data_selector.all_image:
            n = self.args.input_num % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[i])
                    self.args.input_num += 1
            self.args.input_num = self.args.input_num // self.batch_size

    def init_ppa(self):
        self.ppa_list = []
        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.parser.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)

    def load_net_weights(self):
        weight_file = self.parser.module_weight_file
        print('load weights from {}'.format(weight_file))
        weights = np.load(weight_file)
        self.module_weights = {k: weights[k] for k in weights}

    def load_net_inputs(self):
        self.input_tensors = {}

        if self.data_selector.all_image:
            batched_inputs = [[] for i in range(self.input_num)]
        else:
            batched_inputs = {}
        only_one = len(self.module.input_names) == 1
        for data_idx, data in enumerate(self.data_list):
            if self.data_selector.all_npz:
                x = np.load(data)
                batch_idx = len(self.input_tensors)
                if batch_idx not in self.input_tensors:
                    self.input_tensors[batch_idx] = {}
                if only_one:
                    assert (len(x.files) == 1)
                    n0 = self.module.input_names[0]
                    n1 = x.files[0]
                    if n1 in batched_inputs:
                        batched_inputs[n1] = np.concatenate(
                            [batched_inputs[n1], x[n1].astype(np.float32)],
                            axis=0
                        )
                    else:
                        batched_inputs[n1] = x[n1].astype(np.float32)
                    if batched_inputs[n1].shape[0] >= self.batch_size:
                        self.input_tensors[batch_idx][n0] = batched_inputs[n1][:self.batch_size]
                        batched_inputs[n1] = batched_inputs[n1][self.batch_size:]

                else:
                    for input in self.module.input_names:
                        assert (input in x)
                        if input in batched_inputs:
                            batched_inputs[input] = np.concatenate(
                                [batched_inputs[input], x[input].astype(np.float32)],
                                axis=0
                            )
                        else:
                            batched_inputs[input] = x[input].astype(np.float32)
                        real_batch_size = self.parser.get_op_by_op_name(input).shape[0]
                        self.input_tensors[batch_idx][input] = batched_inputs[input][:real_batch_size]
                        batched_inputs[input] = batched_inputs[input][real_batch_size:]

            elif self.data_selector.all_image:
                inputs = [s.strip() for s in data.split(',')]
                assert (self.input_num == len(inputs))
                for i in range(self.input_num):
                    batched_inputs[i].append(inputs[i])
                if (data_idx + 1) % self.batch_size == 0:
                    batch_idx = (data_idx + 1) // self.batch_size - 1
                    self.input_tensors[batch_idx] = {}
                    for i in range(self.input_num):
                        x = self.ppa_list[i].run(','.join(batched_inputs[i]))
                        name = self.ppa_list[i].input_name
                        self.input_tensors[batch_idx][name] = x
                        batched_inputs = [[] for i in range(self.input_num)]

            else:
                self.input_tensors[data_idx] = {}
                inputs = [s.strip() for s in data.split(',')]
                assert (self.input_num == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    x = np.load(input)
                    self.input_tensors[data_idx][name] = x

        self.args.input_num = min(self.args.input_num, len(self.input_tensors))
        print(f"input_num = {self.args.input_num}, ref = {len(self.input_tensors)}")
        print(f"real input_num = {self.args.input_num}")
        assert self.args.input_num > 0

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
                else: # all user is top.MatMul
                    self.smooth_op_names.append(op)
            elif op_type == 'top.Mul':
                mul_op = self.parser.get_op_by_op_name(op)
                if mul_op.opds[1] not in self.module_weights:
                    continue # opds[1] is not weight op
                if self.module_weights[mul_op.opds[1]].shape[-1] != mul_op.shape[-1]:
                    continue # weight shape is different from act shape
                users = self.skip_reshape_like_users(op)
                for user in users:
                    if self.parser.get_op_type_by_op_name(user) != 'top.MatMul':
                        break
                else: # all user is top.MatMul
                    self.smooth_op_names.append(op)
            elif self.is_vproj_matmul(op): # smooth between v proj and o proj, now only support bert model
                self.smooth_op_names.append(op)
        for i, op_name in enumerate(self.smooth_op_names):
            print(f"smooth op {i} {op_name}")

    def skip_reshape_like_users(self, op):
        candidates = self.parser.get_next_op_by_op_name(op)
        candidates = sorted(
            candidates,
            key=lambda x: self.parser.get_op_type_by_op_name(x) != 'top.Concat'
        )
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
                self.activation_scales[output] = np.maximum(
                    scales,
                    self.activation_scales[output]
                )

    def collect_activation_scales(self):
        self.activation_scales = {}
        self.activation_collection = {}
        pbar = tqdm([i for i in range(self.args.input_num)],
                    total=self.args.input_num, position=0, leave=True)
        for idx in range(self.args.input_num):
            pbar.set_description(f"collect activations for sample {idx}")
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
                        self.weight_scales[op_name] = np.maximum(
                            user_weight_scale,
                            self.weight_scales[op_name]
                        )

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
                    self.weight_scales[op_name] = np.maximum(
                        oproj_weight_scale,
                        self.weight_scales[op_name]
                    )

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
            smooth_scale = activation_scale ** alpha / weight_scale ** (1 - alpha)
            smooth_scale[np.isinf(smooth_scale)] = 1
            smooth_scale[np.isnan(smooth_scale)] = 1
            scaled_mxa = (activation_scale / smooth_scale).max()
            scaled_inputs = []
            for x in inputs:
                scaled_inputs.append(qdq_act(x / smooth_scale, scaled_mxa))
            for user in users:
                scaled_weight = user_weights[user] * smooth_scale.reshape(-1, 1)
                scaled_weight = qdq_weight(
                    scaled_weight,
                )
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
                scaled_weight = qdq_weight(
                    scaled_weight,
                )
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
                assert np.all(ori_weight_shape == new_weight_shape), f"{op_name} weight {ori_weight_shape} {new_weight_shape}"
                if len(opds) > 2:
                    ori_weight_shape = self.module_weights[opds[2]].shape
                    self.module_weights[opds[2]] /= smooth_scale
                    new_weight_shape = self.module_weights[opds[2]].shape
                    assert np.all(ori_weight_shape == new_weight_shape), f"{op_name} bias {ori_weight_shape} {new_weight_shape}"
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
                        repeat_num = self.module_weights[user_opds[1]].shape[0] // smooth_scale.shape[0]
                        tmp_scale = np.repeat(smooth_scale, repeat_num, axis=1).flatten()
                        self.module_weights[user_opds[1]] *= tmp_scale
                    else:
                        self.module_weights[user_opds[1]] *= smooth_scale
                    new_weight_shape = self.module_weights[user_opds[1]].shape
                    assert np.all(ori_weight_shape == new_weight_shape), f"{op_name} {user} {ori_weight_shape} {new_weight_shape}"
        print(f"save weights to {self.parser.module_weight_file}")
        os.system(f'rm -f {self.parser.module_weight_file}')
        np.savez(self.parser.module_weight_file, **self.module_weights)

    def run(self):
        if self.data_selector.all_image:
            self.init_ppa()
        self.load_net_weights()
        self.load_net_inputs()
        self.get_smooth_op_names()
        if len(self.smooth_op_names):
            self.collect_weight_scales()
            self.collect_activation_scales()
            self.smoothquant()
