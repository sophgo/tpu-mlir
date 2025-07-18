# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from .LlmConverter import *
from typing_extensions import override


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(self,
                     seq_len: int,
                     n_elem: int,
                     dtype: torch.dtype,
                     device: torch.device,
                     base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base
                       **(torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len):
        return self.forward_impl(max_seq_len,
                                 self.dim,
                                 dtype=self.inv_freq.dtype,
                                 device=self.inv_freq.device)


# support chatglm
class Chatglm3Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)

    @override
    def rotary_embedding(self):
        rotary_dim = (self.head_dim
                      if self.llm_config.kv_channels is None else self.llm_config.kv_channels)
        rotary_embedding = RotaryEmbedding(rotary_dim // 2,
                                           original_impl=True,
                                           device="cpu",
                                           dtype=torch.float32)
        rotary_pos_emb = rotary_embedding(self.seq_length)
        cos, sin = torch.split(rotary_pos_emb, 1, dim=-1)
        cos = cos.squeeze(-1).unsqueeze(-1)
        sin = sin.squeeze(-1).unsqueeze(-1)
        self.rot_dim = cos.shape[-2]
        return cos, sin

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = CHATGLM3_INFO

    @override
    def rotary_pos(self, mlir_gen, in_op, cos_op, sin_op, out_name: str):
        in_shape = in_op.type.shape
        prefix = f"{out_name}.rotary_pos"
        rot_dim = self.rot_dim * 2
        half_shape = list(in_shape)
        half_shape[-1] = rot_dim
        half_shape_s = list(in_shape)
        half_shape_s[-1] = self.rot_dim
        x = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                        in_op,
                        mlir_gen.none_op,
                        mlir_gen.none_op,
                        mlir_gen.none_op,
                        offset=[0, 0, 0, 0],
                        steps=[1, 1, 1, 1],
                        ends=half_shape,
                        axes=[],
                        loc=self.get_loc(prefix + ".slicex", mlir_gen),
                        ip=mlir_gen.insert_point).output
        x_pass = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                             in_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             offset=[0, 0, 0, half_shape[-1]],
                             steps=[1, 1, 1, 1],
                             ends=in_shape,
                             axes=[],
                             loc=self.get_loc(prefix + ".slicep", mlir_gen),
                             ip=mlir_gen.insert_point).output
        x_shape_cal = [
            half_shape[0], half_shape[1] * half_shape[3] // rot_dim, half_shape[2], self.rot_dim, 2
        ]
        x_split_shape = [
            x_shape_cal[0], x_shape_cal[1], x_shape_cal[2], x_shape_cal[3], x_shape_cal[4] // 2
        ]
        x_reshape = top.ReshapeOp(mlir_gen.get_tensor_type(x_shape_cal),
                                  x,
                                  loc=self.get_loc(prefix + "_x.reshpae", mlir_gen),
                                  ip=mlir_gen.insert_point).output
        x0 = top.SliceOp(mlir_gen.get_tensor_type(x_split_shape),
                         x_reshape,
                         mlir_gen.none_op,
                         mlir_gen.none_op,
                         mlir_gen.none_op,
                         offset=[0, 0, 0, 0, 0],
                         steps=[1, 1, 1, 1, 1],
                         ends=x_split_shape,
                         axes=[],
                         loc=self.get_loc(prefix + ".slice0", mlir_gen),
                         ip=mlir_gen.insert_point).output
        x1 = top.SliceOp(mlir_gen.get_tensor_type(x_split_shape),
                         x_reshape,
                         mlir_gen.none_op,
                         mlir_gen.none_op,
                         mlir_gen.none_op,
                         offset=[0, 0, 0, 0, 1],
                         steps=[1, 1, 1, 1, 1],
                         ends=x_shape_cal,
                         axes=[],
                         loc=self.get_loc(prefix + ".slice1", mlir_gen),
                         ip=mlir_gen.insert_point).output
        x0 = top.SqueezeOp(mlir_gen.get_tensor_type(half_shape_s),
                           x0,
                           axes=[-1],
                           loc=self.get_loc(prefix + ".squeeze0", mlir_gen),
                           ip=mlir_gen.insert_point).output
        x1 = top.SqueezeOp(mlir_gen.get_tensor_type(half_shape_s),
                           x1,
                           axes=[-1],
                           loc=self.get_loc(prefix + ".squeeze1", mlir_gen),
                           ip=mlir_gen.insert_point).output
        x0_cos = top.MulOp(mlir_gen.get_tensor_type(half_shape_s), [x0, cos_op],
                           loc=self.get_loc(prefix + ".mul0", mlir_gen),
                           ip=mlir_gen.insert_point).output
        x1_cos = top.MulOp(mlir_gen.get_tensor_type(half_shape_s), [x1, cos_op],
                           loc=self.get_loc(prefix + ".mul1", mlir_gen),
                           ip=mlir_gen.insert_point).output
        x0_sin = top.MulOp(mlir_gen.get_tensor_type(half_shape_s), [x0, sin_op],
                           loc=self.get_loc(prefix + ".mul2", mlir_gen),
                           ip=mlir_gen.insert_point).output
        x1_sin = top.MulOp(mlir_gen.get_tensor_type(half_shape_s), [x1, sin_op],
                           loc=self.get_loc(prefix + ".mul3", mlir_gen),
                           ip=mlir_gen.insert_point).output
        sub = top.SubOp(mlir_gen.get_tensor_type(half_shape_s), [x0_cos, x1_sin],
                        loc=self.get_loc(prefix + ".sub0", mlir_gen),
                        ip=mlir_gen.insert_point).output
        add = top.AddOp(mlir_gen.get_tensor_type(half_shape_s), [x1_cos, x0_sin],
                        loc=self.get_loc(prefix + ".add0", mlir_gen),
                        ip=mlir_gen.insert_point).output
        sub = top.UnsqueezeOp(mlir_gen.get_tensor_type(half_shape_s + [1]),
                              sub,
                              axes=[-1],
                              loc=self.get_loc(prefix + "sub_unsqueeze", mlir_gen),
                              ip=mlir_gen.insert_point).output
        add = top.UnsqueezeOp(mlir_gen.get_tensor_type(half_shape_s + [1]),
                              add,
                              axes=[-1],
                              loc=self.get_loc(prefix + "add_unsqueeze", mlir_gen),
                              ip=mlir_gen.insert_point).output
        conc_q1 = top.ConcatOp(mlir_gen.get_tensor_type(half_shape_s + [2]), [sub, add],
                               axis=4,
                               loc=self.get_loc(prefix + ".conc0", mlir_gen),
                               ip=mlir_gen.insert_point).output
        conc_reshape = top.ReshapeOp(mlir_gen.get_tensor_type(half_shape),
                                     conc_q1,
                                     loc=self.get_loc(prefix + "_conc.reshpae", mlir_gen),
                                     ip=mlir_gen.insert_point).output
        conc_q2 = top.ConcatOp(mlir_gen.get_tensor_type(in_shape), [conc_reshape, x_pass],
                               axis=3,
                               loc=self.get_loc(out_name, mlir_gen),
                               ip=mlir_gen.insert_point).output

        return conc_q2

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        dim = pos_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight",
                                              [self.seq_length, 1, self.rot_dim])
        cos_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.rot_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_cos, mlir_gen),
                              ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight",
                                              [self.seq_length, 1, self.rot_dim])
        sin_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.rot_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_sin, mlir_gen),
                              ip=mlir_gen.insert_point).output
        # ===== q_proj rotary ========
        q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")

        # ===== k_proj rotary ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")
        return q_op, k_op

    @override
    def set_linear_weight(self, path: str, weight_dict: dict):
        is_quant = self.quant_mode is not None and self.model.is_exist(path + ".qweight")
        if not is_quant:
            weight_path = path
            bias_path = path
            if self.model.is_exist(weight_path + '.weight'):
                data = self.model.read(weight_path + '.weight')
                if 'query_key_value' in weight_path:
                    weight_dict[weight_path + '_q.weight'] = np.ascontiguousarray(
                        np.transpose(data[
                            :self.hidden_size,
                        ], (1, 0)))
                    weight_dict[weight_path + '_k.weight'] = np.ascontiguousarray(
                        np.transpose(data[
                            self.hidden_size:self.hidden_size + self.kv_dim,
                        ], (1, 0)))
                    weight_dict[weight_path + '_v.weight'] = np.ascontiguousarray(
                        np.transpose(data[
                            self.hidden_size + self.kv_dim:,
                        ], (1, 0)))
                elif 'dense_h_to_4h' in weight_path:
                    weight_dict[weight_path + '_gate.weight'] = np.ascontiguousarray(
                        np.transpose(data[
                            :self.intermediate_size,
                        ], (1, 0)))
                    weight_dict[weight_path + '_up.weight'] = np.ascontiguousarray(
                        np.transpose(data[
                            self.intermediate_size:,
                        ], (1, 0)))
                else:
                    weight_dict[weight_path + '.weight'] = np.ascontiguousarray(
                        np.transpose(data, (1, 0)))
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        else:
            qweight_path = path + ".qweight"
            scale_path = path + ".scales"
            zp_path = path + ".qzeros"
            bias_path = path + ".bias"
            if self.model.is_exist(qweight_path):
                qweigth_data = self.model.read(qweight_path)
                scale_data = self.model.read(scale_path)
                zp_data = self.model.read(zp_path)
                unpacked_weights, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                    qweigth_data, zp_data, self.quant_bits, self.quant_mode)
                weight_dict[qweight_path] = np.ascontiguousarray(
                    np.transpose(pack_int8_weights, (1, 0)))
                weight_dict[scale_path] = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))
                weight_dict[zp_path] = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        if self.model.is_exist(bias_path + '.bias'):
            b_data = self.model.read(bias_path + '.bias')
            weight_dict[bias_path + '_q.bias'] = b_data[:self.hidden_size]
            weight_dict[bias_path + '_k.bias'] = b_data[self.hidden_size:self.hidden_size +
                                                        self.kv_dim]
            weight_dict[bias_path + '_v.bias'] = b_data[self.hidden_size + self.kv_dim:]

    @override
    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        qkv_w = TOP_PATH + self.model_info.weights[LlmList.QKV_WB]
        att_dense = TOP_PATH + self.model_info.weights[LlmList.ATT_D]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        mlp_g_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.do_lmhead_merge and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict)
        self.set_linear_weight(qkv_w, weight_dict)
        self.set_linear_weight(att_dense, weight_dict)
        self.set_common_weight(post_attn_ln, weight_dict)
        self.set_linear_weight(mlp_g_up, weight_dict)
        self.set_linear_weight(mlp_down, weight_dict)
        if do_norm:
            self.set_common_weight(norm, weight_dict)

        np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            len = input_shape[1]
            new_op = in_op
            mlp_gate = mlp_g_up + "_gate"
            mlp_up = mlp_g_up + "_up"

            new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            gate_op = self.linear(mlir_gen, mlp_gate, new_op,
                                  [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size])
            if self.hidden_act == "silu":
                act_op = top.SiLUOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                                    gate_op,
                                    loc=self.get_loc(mlp_gate + ".silu", mlir_gen),
                                    ip=ip).output
            elif self.hidden_act == "gelu_pytorch_tanh":
                act_op = top.GELUOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                                    gate_op,
                                    approx_mode=StringAttr.get("tanh"),
                                    loc=self.get_loc(mlp_gate + ".gelu", mlir_gen),
                                    ip=ip).output
            else:
                raise NotImplementedError(f"Unsupported activation type: {self.hidden_act}")

            up_op = self.linear(mlir_gen, mlp_up, new_op,
                                [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size])

            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen, mlp_down, new_op,
                                  [self.intermediate_size, self.hidden_size], input_shape)
            last_name = "output_states"
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        # create block mlir
        def gen_block():
            name = f"block_{idx}"
            input_shape = [1, self.seq_length, self.hidden_size]
            id_shape = list(self.position_shape)
            mask_shape = [1, 1, self.seq_length, self.seq_length]

            q_shape = [1, self.seq_length, self.num_attention_heads, self.head_dim]
            kv_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter([input_shape, id_shape, mask_shape],
                                      [input_shape, kv_shape, kv_shape],
                                      name,
                                      Platform.LLM, ["F32", "INT32", "F32"],
                                      weight_file=weight_file)

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir,
                               qkv_w + '_q',
                               ln_op, [self.hidden_size, q_dim], [1, self.seq_length, q_dim],
                               force_bias=True)
            # k_proj
            k_op = self.linear(block_mlir,
                               qkv_w + '_k',
                               ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim],
                               force_bias=True)

            # v_proj
            v_op = self.linear(block_mlir,
                               qkv_w + '_v',
                               ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim],
                               force_bias=True)

            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(qkv_w + "_q.reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(qkv_w + "_k.reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, self.seq_length, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=self.seq_length,
                                     mk=self.seq_length,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, att_dense, fa_op, [q_dim, self.hidden_size], input_shape)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(att_dense + ".add"),
                             ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [1, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            mask_shape = [1, 1, 1, self.seq_length + 1]
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, 1, self.num_attention_heads, self.head_dim]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]

            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, history_shape, history_shape],
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                weight_file=weight_file)

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir,
                               qkv_w + '_q',
                               ln_op, [self.hidden_size, q_dim], [1, 1, q_dim],
                               force_bias=True)
            # k_proj
            k_op = self.linear(block_mlir,
                               qkv_w + '_k',
                               ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                               force_bias=True)
            # v_proj
            v_op = self.linear(block_mlir,
                               qkv_w + '_v',
                               ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                               force_bias=True)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(qkv_w + "_q.reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(qkv_w + "_k.reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)

            # return_ops.append(k_op)
            # return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in3_op, k_op],
                                axis=1,
                                loc=L(qkv_w + "_k.concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in4_op, v_op],
                                axis=1,
                                loc=L(qkv_w + "_v.concat"),
                                ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, 1, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=1,
                                     mk=self.seq_length + 1,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, att_dense, fa_op, [q_dim, self.hidden_size], input_shape)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(att_dense + ".add"),
                             ip=ip).output
            # ========== slice ===========
            k_op = top.SliceOp(
                T([1, self.seq_length, self.num_key_value_heads, self.head_dim]),
                k_op,
                block_mlir.none_op,
                block_mlir.none_op,
                block_mlir.none_op,
                offset=[0, 1, 0, 0],
                steps=[1, 1, 1, 1],
                ends=[1, self.seq_length + 1, self.num_key_value_heads, self.head_dim],
                axes=[],
                loc=L(TOP_PATH + "slice_k"),
                ip=ip).output
            v_op = top.SliceOp(
                T([1, self.seq_length, self.num_key_value_heads, self.head_dim]),
                v_op,
                block_mlir.none_op,
                block_mlir.none_op,
                block_mlir.none_op,
                offset=[0, 1, 0, 0],
                steps=[1, 1, 1, 1],
                hasparamConvert_axes=[1],
                ends=[1, self.seq_length + 1, self.num_key_value_heads, self.head_dim],
                axes=[],
                loc=L(TOP_PATH + "slice_v"),
                ip=ip).output
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        gen_block()
        gen_block_cache()
