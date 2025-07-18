from .Chatglm3Converter import *
from typing_extensions import override


class Phi3RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (self.base**(torch.arange(
                0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim))
        inv_freq_expanded = self.inv_freq[None, :,
                                          None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded = position_ids[:, None, :].float()
        # # Force float32 since bfloat16 loses precision on long contexts
        # # See https://github.com/huggingface/transformers/pull/29285
        # device_type = x.device.type
        # device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()
        return inv_freq_expanded.float(), inv_freq_expanded.float(
        )  # cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# support chatglm
class Phi3Converter(Chatglm3Converter):

    def __init__(self, args, config):
        super().__init__(args, config)

    @override
    def rotary_embedding(self):
        rotary_embedding = Phi3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.llm_config.max_position_embeddings,
            base=self.rope_theta)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(1, self.seq_length)
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        inv_freq = rotary_embedding(x, position_ids)
        return inv_freq

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = PHI3_INFO

    @override
    def linear(self,
               mlir_gen,
               proj: str,
               input_op,
               weight_shape: list,
               out_shape: list,
               force_bias: bool = False):
        if 'qkv_proj' in proj or 'gate_up_proj' in proj:
            proj_i = proj.rsplit('_', 1)[0]
        else:
            proj_i = proj
        if self.model.is_exist(proj_i + ".bias") or force_bias:
            bias_shape = [1] * (len(out_shape) - 1) + [out_shape[-1]]
            bias_op = mlir_gen.create_weight_op(proj + ".bias", bias_shape)
        else:
            bias_op = mlir_gen.none_op
        if self.quant_mode and self.model.is_exist(proj_i + ".qweight"):
            qweight_op = mlir_gen.create_weight_op(
                proj + ".qweight", [weight_shape[1], weight_shape[0] // (8 // self.quant_bits)],
                'UINT8')
            scale_shape = [weight_shape[1], weight_shape[0] //
                           self.q_group_size] if self.q_group_size > 0 else [weight_shape[1], 1]
            scale_op = mlir_gen.create_weight_op(proj + ".scales", scale_shape)
            zp_op = mlir_gen.create_weight_op(proj + ".qzeros", scale_shape, 'UINT8')
            return top.A16MatMulOp(mlir_gen.get_tensor_type(out_shape),
                                   input_op,
                                   qweight_op,
                                   scale_op,
                                   zp_op,
                                   bias_op,
                                   right_transpose=True,
                                   q_group_size=self.q_group_size,
                                   weight_bits=self.quant_bits,
                                   loc=self.get_loc(proj, mlir_gen),
                                   ip=mlir_gen.insert_point).output

        weight_op = mlir_gen.create_weight_op(proj + ".weight", weight_shape)
        return top.MatMulOp(mlir_gen.get_tensor_type(out_shape),
                            input_op,
                            weight_op,
                            bias_op,
                            do_relu=False,
                            loc=self.get_loc(proj, mlir_gen),
                            ip=mlir_gen.insert_point).output

    @override
    def rotary_pos(self, mlir_gen, in_op, cos_op, sin_op, out_name: str):
        in_shape = in_op.type.shape
        prefix = f"{out_name}.rotary_pos"
        half_shape = list(in_shape)
        half_shape[-1] = half_shape[-1] // 2
        mul_q_proj = top.MulOp(mlir_gen.get_tensor_type(in_shape), [in_op, cos_op],
                               loc=self.get_loc(prefix + ".mul0", mlir_gen),
                               ip=mlir_gen.insert_point).output
        half_q0 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                              in_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, 0],
                              steps=[1, 1, 1, 1],
                              ends=half_shape,
                              axes=[],
                              loc=self.get_loc(prefix + ".slice1", mlir_gen),
                              ip=mlir_gen.insert_point).output

        half_q1 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                              in_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, half_shape[-1]],
                              steps=[1, 1, 1, 1],
                              ends=in_shape,
                              axes=[],
                              loc=self.get_loc(prefix + ".slice2", mlir_gen),
                              ip=mlir_gen.insert_point).output

        neg_half_q1 = top.MulConstOp(mlir_gen.get_tensor_type(half_shape),
                                     half_q1,
                                     const_val=-1.0,
                                     loc=self.get_loc(prefix + ".neg3", mlir_gen),
                                     ip=mlir_gen.insert_point).output
        new_q = top.ConcatOp(mlir_gen.get_tensor_type(in_shape), [neg_half_q1, half_q0],
                             axis=3,
                             loc=self.get_loc(prefix + ".concat4", mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_q = top.MulOp(mlir_gen.get_tensor_type(in_shape), [new_q, sin_op],
                          loc=self.get_loc(prefix + ".mul5", mlir_gen),
                          ip=mlir_gen.insert_point).output
        new_q = top.AddOp(mlir_gen.get_tensor_type(in_shape), [mul_q_proj, new_q],
                          loc=self.get_loc(out_name, mlir_gen),
                          ip=mlir_gen.insert_point).output
        return new_q

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_inv_freq: str):
        dim = pos_op.type.shape[-1]

        pos_unsq = top.UnsqueezeOp(mlir_gen.get_tensor_type([1, 1, dim]),
                                   pos_op,
                                   axes=[1],
                                   loc=self.get_loc("rotary_pos_unsqueeze", mlir_gen),
                                   ip=mlir_gen.insert_point).output
        w_op = mlir_gen.create_weight_op(rotary_inv_freq + ".weight", [1, self.head_dim // 2, 1])
        mm_op = top.MatMulOp(mlir_gen.get_tensor_type([1, self.head_dim // 2, dim]),
                             w_op,
                             pos_unsq,
                             mlir_gen.none_op,
                             do_relu=False,
                             loc=self.get_loc("rotary_mm", mlir_gen),
                             ip=mlir_gen.insert_point).output
        mm_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, self.head_dim // 2]),
                              mm_op,
                              order=[0, 2, 1],
                              loc=self.get_loc("rotary_mm_permute", mlir_gen),
                              ip=mlir_gen.insert_point).output
        concop = top.ConcatOp(mlir_gen.get_tensor_type([1, dim, self.head_dim]), [mm_op, mm_op],
                              axis=2,
                              loc=self.get_loc("rotary_pos_concat", mlir_gen),
                              ip=mlir_gen.insert_point).output
        cos_op = top.CosOp(mlir_gen.get_tensor_type([1, dim, self.head_dim]),
                           concop,
                           loc=self.get_loc("rotary_pos_cos", mlir_gen),
                           ip=mlir_gen.insert_point).output
        sin_op = top.SinOp(mlir_gen.get_tensor_type([1, dim, self.head_dim]),
                           concop,
                           loc=self.get_loc("rotary_pos_sin", mlir_gen),
                           ip=mlir_gen.insert_point).output
        cos_op = top.UnsqueezeOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                                 cos_op,
                                 axes=[2],
                                 loc=self.get_loc("rotary_pos_cos_unsq", mlir_gen),
                                 ip=mlir_gen.insert_point).output
        sin_op = top.UnsqueezeOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                                 sin_op,
                                 axes=[2],
                                 loc=self.get_loc("rotary_pos_sin_unsq", mlir_gen),
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
                if 'qkv_proj' in weight_path:
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
                elif 'gate_up_proj' in weight_path:
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

                if 'qkv_proj' in qweight_path:
                    weight_dict[path + '_q.qweight'] = np.ascontiguousarray(
                        np.transpose(pack_int8_weights[:, :self.hidden_size], (1, 0)))
                    weight_dict[path + '_k.qweight'] = np.ascontiguousarray(
                        np.transpose(
                            pack_int8_weights[:, self.hidden_size:self.hidden_size + self.kv_dim],
                            (1, 0)))
                    weight_dict[path + '_v.qweight'] = np.ascontiguousarray(
                        np.transpose(pack_int8_weights[:, self.hidden_size + self.kv_dim:], (1, 0)))
                    weight_dict[path + '_q.scales'] = np.ascontiguousarray(
                        np.transpose(scale_data[:, :self.hidden_size], (1, 0)))
                    weight_dict[path + '_k.scales'] = np.ascontiguousarray(
                        np.transpose(scale_data[:, self.hidden_size:self.hidden_size + self.kv_dim],
                                     (1, 0)))
                    weight_dict[path + '_v.scales'] = np.ascontiguousarray(
                        np.transpose(scale_data[:, self.hidden_size + self.kv_dim:], (1, 0)))
                    weight_dict[path + '_q.qzeros'] = np.ascontiguousarray(
                        np.transpose(unpacked_zeros[:, :self.hidden_size], (1, 0)))
                    weight_dict[path + '_k.qzeros'] = np.ascontiguousarray(
                        np.transpose(
                            unpacked_zeros[:, self.hidden_size:self.hidden_size + self.kv_dim],
                            (1, 0)))
                    weight_dict[path + '_v.qzeros'] = np.ascontiguousarray(
                        np.transpose(unpacked_zeros[:, self.hidden_size + self.kv_dim:], (1, 0)))
                elif 'gate_up_proj' in qweight_path:
                    weight_dict[path + '_gate.qweight'] = np.ascontiguousarray(
                        np.transpose(pack_int8_weights[:, :self.intermediate_size], (1, 0)))
                    weight_dict[path + '_up.qweight'] = np.ascontiguousarray(
                        np.transpose(pack_int8_weights[:, self.intermediate_size:], (1, 0)))
                    weight_dict[path + '_gate.scales'] = np.ascontiguousarray(
                        np.transpose(scale_data[:, :self.intermediate_size], (1, 0)))
                    weight_dict[path + '_up.scales'] = np.ascontiguousarray(
                        np.transpose(scale_data[:, self.intermediate_size:], (1, 0)))
                    weight_dict[path + '_gate.qzeros'] = np.ascontiguousarray(
                        np.transpose(unpacked_zeros[:, :self.intermediate_size], (1, 0)))
                    weight_dict[path + '_up.qzeros'] = np.ascontiguousarray(
                        np.transpose(unpacked_zeros[:, self.intermediate_size:], (1, 0)))
                else:
                    weight_dict[qweight_path] = np.ascontiguousarray(
                        np.transpose(pack_int8_weights, (1, 0)))
                    weight_dict[scale_path] = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))
                    weight_dict[zp_path] = np.ascontiguousarray(np.transpose(
                        unpacked_zeros, (1, 0)))
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        if self.model.is_exist(path + '.bias'):
            b_data = self.model.read(path + '.bias')
            weight_dict[path + '_q.bias'] = b_data[:self.hidden_size]
            weight_dict[path + '_k.bias'] = b_data[self.hidden_size:self.hidden_size + self.kv_dim]
            weight_dict[path + '_v.bias'] = b_data[self.hidden_size + self.kv_dim:]

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
        rotary_inv_freq = "inv_freq_float"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_inv_freq + ".weight": self.cos,
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
            q_op = self.linear(block_mlir, qkv_w + '_q', ln_op, [self.hidden_size, q_dim],
                               [1, self.seq_length, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, qkv_w + '_k', ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim])

            # v_proj
            v_op = self.linear(block_mlir, qkv_w + '_v', ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim])

            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(qkv_w + "_q.reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(qkv_w + "_k.reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_inv_freq)
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
                [input_shape, kv_shape, kv_shape],
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
            q_op = self.linear(block_mlir, qkv_w + '_q', ln_op, [self.hidden_size, q_dim],
                               [1, 1, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, qkv_w + '_k', ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, qkv_w + '_v', ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(qkv_w + "_q.reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(qkv_w + "_k.reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_inv_freq)

            return_ops.append(k_op)
            return_ops.append(v_op)
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
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        gen_block()
        gen_block_cache()
