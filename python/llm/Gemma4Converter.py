# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
import torch.nn.functional as F


class Gemma4Converter(LlmConverter):

    def __init__(self, args, config, loader=None):
        self.rmsnorm_type = WeightType.RMSNORM
        super().__init__(args, config, loader=None)
        # Override values set by base class __init__
        self.do_vit = True
        self.vit_f16_out_bf16 = True  # Gemma4 vit is f16, but we force output to bf16
        self.do_audio = True
        if self.do_audio:
            if args.audio_length <= 0:
                self.audio_length = 200
                print("audio_length not specified, using default 200")
            elif args.audio_length > 750:
                self.audio_length = 750
                print(f"audio_length {args.audio_length} exceeds max 750, capped to 750")
            else:
                self.audio_length = args.audio_length
            self.all_gen_mlirs.append(self.gen_audio_mlir)
            self.all_compiles.append(self.compile_audio)
        # Override rotary_embedding was called in super().__init__,
        # which already set self.cos_sliding, self.sin_sliding, self.cos_full, self.sin_full
        # and self.cos, self.sin

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GEMMA4_INFO
        self.llm_config = config.text_config
        self.llm_type = LlmType.GEMMA4

    @override
    def init_config(self):
        super().init_config()
        self.tie_word_embeddings = True
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2
        # Gemma4 specific config
        self.layer_types = getattr(self.llm_config, 'layer_types', None)
        self.global_head_dim = getattr(self.llm_config, 'global_head_dim', self.head_dim)
        self.num_kv_shared_layers = getattr(self.llm_config, 'num_kv_shared_layers', 0)
        self.sliding_window = getattr(self.llm_config, 'sliding_window', None)
        first_kv_shared = self.num_layers - self.num_kv_shared_layers if self.num_kv_shared_layers > 0 else self.num_layers
        self.is_kv_shared_layer = [i >= first_kv_shared for i in range(self.num_layers)]
        self.hidden_size_per_layer_input = getattr(self.llm_config, 'hidden_size_per_layer_input',
                                                   0)
        self.vocab_size_per_layer_input = getattr(self.llm_config, 'vocab_size_per_layer_input', 0)
        self.use_double_wide_mlp = getattr(self.llm_config, 'use_double_wide_mlp', False)
        # RoPE parameters per layer type
        rope_parameters = getattr(self.llm_config, 'rope_parameters', {})
        if rope_parameters:
            sliding_params = rope_parameters.get('sliding_attention', {})
            full_params = rope_parameters.get('full_attention', {})
            self.sliding_rope_theta = sliding_params.get('rope_theta', 10000.0)
            self.full_rope_theta = full_params.get('rope_theta', 1000000.0)
            self.partial_rotary_factor = full_params.get('partial_rotary_factor', 1.0)
        else:
            self.sliding_rope_theta = 10000.0
            self.full_rope_theta = 1000000.0
            self.partial_rotary_factor = 1.0

    @override
    def rotary_embedding(self):
        """Generate two sets of cos/sin for sliding and full attention layers.
        Computation matches HF's Gemma4TextRotaryEmbedding.forward exactly."""
        seq_len = self.seq_length
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)

        # sliding_attention: default rope, head_dim=256
        # HF computation: inv_freq[None,:,None] @ position_ids[:,None,:] → transpose(1,2)
        dim_sliding = self.head_dim
        inv_freq_sliding = 1.0 / (self.sliding_rope_theta**(
            torch.arange(0, dim_sliding, 2, dtype=torch.float) / dim_sliding))
        inv_freq_sliding_exp = inv_freq_sliding[None, :, None].float()  # [1, dim//2, 1]
        position_ids_exp = position_ids[:, None, :].float()  # [1, 1, seq_len]
        freqs_sliding = (inv_freq_sliding_exp @ position_ids_exp).transpose(
            1, 2)  # [1, seq_len, dim//2]
        emb_sliding = torch.cat((freqs_sliding, freqs_sliding), dim=-1)  # [1, seq_len, dim]
        cos_sliding = emb_sliding.cos().reshape(seq_len, 1, -1).numpy()
        sin_sliding = emb_sliding.sin().reshape(seq_len, 1, -1).numpy()

        # full_attention: proportional rope (matches HF _compute_proportional_rope_parameters)
        # Zero-pad inv_freq so cos/sin have full global_head_dim dimensions.
        # Non-rotary positions get freq=0 → cos=1, sin=0 (identity rotation),
        # so RopeOp can be applied directly on the full-dim q/k without splitting.
        head_dim_full = self.global_head_dim
        rope_angles = int(self.partial_rotary_factor * head_dim_full // 2)
        inv_freq_rotated = 1.0 / (self.full_rope_theta**(
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim_full))
        nope_angles = head_dim_full // 2 - rope_angles
        if nope_angles > 0:
            inv_freq_full = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
        else:
            inv_freq_full = inv_freq_rotated
        inv_freq_full_exp = inv_freq_full[None, :, None].float()  # [1, head_dim//2, 1]
        freqs_full = (inv_freq_full_exp @ position_ids_exp).transpose(
            1, 2)  # [1, seq_len, head_dim//2]
        emb_full = torch.cat((freqs_full, freqs_full), dim=-1)  # [1, seq_len, head_dim]
        cos_full = emb_full.cos().reshape(seq_len, 1, -1).numpy()
        sin_full = emb_full.sin().reshape(seq_len, 1, -1).numpy()

        # Store all four internally
        self.cos_sliding = cos_sliding
        self.sin_sliding = sin_sliding
        self.cos_full = cos_full
        self.sin_full = sin_full

        # Return sliding cos/sin for base class compatibility
        return cos_sliding, sin_sliding

    def _compute_layer_params(self, idx):
        """Compute layer-specific parameters based on layer type and KV sharing."""
        layer_type = self.layer_types[idx]
        is_full = layer_type == "full_attention"
        is_shared = self.is_kv_shared_layer[idx]

        cur_head_dim = self.global_head_dim if is_full else self.head_dim
        cur_q_dim = self.num_attention_heads * cur_head_dim
        cur_kv_dim = self.num_key_value_heads * cur_head_dim
        cur_intermediate = self.intermediate_size
        if self.use_double_wide_mlp and is_shared:
            cur_intermediate *= 2

        # For full_attention, cos/sin already have cur_head_dim dimensions (proportional RoPE
        # zero-padded), so RopeOp applies directly on full-dim q/k — no partial rotary split.
        # For sliding_attention, full rotary (rotary_dim = head_dim).
        rotary_dim = cur_head_dim

        return {
            'is_full': is_full,
            'is_shared': is_shared,
            'cur_head_dim': cur_head_dim,
            'cur_q_dim': cur_q_dim,
            'cur_kv_dim': cur_kv_dim,
            'cur_intermediate': cur_intermediate,
            'rotary_dim': rotary_dim,
            'rotary_cos_name': "rotary_cos_full" if is_full else "rotary_cos_sliding",
            'rotary_sin_name': "rotary_sin_full" if is_full else "rotary_sin_sliding",
            'rotary_cos': self.cos_full if is_full else self.cos_sliding,
            'rotary_sin': self.sin_full if is_full else self.sin_sliding,
        }

    def _rms_norm_no_scale(self, mlir_gen, in_op, norm_path, name="", eps=None):
        """RMSNorm without learnable weight (with_scale=False). Uses all-ones weight."""
        input_shape = list(in_op.type.shape)
        norm_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        weight_op = mlir_gen.create_weight_op(norm_path + ".weight", norm_shape)
        loc_name = name if name else norm_path
        eps = self.rms_norm_eps if eps is None else eps
        return top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                             in_op,
                             weight_op,
                             eps=eps,
                             weight_keep_f32=True,
                             loc=self.get_loc(loc_name, mlir_gen),
                             ip=mlir_gen.insert_point).output

    def _apply_rotary_pos_q_only(self, mlir_gen, pos_op, q_op, rotary_cos_name, rotary_sin_name,
                                 rotary_dim, head_dim):
        """Apply RoPE only to q, for shared KV layers where k/v already have RoPE."""
        ip = mlir_gen.insert_point

        def T(shape):
            return mlir_gen.get_tensor_type(shape)

        def L(name):
            return self.get_loc(name, mlir_gen)

        dim = pos_op.type.shape[-1]
        cos_weight_op = mlir_gen.create_weight_op(rotary_cos_name + ".weight",
                                                  [self.seq_length, 1, rotary_dim])
        cos_op = top.GatherOp(T([1, dim, 1, rotary_dim]),
                              cos_weight_op,
                              pos_op,
                              axis=0,
                              loc=L(rotary_cos_name),
                              ip=ip).output
        sin_weight_op = mlir_gen.create_weight_op(rotary_sin_name + ".weight",
                                                  [self.seq_length, 1, rotary_dim])
        sin_op = top.GatherOp(T([1, dim, 1, rotary_dim]),
                              sin_weight_op,
                              pos_op,
                              axis=0,
                              loc=L(rotary_sin_name),
                              ip=ip).output

        if rotary_dim == head_dim:
            q_shape = list(q_op.type.shape)
            return top.RopeOp(T(q_shape),
                              q_op,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("q_proj.rotary"),
                              ip=ip).output

        # Partial rotary on q only
        q_shape = list(q_op.type.shape)
        non_rotary_dim = head_dim - rotary_dim
        q_rotary_shape = q_shape[:-1] + [rotary_dim]
        q_non_rotary_shape = q_shape[:-1] + [non_rotary_dim]

        q_rotary = top.SliceOp(T(q_rotary_shape),
                               q_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               offset=[0, 0, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=q_rotary_shape,
                               axes=[],
                               loc=self.get_loc("q_proj.rotary_slice", mlir_gen),
                               ip=ip).output
        q_non_rotary = top.SliceOp(T(q_non_rotary_shape),
                                   q_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   offset=[0, 0, 0, rotary_dim],
                                   steps=[1, 1, 1, 1],
                                   ends=q_shape,
                                   axes=[],
                                   loc=self.get_loc("q_proj.non_rotary_slice", mlir_gen),
                                   ip=ip).output
        q_rotary = top.RopeOp(T(q_rotary_shape),
                              q_rotary,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("q_proj.rotary"),
                              ip=ip).output
        return top.ConcatOp(T(q_shape), [q_rotary, q_non_rotary],
                            axis=len(q_shape) - 1,
                            loc=L("q_proj.rotary_concat"),
                            ip=ip).output

    def _apply_rotary_pos_partial(self, mlir_gen, pos_op, q_op, k_op, rotary_cos_name,
                                  rotary_sin_name, rotary_dim, head_dim):
        """Apply RoPE with partial rotary factor.
        For partial rotary (rotary_dim < head_dim), split q/k into rotary and non-rotary parts,
        apply RoPE only to the rotary part, then concatenate back.
        """
        ip = mlir_gen.insert_point

        def T(shape):
            return mlir_gen.get_tensor_type(shape)

        def L(name):
            return self.get_loc(name, mlir_gen)

        # Load cos/sin weights with rotary_dim
        dim = pos_op.type.shape[-1]
        cos_weight_op = mlir_gen.create_weight_op(rotary_cos_name + ".weight",
                                                  [self.seq_length, 1, rotary_dim])
        cos_op = top.GatherOp(T([1, dim, 1, rotary_dim]),
                              cos_weight_op,
                              pos_op,
                              axis=0,
                              loc=L(rotary_cos_name),
                              ip=ip).output
        sin_weight_op = mlir_gen.create_weight_op(rotary_sin_name + ".weight",
                                                  [self.seq_length, 1, rotary_dim])
        sin_op = top.GatherOp(T([1, dim, 1, rotary_dim]),
                              sin_weight_op,
                              pos_op,
                              axis=0,
                              loc=L(rotary_sin_name),
                              ip=ip).output

        if rotary_dim == head_dim:
            # Full rotary - use RopeOp directly
            q_op_shape = list(q_op.type.shape)
            q_op = top.RopeOp(T(q_op_shape),
                              q_op,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("q_proj.rotary"),
                              ip=ip).output
            k_op_shape = list(k_op.type.shape)
            k_op = top.RopeOp(T(k_op_shape),
                              k_op,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("k_cache"),
                              ip=ip).output
            return q_op, k_op

        # Partial rotary - split into rotary and non-rotary parts
        q_shape = list(q_op.type.shape)
        k_shape = list(k_op.type.shape)
        non_rotary_dim = head_dim - rotary_dim

        # Split q: [batch, seq, heads, head_dim] -> rotary [.., rotary_dim] + non_rotary [.., non_rotary_dim]
        q_rotary_shape = q_shape[:-1] + [rotary_dim]
        q_non_rotary_shape = q_shape[:-1] + [non_rotary_dim]
        q_rotary = top.SliceOp(T(q_rotary_shape),
                               q_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               offset=[0, 0, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=q_rotary_shape,
                               axes=[],
                               loc=self.get_loc("q_proj.rotary_slice", mlir_gen),
                               ip=ip).output
        q_non_rotary = top.SliceOp(T(q_non_rotary_shape),
                                   q_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   offset=[0, 0, 0, rotary_dim],
                                   steps=[1, 1, 1, 1],
                                   ends=q_shape,
                                   axes=[],
                                   loc=self.get_loc("q_proj.non_rotary_slice", mlir_gen),
                                   ip=ip).output

        # Apply RoPE to rotary part
        q_rotary = top.RopeOp(T(q_rotary_shape),
                              q_rotary,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("q_proj.rotary"),
                              ip=ip).output

        # Split k similarly
        k_rotary_shape = k_shape[:-1] + [rotary_dim]
        k_non_rotary_shape = k_shape[:-1] + [non_rotary_dim]
        k_rotary = top.SliceOp(T(k_rotary_shape),
                               k_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               offset=[0, 0, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=k_rotary_shape,
                               axes=[],
                               loc=self.get_loc("k_proj.rotary_slice", mlir_gen),
                               ip=ip).output
        k_non_rotary = top.SliceOp(T(k_non_rotary_shape),
                                   k_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   mlir_gen.none_op,
                                   offset=[0, 0, 0, rotary_dim],
                                   steps=[1, 1, 1, 1],
                                   ends=k_shape,
                                   axes=[],
                                   loc=self.get_loc("k_proj.non_rotary_slice", mlir_gen),
                                   ip=ip).output

        k_rotary = top.RopeOp(T(k_rotary_shape),
                              k_rotary,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L("k_cache.rotary"),
                              ip=ip).output

        # Concatenate rotary and non-rotary back
        q_op = top.ConcatOp(T(q_shape), [q_rotary, q_non_rotary],
                            axis=len(q_shape) - 1,
                            loc=L("q_proj.rotary_concat"),
                            ip=ip).output
        k_op = top.ConcatOp(T(k_shape), [k_rotary, k_non_rotary],
                            axis=len(k_shape) - 1,
                            loc=L("k_cache"),
                            ip=ip).output

        return q_op, k_op

    @override
    def gen_block_mlir(self, idx):
        tqdm.write(f"generate block_{idx} mlir ...")
        params = self._compute_layer_params(idx)
        is_full = params['is_full']
        is_shared = params['is_shared']
        cur_head_dim = params['cur_head_dim']
        cur_q_dim = params['cur_q_dim']
        cur_kv_dim = params['cur_kv_dim']
        cur_intermediate = params['cur_intermediate']
        rotary_dim = params['rotary_dim']
        rotary_cos_name = params['rotary_cos_name']
        rotary_sin_name = params['rotary_sin_name']
        rotary_cos = params['rotary_cos']
        rotary_sin = params['rotary_sin']

        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
        q_norm = TOP_PATH + self.model_info.weights[LlmList.Q_NORM]
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
        k_norm = TOP_PATH + self.model_info.weights[LlmList.K_NORM]
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
        v_norm = TOP_PATH + self.model_info.weights[LlmList.V_NORM]
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        pre_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.PRE_MLP_LN]
        post_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.POST_MLP_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        layer_scalar_path = TOP_PATH + self.model_info.weights[LlmList.LAYER_SCALAR]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.num_device < 2 and idx == self.num_layers - 1

        has_per_layer_input = self.hidden_size_per_layer_input > 0
        per_layer_input_gate = TOP_PATH + self.model_info.weights[LlmList.PER_LAYER_INPUT_GATE]
        per_layer_projection = TOP_PATH + self.model_info.weights[LlmList.PER_LAYER_PROJECTION]
        post_per_layer_input_norm = TOP_PATH + self.model_info.weights[
            LlmList.POST_PER_LAYER_INPUT_NORM]

        # save weights
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos_name + ".weight": rotary_cos,
            rotary_sin_name + ".weight": rotary_sin,
        }
        self.set_common_weight(input_ln, weight_dict, self.rmsnorm_type)
        self.set_linear_weight(q_proj, weight_dict)
        self.set_common_weight(q_norm, weight_dict, self.rmsnorm_type)
        if not is_shared:
            self.set_linear_weight(k_proj, weight_dict)
            self.set_common_weight(k_norm, weight_dict, self.rmsnorm_type)
            self.set_linear_weight(v_proj, weight_dict)
            # v_norm: with_scale=False, create all-ones weight manually
            weight_dict[v_norm + ".weight"] = np.ones(cur_head_dim, dtype=np.float32)
        self.set_linear_weight(o_proj, weight_dict)
        self.set_common_weight(post_attn_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(pre_mlp_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(post_mlp_ln, weight_dict, self.rmsnorm_type)
        self.set_linear_weight(mlp_gate, weight_dict)
        self.set_linear_weight(mlp_up, weight_dict)
        self.set_linear_weight(mlp_down, weight_dict)
        if has_per_layer_input:
            self.set_linear_weight(per_layer_input_gate, weight_dict)
            self.set_linear_weight(per_layer_projection, weight_dict)
            self.set_common_weight(post_per_layer_input_norm, weight_dict, self.rmsnorm_type)
            # Per-layer weights sliced for layer idx (avoids computing full N*D tensor per block)
            embed_per_layer = self.model_info.weights[LlmList.EMBEDING_PER_LAYER]
            model_projection = self.model_info.weights[LlmList.PER_LAYER_MODEL_PROJECTION]
            projection_norm = self.model_info.weights[LlmList.PER_LAYER_PROJECTION_NORM]
            per_layer_dim = self.hidden_size_per_layer_input
            # Slice embed_tokens_per_layer.weight: [vocab, N*D] → [vocab, D] for layer idx
            full_emb = self.model.read(embed_per_layer + ".weight")
            emb_slice = full_emb[:, idx * per_layer_dim:(idx + 1) * per_layer_dim]
            weight_dict[embed_per_layer + f".weight.{idx}"] = emb_slice.copy()
            # Slice per_layer_model_projection.weight: [N*D, hidden] → transpose → [hidden, N*D]
            # then slice columns → [hidden, D] for layer idx
            full_proj = self.model.read(model_projection + ".weight")
            full_proj_t = np.ascontiguousarray(np.transpose(full_proj, (1, 0)))  # [hidden, N*D]
            proj_slice = full_proj_t[:, idx * per_layer_dim:(idx + 1) * per_layer_dim]
            weight_dict[model_projection + f".{idx}.weight"] = proj_slice.copy()
            self.set_common_weight(projection_norm, weight_dict, self.rmsnorm_type)
        # layer_scalar: read as common weight and extract value for MulConstOp
        self.set_common_weight(layer_scalar_path, weight_dict)
        if do_norm:
            self.set_common_weight(norm, weight_dict, self.rmsnorm_type)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weight_keys.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        # read layer_scalar actual value from weight_dict
        layer_scalar_weight_key = layer_scalar_path + ".weight" if (
            layer_scalar_path + ".weight") in weight_dict else layer_scalar_path
        layer_scalar_val = float(weight_dict[layer_scalar_weight_key].item())

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            batch = input_shape[0]
            len = input_shape[1]
            new_op = self.rms_norm(mlir_gen, in_op, pre_mlp_ln)

            gate_op = self.linear(mlir_gen, mlp_gate, new_op, [self.hidden_size, cur_intermediate],
                                  [batch, len, cur_intermediate])
            act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
            up_op = self.linear(mlir_gen, mlp_up, new_op, [self.hidden_size, cur_intermediate],
                                [batch, len, cur_intermediate])
            new_op = top.MulOp(mlir_gen.get_tensor_type([batch, len, cur_intermediate]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen, mlp_down, new_op, [cur_intermediate, self.hidden_size],
                                  input_shape)

            down_op = self.rms_norm(mlir_gen, down_op, post_mlp_ln)
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(mlp_down + ".add", mlir_gen),
                               ip=ip).output
            return new_op

        def gen_per_layer_input(mlir_gen, input_shape, hidden_states, residual_op, ids_op,
                                embeds_op):
            """Generate per_layer_input subgraph with per-layer sliced weights.
            Each block uses its own slice of embed_tokens_per_layer and per_layer_model_projection,
            avoiding the need to compute the full N*D tensor and then slice."""
            ip = mlir_gen.insert_point
            per_layer_dim = self.hidden_size_per_layer_input
            batch = input_shape[0]
            len = input_shape[1]

            # Scale constants from HF source
            per_layer_embed_scale = per_layer_dim**0.5
            model_projection_scale = self.hidden_size**-0.5
            per_layer_input_scale = 2.0**-0.5

            # Path A: embed_tokens_per_layer(ids) * per_layer_embed_scale (sliced for layer idx)
            per_layer_slice_shape = [batch, len, per_layer_dim]
            emb_per_layer_weight = mlir_gen.create_weight_op(
                embed_per_layer + f".weight.{idx}",
                [self.vocab_size_per_layer_input, per_layer_dim])
            gather_op = top.GatherOp(mlir_gen.get_tensor_type(per_layer_slice_shape),
                                     emb_per_layer_weight,
                                     ids_op,
                                     axis=0,
                                     loc=self.get_loc("embed_tokens_per_layer.gather", mlir_gen),
                                     ip=ip).output
            per_layer_inputs_op = top.MulConstOp(mlir_gen.get_tensor_type(per_layer_slice_shape),
                                                 gather_op,
                                                 const_val=per_layer_embed_scale,
                                                 loc=self.get_loc("embed_tokens_per_layer.scale",
                                                                  mlir_gen),
                                                 ip=ip).output

            # Path B: per_layer_model_projection(embeds) * model_projection_scale (sliced for layer idx)
            proj_op = self.linear(mlir_gen, model_projection + f".{idx}", embeds_op,
                                  [self.hidden_size, per_layer_dim], per_layer_slice_shape)
            proj_op = top.MulConstOp(mlir_gen.get_tensor_type(per_layer_slice_shape),
                                     proj_op,
                                     const_val=model_projection_scale,
                                     loc=self.get_loc("per_layer_model_projection.scale", mlir_gen),
                                     ip=ip).output
            proj_op = self.rms_norm(mlir_gen, proj_op, projection_norm)

            # (projection + per_layer_inputs) * per_layer_input_scale
            add_op = top.AddOp(mlir_gen.get_tensor_type(per_layer_slice_shape),
                               [proj_op, per_layer_inputs_op],
                               loc=self.get_loc("per_layer.add", mlir_gen),
                               ip=ip).output
            per_layer_input_op = top.MulConstOp(mlir_gen.get_tensor_type(per_layer_slice_shape),
                                                add_op,
                                                const_val=per_layer_input_scale,
                                                loc=self.get_loc("per_layer.scale", mlir_gen),
                                                ip=ip).output

            # Per-layer gate subgraph: gate -> act -> mul -> proj -> norm -> add
            gate_shape = [batch, len, per_layer_dim]
            gate_op = self.linear(mlir_gen, per_layer_input_gate, hidden_states,
                                  [self.hidden_size, per_layer_dim], gate_shape)
            gate_op = self.activate(mlir_gen, gate_op, self.hidden_act, per_layer_input_gate)
            gate_op = top.MulOp(mlir_gen.get_tensor_type(gate_shape), [gate_op, per_layer_input_op],
                                loc=self.get_loc(per_layer_input_gate + ".per_layer_mul", mlir_gen),
                                ip=ip).output
            proj_op = self.linear(mlir_gen, per_layer_projection, gate_op,
                                  [per_layer_dim, self.hidden_size], input_shape)
            proj_op = self.rms_norm(mlir_gen, proj_op, post_per_layer_input_norm)
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [residual_op, proj_op],
                               loc=self.get_loc(per_layer_projection + ".add", mlir_gen),
                               ip=ip).output
            return new_op

        # ============ gen_block (prefill) ============
        def gen_block_by_length(name, input_len):
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = input_len
            mask_shape = [1, 1, input_len, input_len]
            q_shape = [1, input_len, self.num_attention_heads, cur_head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, cur_head_dim]

            # Input/output shapes depend on whether shared KV layer
            input_types = ["F32", "INT32", "F32"]
            input_shapes = [input_shape, id_shape, mask_shape]
            output_shapes = []
            return_ops_list = []

            if has_per_layer_input:
                input_shapes.extend([id_shape, input_shape])
                input_types.extend(["INT32", "F32"])

            if is_shared:
                # Shared layer: receives shared_k and shared_v as inputs, outputs only hidden_states
                shared_k_shape = [1, input_len, self.num_key_value_heads, cur_head_dim]
                shared_v_shape = [1, input_len, self.num_key_value_heads, cur_head_dim]
                input_shapes.extend([shared_k_shape, shared_v_shape])
                input_types.extend(["F32", "F32"])
                output_shapes.append(input_shape)
            else:
                # Normal layer: outputs hidden_states + k + v
                output_shapes.extend([input_shape, kv_shape, kv_shape])

            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      Platform.LLM,
                                      input_types,
                                      weight_file=f"../{weight_file}")

            def T(shape):
                return block_mlir.get_tensor_type(shape)

            def L(name):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            input_idx = 3
            ids_op = None
            embeds_op = None
            if has_per_layer_input:
                ids_op = block_mlir.create_input_op(L("input_ids"), input_idx)
                embeds_op = block_mlir.create_input_op(L("inputs_embeds"), input_idx + 1)
                input_idx += 2

            shared_k_op = None
            shared_v_op = None
            if is_shared:
                shared_k_op = block_mlir.create_input_op(L("shared_k"), input_idx)
                input_idx += 1
                shared_v_op = block_mlir.create_input_op(L("shared_v"), input_idx)
                input_idx += 1

            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, cur_q_dim],
                               [1, input_len, cur_q_dim])
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, cur_head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)

            if not is_shared:
                # k_proj, k_norm
                k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [1, input_len, cur_kv_dim])
                k_op = top.ReshapeOp(T(kv_shape),
                                     k_op,
                                     shape=[1, -1, self.num_key_value_heads, cur_head_dim],
                                     loc=L(k_proj + ".reshape"),
                                     ip=ip).output
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

                # v_proj, v_norm (no scale)
                v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [1, input_len, cur_kv_dim])
                v_op = top.ReshapeOp(T(kv_shape),
                                     v_op,
                                     shape=[1, -1, self.num_key_value_heads, cur_head_dim],
                                     loc=L("v_cache.reshape"),
                                     ip=ip).output
                v_op = self._rms_norm_no_scale(block_mlir, v_op, v_norm, name="v_cache")

                # RoPE on q and k
                q_op, k_op = self._apply_rotary_pos_partial(block_mlir, in1_op, q_op, k_op,
                                                            rotary_cos_name, rotary_sin_name,
                                                            rotary_dim, cur_head_dim)

                return_ops_list.append(k_op)
                return_ops_list.append(v_op)
            else:
                # Shared KV layer: k and v come from shared inputs
                # Apply RoPE only to q; shared k/v already have RoPE from source layer
                q_op = self._apply_rotary_pos_q_only(block_mlir, in1_op, q_op, rotary_cos_name,
                                                     rotary_sin_name, rotary_dim, cur_head_dim)
                k_op = shared_k_op
                v_op = shared_v_op

            # FAttention
            fa_op = top.FAttentionOp(T([1, input_len, cur_q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=1.0,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=cur_head_dim,
                                     mq=input_len,
                                     mk=input_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, o_proj, fa_op, [cur_q_dim, self.hidden_size],
                               input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output

            # MLP
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            residual_mlp = new_op  # POST-MLP residual (HF uses this for per_layer_projection.add)

            # per_layer_input
            if has_per_layer_input:
                new_op = gen_per_layer_input(block_mlir, input_shape, new_op, residual_mlp, ids_op,
                                             embeds_op)

            # layer_scalar
            layer_scalar_loc = "layer_scalar" if do_norm else "output_states"
            new_op = top.MulConstOp(T(input_shape),
                                    new_op,
                                    const_val=layer_scalar_val,
                                    loc=L(layer_scalar_loc),
                                    ip=ip).output

            # final norm (only for last layer, after all layer logic)
            if do_norm:
                new_op = self.rms_norm(block_mlir, new_op, norm, "output_states")

            block_mlir.create_return_op([new_op] + return_ops_list)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # ============ gen_block_cache (decode) ============
        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [self.batch, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            mask_len = self.seq_length + 1
            mask_shape = [self.batch, 1, 1, mask_len]
            history_shape = [self.batch, self.seq_length, self.num_key_value_heads, cur_head_dim]
            q_shape = [self.batch, 1, self.num_attention_heads, cur_head_dim]
            kv_shape = [self.batch, 1, self.num_key_value_heads, cur_head_dim]

            input_types = ["F32", "INT32", "F32"]
            input_shapes = [input_shape, id_shape, mask_shape]

            if has_per_layer_input:
                input_ids_shape = [self.batch, 1]
                input_shapes.extend([input_ids_shape, input_shape])
                input_types.extend(["INT32", "F32"])

            if is_shared:
                # Shared KV layer in decode mode
                shared_k_len = mask_len  # same as total kv length
                shared_k_shape = [self.batch, shared_k_len, self.num_key_value_heads, cur_head_dim]
                shared_v_shape = [self.batch, shared_k_len, self.num_key_value_heads, cur_head_dim]
                input_shapes.extend([shared_k_shape, shared_v_shape])
                input_types.extend(["F32", "F32"])
                output_shapes = [input_shape]
            else:
                output_shapes = [input_shape, kv_shape, kv_shape]
                input_shapes.extend([history_shape, history_shape])
                input_types.extend(["F32", "F32"])

            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      Platform.LLM,
                                      input_types,
                                      weight_file=f"../{weight_file}")

            def T(shape):
                return block_mlir.get_tensor_type(shape)

            def L(name):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            input_idx = 3
            ids_op = None
            embeds_op = None
            if has_per_layer_input:
                ids_op = block_mlir.create_input_op(L("input_ids"), input_idx)
                embeds_op = block_mlir.create_input_op(L("inputs_embeds"), input_idx + 1)
                input_idx += 2

            shared_k_op = None
            shared_v_op = None
            if not is_shared:
                in3_op = block_mlir.create_input_op(L("history_k"), input_idx)
                input_idx += 1
                in4_op = block_mlir.create_input_op(L("history_v"), input_idx)
                input_idx += 1
            else:
                shared_k_op = block_mlir.create_input_op(L("shared_k"), input_idx)
                input_idx += 1
                shared_v_op = block_mlir.create_input_op(L("shared_v"), input_idx)
                input_idx += 1

            return_ops = []

            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, cur_q_dim],
                               [self.batch, 1, cur_q_dim])
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)

            if not is_shared:
                k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [self.batch, 1, cur_kv_dim])
                k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

                v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [self.batch, 1, cur_kv_dim])
                v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache.reshape"), ip=ip).output
                v_op = self._rms_norm_no_scale(block_mlir, v_op, v_norm, name="v_cache")

                # RoPE on q and k
                q_op, k_op = self._apply_rotary_pos_partial(block_mlir, in1_op, q_op, k_op,
                                                            rotary_cos_name, rotary_sin_name,
                                                            rotary_dim, cur_head_dim)

                return_ops.append(k_op)
                return_ops.append(v_op)

                # KV concat/insert
                k_op = top.ConcatOp(T(
                    [1, self.seq_length + 1, self.num_key_value_heads, cur_head_dim]),
                                    [in3_op, k_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(k_proj + ".concat"),
                                    ip=ip).output
                v_op = top.ConcatOp(T(
                    [1, self.seq_length + 1, self.num_key_value_heads, cur_head_dim]),
                                    [in4_op, v_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(v_proj + ".concat"),
                                    ip=ip).output

                # FAttention
                fa_op = top.FAttentionOp(T([self.batch, 1, cur_q_dim]),
                                         q_op,
                                         k_op,
                                         v_op,
                                         in2_op,
                                         block_mlir.none_op,
                                         scale=1.0,
                                         batch=self.batch,
                                         q_head=self.num_attention_heads,
                                         kv_head=self.num_key_value_heads,
                                         dim=cur_head_dim,
                                         mq=1,
                                         mk=mask_len,
                                         keep_dims=False,
                                         loc=L(TOP_PATH + "fattention"),
                                         ip=ip).output
            else:
                # Shared KV layer: apply RoPE only to q
                q_op = self._apply_rotary_pos_q_only(block_mlir, in1_op, q_op, rotary_cos_name,
                                                     rotary_sin_name, rotary_dim, cur_head_dim)

                k_op = shared_k_op
                v_op = shared_v_op

                # FAttention
                fa_op = top.FAttentionOp(T([self.batch, 1, cur_q_dim]),
                                         q_op,
                                         k_op,
                                         v_op,
                                         in2_op,
                                         block_mlir.none_op,
                                         scale=1.0,
                                         batch=self.batch,
                                         q_head=self.num_attention_heads,
                                         kv_head=self.num_key_value_heads,
                                         dim=cur_head_dim,
                                         mq=1,
                                         mk=mask_len,
                                         keep_dims=False,
                                         loc=L(TOP_PATH + "fattention"),
                                         ip=ip).output

            o_op = self.linear(block_mlir, o_proj, fa_op, [cur_q_dim, self.hidden_size],
                               input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output

            # MLP
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            residual_mlp = new_op  # POST-MLP residual

            # per_layer_input
            if has_per_layer_input:
                new_op = gen_per_layer_input(block_mlir, input_shape, new_op, residual_mlp, ids_op,
                                             embeds_op)

            # layer_scalar
            layer_scalar_loc = "layer_scalar" if do_norm else "output_states"
            new_op = top.MulConstOp(T(input_shape),
                                    new_op,
                                    const_val=layer_scalar_val,
                                    loc=L(layer_scalar_loc),
                                    ip=ip).output

            # final norm (only for last layer, after all layer logic)
            if do_norm:
                new_op = self.rms_norm(block_mlir, new_op, norm, "output_states")

            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # ============ gen_block_with_kv (prefill with history) ============
        def gen_block_with_kv():
            name = f"block_{idx}"
            input_len = self.max_input_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = input_len
            max_kv_len = self.max_prefill_kv_length + input_len
            mask_shape = [1, 1, input_len, max_kv_len]
            history_shape = [1, self.max_prefill_kv_length, self.num_key_value_heads, cur_head_dim]
            q_shape = [1, input_len, self.num_attention_heads, cur_head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, cur_head_dim]

            input_types = ["F32", "INT32", "F32"]
            input_shapes = [input_shape, id_shape, mask_shape]

            if has_per_layer_input:
                input_shapes.extend([id_shape, input_shape])
                input_types.extend(["INT32", "F32"])

            if is_shared:
                # Shared KV layer: receives shared_k (full length), shared_v (full length)
                shared_k_shape = [1, max_kv_len, self.num_key_value_heads, cur_head_dim]
                shared_v_shape = [1, max_kv_len, self.num_key_value_heads, cur_head_dim]
                input_shapes.extend([shared_k_shape, shared_v_shape])
                input_types.extend(["F32", "F32"])
                output_shapes = [input_shape]
            else:
                input_shapes.extend([history_shape, history_shape])
                input_types.extend(["F32", "F32"])
                output_shapes = [input_shape, kv_shape, kv_shape]

            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      Platform.LLM,
                                      input_types,
                                      weight_file=f"../{weight_file}")

            def T(shape):
                return block_mlir.get_tensor_type(shape)

            def L(name):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            input_idx = 3
            ids_op = None
            embeds_op = None
            if has_per_layer_input:
                ids_op = block_mlir.create_input_op(L("input_ids"), input_idx)
                embeds_op = block_mlir.create_input_op(L("inputs_embeds"), input_idx + 1)
                input_idx += 2

            return_ops = []

            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, cur_q_dim],
                               [1, input_len, cur_q_dim])
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, cur_head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)

            if not is_shared:
                in3_op = block_mlir.create_input_op(L("history_k"), input_idx)
                input_idx += 1
                in4_op = block_mlir.create_input_op(L("history_v"), input_idx)
                input_idx += 1

                k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [1, input_len, cur_kv_dim])
                k_op = top.ReshapeOp(T(kv_shape),
                                     k_op,
                                     shape=[1, -1, self.num_key_value_heads, cur_head_dim],
                                     loc=L(k_proj + ".reshape"),
                                     ip=ip).output
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

                v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, cur_kv_dim],
                                   [1, input_len, cur_kv_dim])
                v_op = top.ReshapeOp(T(kv_shape),
                                     v_op,
                                     shape=[1, -1, self.num_key_value_heads, cur_head_dim],
                                     loc=L("v_cache.reshape"),
                                     ip=ip).output
                v_op = self._rms_norm_no_scale(block_mlir, v_op, v_norm, name="v_cache")

                # RoPE on q and k
                q_op, k_op = self._apply_rotary_pos_partial(block_mlir, in1_op, q_op, k_op,
                                                            rotary_cos_name, rotary_sin_name,
                                                            rotary_dim, cur_head_dim)

                return_ops.append(k_op)
                return_ops.append(v_op)

                # KV concat with history
                k_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, cur_head_dim]),
                                    [in3_op, k_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(k_proj + ".concat"),
                                    ip=ip).output
                v_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, cur_head_dim]),
                                    [in4_op, v_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(v_proj + ".concat"),
                                    ip=ip).output

                # FAttention
                fa_op = top.FAttentionOp(T([1, input_len, cur_q_dim]),
                                         q_op,
                                         k_op,
                                         v_op,
                                         in2_op,
                                         block_mlir.none_op,
                                         scale=1.0,
                                         batch=1,
                                         q_head=self.num_attention_heads,
                                         kv_head=self.num_key_value_heads,
                                         dim=cur_head_dim,
                                         mq=input_len,
                                         mk=max_kv_len,
                                         keep_dims=False,
                                         loc=L(TOP_PATH + "fattention"),
                                         ip=ip).output
            else:
                shared_k_op = block_mlir.create_input_op(L("shared_k"), input_idx)
                input_idx += 1
                shared_v_op = block_mlir.create_input_op(L("shared_v"), input_idx)
                input_idx += 1

                # Apply RoPE only to q (shared k/v already have RoPE from source layer)
                q_op = self._apply_rotary_pos_q_only(block_mlir, in1_op, q_op, rotary_cos_name,
                                                     rotary_sin_name, rotary_dim, cur_head_dim)

                # FAttention
                fa_op = top.FAttentionOp(T([1, input_len, cur_q_dim]),
                                         q_op,
                                         shared_k_op,
                                         shared_v_op,
                                         in2_op,
                                         block_mlir.none_op,
                                         scale=1.0,
                                         batch=1,
                                         q_head=self.num_attention_heads,
                                         kv_head=self.num_key_value_heads,
                                         dim=cur_head_dim,
                                         mq=input_len,
                                         mk=max_kv_len,
                                         keep_dims=False,
                                         loc=L(TOP_PATH + "fattention"),
                                         ip=ip).output

            o_op = self.linear(block_mlir, o_proj, fa_op, [cur_q_dim, self.hidden_size],
                               input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output

            # MLP
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            residual_mlp = new_op  # POST-MLP residual

            # per_layer_input
            if has_per_layer_input:
                new_op = gen_per_layer_input(block_mlir, input_shape, new_op, residual_mlp, ids_op,
                                             embeds_op)

            # layer_scalar
            layer_scalar_loc = "layer_scalar" if do_norm else "output_states"
            new_op = top.MulConstOp(T(input_shape),
                                    new_op,
                                    const_val=layer_scalar_val,
                                    loc=L(layer_scalar_loc),
                                    ip=ip).output

            # final norm (only for last layer, after all layer logic)
            if do_norm:
                new_op = self.rms_norm(block_mlir, new_op, norm, "output_states")

            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # ============ dispatch block generation ============
        if self.use_block_with_kv:
            gen_block_with_kv()
        else:
            name = f"block_{idx}"
            if self.share_prompt:
                name = f"block_prompt_{idx}"
                gen_block_by_length(name, self.max_prefill_kv_length)
            else:
                gen_block_by_length(name, self.max_input_length)
        if self.share_prompt:
            # share_prompt needs separate prompt block, then normal block
            gen_block_by_length(f"block_{idx}", self.max_input_length)
        gen_block_cache()

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        vconfig = self.config.vision_config

        embed_dim = vconfig.hidden_size
        hidden_act = vconfig.hidden_activation
        patch_size = vconfig.patch_size
        pooling_kernel_size = vconfig.pooling_kernel_size
        patch_dim = 3 * patch_size * patch_size
        head_dim = vconfig.head_dim
        num_attention_heads = vconfig.num_attention_heads
        num_key_value_heads = vconfig.num_key_value_heads
        spatial_dim = head_dim // 2
        position_embedding_size = vconfig.position_embedding_size

        tower_path = "model.vision_tower"
        patch_embedding = f"{tower_path}.patch_embedder"
        mm_projector_norm = "model.embed_vision.embedding_pre_projection_norm"
        mm_projector_mm = "model.embed_vision.embedding_projection"

        def _gen_vit_variant(name, batch_size, mm_tokens, vit_npz):
            max_patches = mm_tokens * pooling_kernel_size**2

            # ====== Save weights ======
            weights_dict = {}
            self.set_linear_weight(patch_embedding + ".input_proj", weights_dict)
            pos_emb_table = self.model.read(patch_embedding + ".position_embedding_table")
            weights_dict[patch_embedding + ".position_embedding_table_x"] = pos_emb_table[0]
            weights_dict[patch_embedding + ".position_embedding_table_y"] = pos_emb_table[1]

            rope_theta = vconfig.rope_parameters.get("rope_theta", 100.0)
            inv_freq = 1.0 / (rope_theta
                              **(torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
            pos_range = torch.arange(position_embedding_size, dtype=torch.float)
            freqs = torch.einsum('i,d->id', inv_freq, pos_range)
            emb = torch.cat([freqs, freqs], dim=0)
            weights_dict["vision_rotary.cos_table"] = emb.cos().T.unsqueeze(1).numpy()
            weights_dict["vision_rotary.sin_table"] = emb.sin().T.unsqueeze(1).numpy()

            weights_dict["padding_mask.ones"] = np.ones([1, max_patches], dtype=np.float32)
            weights_dict["padding_mask.zeros"] = np.zeros([1, max_patches], dtype=np.float32)
            weights_dict["padding_mask.neg_inf"] = np.full([1, max_patches],
                                                           -65504.0,
                                                           dtype=np.float32)

            for idx in range(vconfig.num_hidden_layers):
                layer_path = f"{tower_path}.encoder.layers.{idx}"
                self.set_common_weight(f"{layer_path}.input_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.post_attention_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.pre_feedforward_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.post_feedforward_layernorm", weights_dict,
                                       self.rmsnorm_type)
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    self._set_clippable_linear_weight(f"{layer_path}.mlp.{proj}", weights_dict)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    self._set_clippable_linear_weight(f"{layer_path}.self_attn.{proj}",
                                                      weights_dict)
                self.set_common_weight(f"{layer_path}.self_attn.q_norm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.self_attn.k_norm", weights_dict,
                                       self.rmsnorm_type)
                weights_dict[f"{layer_path}.self_attn.v_norm.weight"] = np.ones(head_dim,
                                                                                dtype=np.float32)

            weights_dict[mm_projector_norm + ".weight"] = np.ones(embed_dim, dtype=np.float32)
            self.set_linear_weight(mm_projector_mm, weights_dict)

            k = pooling_kernel_size
            weights_dict["pooler.div_k_table"] = np.array(
                [i // k for i in range(position_embedding_size + 1)], dtype=np.float32)
            weights_dict["pooler.max_x_div_k_table"] = np.array(
                [(i + 1) // k for i in range(position_embedding_size)], dtype=np.float32)
            weights_dict["pooler.identity"] = np.eye(mm_tokens, mm_tokens, dtype=np.float32)

            np.savez(vit_npz, **weights_dict)

            # ====== Generate MLIR ======
            in_shape = [batch_size, max_patches, patch_dim]
            pos_shape = [batch_size, max_patches, 2]
            hidden_shape = [batch_size, max_patches, embed_dim]
            out_shape = [batch_size, mm_tokens, self.hidden_size]

            vit_mlir = MLIRImporter([in_shape, pos_shape], [out_shape],
                                    name,
                                    Platform.LLM, ["F32", "INT32"],
                                    weight_file=f"../{vit_npz}")
            ip = vit_mlir.insert_point

            def T(shape: list):
                return vit_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, vit_mlir)

            in_op = vit_mlir.create_input_op(L('pixel_values'), 0)
            pos_ids_op = vit_mlir.create_input_op(L('pixel_position_ids'), 1)

            # ====== Patch Embedder ======
            new_op = top.MulConstOp(T(in_shape), in_op, const_val=2.0, loc=L("pixel_scale"),
                                    ip=ip).output
            new_op = top.AddConstOp(T(in_shape),
                                    new_op,
                                    const_val=-1.0,
                                    loc=L("pixel_shift"),
                                    ip=ip).output
            new_op = self.linear(vit_mlir, patch_embedding + ".input_proj", new_op,
                                 [patch_dim, embed_dim], hidden_shape)

            # ====== Position Embedding ======
            pos_x_ids = top.SliceOp(T([batch_size, max_patches, 1]),
                                    pos_ids_op,
                                    vit_mlir.none_op,
                                    vit_mlir.none_op,
                                    vit_mlir.none_op,
                                    offset=[0, 0, 0],
                                    steps=[1, 1, 1],
                                    ends=[batch_size, max_patches, 1],
                                    axes=[],
                                    loc=L("pos_x_ids_slice"),
                                    ip=ip).output
            pos_y_ids = top.SliceOp(T([batch_size, max_patches, 1]),
                                    pos_ids_op,
                                    vit_mlir.none_op,
                                    vit_mlir.none_op,
                                    vit_mlir.none_op,
                                    offset=[0, 0, 1],
                                    steps=[1, 1, 1],
                                    ends=[batch_size, max_patches, 2],
                                    axes=[],
                                    loc=L("pos_y_ids_slice"),
                                    ip=ip).output
            pos_x_ids = top.ReshapeOp(T([batch_size, max_patches]),
                                      pos_x_ids,
                                      shape=[batch_size, -1],
                                      loc=L("pos_x_ids_reshape"),
                                      ip=ip).output
            pos_y_ids = top.ReshapeOp(T([batch_size, max_patches]),
                                      pos_y_ids,
                                      shape=[batch_size, -1],
                                      loc=L("pos_y_ids_reshape"),
                                      ip=ip).output

            pos_x_clamped = top.ClipOp(T([batch_size, max_patches]),
                                       pos_x_ids,
                                       min=0,
                                       max=position_embedding_size - 1,
                                       loc=L("pos_x_clamp"),
                                       ip=ip).output
            pos_y_clamped = top.ClipOp(T([batch_size, max_patches]),
                                       pos_y_ids,
                                       min=0,
                                       max=position_embedding_size - 1,
                                       loc=L("pos_y_clamp"),
                                       ip=ip).output

            pos_emb_table_x = vit_mlir.create_weight_op(
                patch_embedding + ".position_embedding_table_x",
                [position_embedding_size, embed_dim])
            pos_emb_table_y = vit_mlir.create_weight_op(
                patch_embedding + ".position_embedding_table_y",
                [position_embedding_size, embed_dim])
            pos_emb_x = top.GatherOp(T([batch_size, max_patches, embed_dim]),
                                     pos_emb_table_x,
                                     pos_x_clamped,
                                     axis=0,
                                     loc=L("pos_emb_x_gather"),
                                     ip=ip).output
            pos_emb_y = top.GatherOp(T([batch_size, max_patches, embed_dim]),
                                     pos_emb_table_y,
                                     pos_y_clamped,
                                     axis=0,
                                     loc=L("pos_emb_y_gather"),
                                     ip=ip).output
            pos_emb = top.AddOp(T(hidden_shape), [pos_emb_x, pos_emb_y],
                                loc=L("pos_emb_xy_add"),
                                ip=ip).output

            # ====== Padding Mask ======
            pos_ids_shifted = top.AddConstOp(T([batch_size, max_patches, 2]),
                                             pos_ids_op,
                                             const_val=1,
                                             loc=L("pos_ids_shift"),
                                             ip=ip).output
            pos_ids_valid = top.ReduceOp(T([batch_size, max_patches]),
                                         pos_ids_shifted,
                                         axes=[2],
                                         keepdims=0,
                                         mode=StringAttr.get("ReduceMin"),
                                         loc=L("pos_ids_reduce_min"),
                                         ip=ip).output
            is_valid = top.CompareConstOp(T([batch_size, max_patches]),
                                          pos_ids_valid,
                                          mode=StringAttr.get("Greater"),
                                          const_val=0.,
                                          inversed=False,
                                          loc=L("is_valid"),
                                          ip=ip).output
            valid_mask = top.WhereOp(T([batch_size, max_patches]),
                                     is_valid,
                                     vit_mlir.create_weight_op("padding_mask.ones",
                                                               [1, max_patches]),
                                     vit_mlir.create_weight_op("padding_mask.zeros",
                                                               [1, max_patches]),
                                     loc=L("valid_mask"),
                                     ip=ip).output
            valid_mask_3d = top.ReshapeOp(T([batch_size, max_patches, 1]),
                                          valid_mask,
                                          shape=[batch_size, -1, 1],
                                          loc=L("valid_mask_3d"),
                                          ip=ip).output
            pos_emb = top.MulOp(T(hidden_shape), [pos_emb, valid_mask_3d],
                                loc=L("pos_emb_mask"),
                                ip=ip).output

            # ====== Attention Mask ======
            attn_mask_per_key = top.WhereOp(T([batch_size, max_patches]),
                                            is_valid,
                                            vit_mlir.create_weight_op("padding_mask.zeros",
                                                                      [1, max_patches]),
                                            vit_mlir.create_weight_op("padding_mask.neg_inf",
                                                                      [1, max_patches]),
                                            loc=L("attn_mask_per_key"),
                                            ip=ip).output
            attn_mask_row = top.ReshapeOp(T([batch_size, 1, 1, max_patches]),
                                          attn_mask_per_key,
                                          shape=[batch_size, 1, 1, -1],
                                          loc=L("attn_mask_row"),
                                          ip=ip).output
            attn_mask = top.TileOp(T([batch_size, 1, max_patches, max_patches]),
                                   attn_mask_row,
                                   tile=[1, 1, max_patches, 1],
                                   loc=L("attn_mask"),
                                   ip=ip).output

            new_op = top.AddOp(T(hidden_shape), [new_op, pos_emb], loc=L("pos_emb_add"),
                               ip=ip).output

            # ====== 2D RoPE ======
            cos_table = vit_mlir.create_weight_op("vision_rotary.cos_table",
                                                  [position_embedding_size, 1, spatial_dim])
            sin_table = vit_mlir.create_weight_op("vision_rotary.sin_table",
                                                  [position_embedding_size, 1, spatial_dim])
            cos_x = top.GatherOp(T([batch_size, max_patches, 1, spatial_dim]),
                                 cos_table,
                                 pos_x_clamped,
                                 axis=0,
                                 loc=L("cos_x_gather"),
                                 ip=ip).output
            sin_x = top.GatherOp(T([batch_size, max_patches, 1, spatial_dim]),
                                 sin_table,
                                 pos_x_clamped,
                                 axis=0,
                                 loc=L("sin_x_gather"),
                                 ip=ip).output
            cos_y = top.GatherOp(T([batch_size, max_patches, 1, spatial_dim]),
                                 cos_table,
                                 pos_y_clamped,
                                 axis=0,
                                 loc=L("cos_y_gather"),
                                 ip=ip).output
            sin_y = top.GatherOp(T([batch_size, max_patches, 1, spatial_dim]),
                                 sin_table,
                                 pos_y_clamped,
                                 axis=0,
                                 loc=L("sin_y_gather"),
                                 ip=ip).output
            q_half_shape = [batch_size, max_patches, num_attention_heads, spatial_dim]
            k_half_shape = [batch_size, max_patches, num_key_value_heads, spatial_dim]
            cos_x_tiled = top.TileOp(T(q_half_shape),
                                     cos_x,
                                     tile=[1, 1, num_attention_heads, 1],
                                     loc=L("cos_x_tile"),
                                     ip=ip).output
            sin_x_tiled = top.TileOp(T(q_half_shape),
                                     sin_x,
                                     tile=[1, 1, num_attention_heads, 1],
                                     loc=L("sin_x_tile"),
                                     ip=ip).output
            cos_y_tiled = top.TileOp(T(q_half_shape),
                                     cos_y,
                                     tile=[1, 1, num_attention_heads, 1],
                                     loc=L("cos_y_tile"),
                                     ip=ip).output
            sin_y_tiled = top.TileOp(T(q_half_shape),
                                     sin_y,
                                     tile=[1, 1, num_attention_heads, 1],
                                     loc=L("sin_y_tile"),
                                     ip=ip).output
            cos_x_k_tiled = top.TileOp(T(k_half_shape),
                                       cos_x,
                                       tile=[1, 1, num_key_value_heads, 1],
                                       loc=L("cos_x_k_tile"),
                                       ip=ip).output
            sin_x_k_tiled = top.TileOp(T(k_half_shape),
                                       sin_x,
                                       tile=[1, 1, num_key_value_heads, 1],
                                       loc=L("sin_x_k_tile"),
                                       ip=ip).output
            cos_y_k_tiled = top.TileOp(T(k_half_shape),
                                       cos_y,
                                       tile=[1, 1, num_key_value_heads, 1],
                                       loc=L("cos_y_k_tile"),
                                       ip=ip).output
            sin_y_k_tiled = top.TileOp(T(k_half_shape),
                                       sin_y,
                                       tile=[1, 1, num_key_value_heads, 1],
                                       loc=L("sin_y_k_tile"),
                                       ip=ip).output

            # ====== Vision Encoder Layers ======
            for idx in range(vconfig.num_hidden_layers):
                layer_path = f"{tower_path}.encoder.layers.{idx}"
                residual_op = new_op

                new_op = self.rms_norm(vit_mlir, new_op, f"{layer_path}.input_layernorm")

                q_op = self._clippable_linear(
                    vit_mlir, f"{layer_path}.self_attn.q_proj", new_op,
                    [embed_dim, num_attention_heads * head_dim],
                    [batch_size, max_patches, num_attention_heads * head_dim])
                q_op = top.ReshapeOp(T([batch_size, max_patches, num_attention_heads, head_dim]),
                                     q_op,
                                     loc=L(f"{layer_path}.self_attn.q_reshape"),
                                     ip=ip).output
                q_op = self.rms_norm(vit_mlir, q_op, f"{layer_path}.self_attn.q_norm")

                k_op = self._clippable_linear(
                    vit_mlir, f"{layer_path}.self_attn.k_proj", new_op,
                    [embed_dim, num_key_value_heads * head_dim],
                    [batch_size, max_patches, num_key_value_heads * head_dim])
                k_op = top.ReshapeOp(T([batch_size, max_patches, num_key_value_heads, head_dim]),
                                     k_op,
                                     loc=L(f"{layer_path}.self_attn.k_reshape"),
                                     ip=ip).output
                k_op = self.rms_norm(vit_mlir, k_op, f"{layer_path}.self_attn.k_norm")

                v_op = self._clippable_linear(
                    vit_mlir, f"{layer_path}.self_attn.v_proj", new_op,
                    [embed_dim, num_key_value_heads * head_dim],
                    [batch_size, max_patches, num_key_value_heads * head_dim])
                v_op = top.ReshapeOp(T([batch_size, max_patches, num_key_value_heads, head_dim]),
                                     v_op,
                                     loc=L(f"{layer_path}.self_attn.v_reshape"),
                                     ip=ip).output
                v_op = self._rms_norm_no_scale(vit_mlir, v_op, f"{layer_path}.self_attn.v_norm")

                # 2D RoPE on q and k
                q_x = top.SliceOp(T(q_half_shape),
                                  q_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  offset=[0, 0, 0, 0],
                                  steps=[1, 1, 1, 1],
                                  ends=q_half_shape,
                                  axes=[],
                                  loc=L(f"{layer_path}.self_attn.q_x_slice"),
                                  ip=ip).output
                q_y = top.SliceOp(T(q_half_shape),
                                  q_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  offset=[0, 0, 0, spatial_dim],
                                  steps=[1, 1, 1, 1],
                                  ends=[batch_size, max_patches, num_attention_heads, head_dim],
                                  axes=[],
                                  loc=L(f"{layer_path}.self_attn.q_y_slice"),
                                  ip=ip).output
                q_x = self.rotary_pos(vit_mlir, q_x, cos_x_tiled, sin_x_tiled,
                                      f"{layer_path}.self_attn.q_x_rope")
                q_y = self.rotary_pos(vit_mlir, q_y, cos_y_tiled, sin_y_tiled,
                                      f"{layer_path}.self_attn.q_y_rope")
                q_op = top.ConcatOp(T([batch_size, max_patches, num_attention_heads, head_dim]),
                                    [q_x, q_y],
                                    axis=3,
                                    loc=L(f"{layer_path}.self_attn.q_rope_concat"),
                                    ip=ip).output

                k_x = top.SliceOp(T(k_half_shape),
                                  k_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  offset=[0, 0, 0, 0],
                                  steps=[1, 1, 1, 1],
                                  ends=k_half_shape,
                                  axes=[],
                                  loc=L(f"{layer_path}.self_attn.k_x_slice"),
                                  ip=ip).output
                k_y = top.SliceOp(T(k_half_shape),
                                  k_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  vit_mlir.none_op,
                                  offset=[0, 0, 0, spatial_dim],
                                  steps=[1, 1, 1, 1],
                                  ends=[batch_size, max_patches, num_key_value_heads, head_dim],
                                  axes=[],
                                  loc=L(f"{layer_path}.self_attn.k_y_slice"),
                                  ip=ip).output
                k_x = self.rotary_pos(vit_mlir, k_x, cos_x_k_tiled, sin_x_k_tiled,
                                      f"{layer_path}.self_attn.k_x_rope")
                k_y = self.rotary_pos(vit_mlir, k_y, cos_y_k_tiled, sin_y_k_tiled,
                                      f"{layer_path}.self_attn.k_y_rope")
                k_op = top.ConcatOp(T([batch_size, max_patches, num_key_value_heads, head_dim]),
                                    [k_x, k_y],
                                    axis=3,
                                    loc=L(f"{layer_path}.self_attn.k_rope_concat"),
                                    ip=ip).output

                fa_op = top.FAttentionOp(T(hidden_shape),
                                         q_op,
                                         k_op,
                                         v_op,
                                         attn_mask,
                                         vit_mlir.none_op,
                                         scale=1.0,
                                         batch=batch_size,
                                         q_head=num_attention_heads,
                                         kv_head=num_key_value_heads,
                                         dim=head_dim,
                                         mq=max_patches,
                                         mk=max_patches,
                                         keep_dims=False,
                                         loc=L(f"{layer_path}.fattention"),
                                         ip=ip).output
                o_op = self._clippable_linear(vit_mlir, f"{layer_path}.self_attn.o_proj", fa_op,
                                              [num_attention_heads * head_dim, embed_dim],
                                              hidden_shape)
                o_op = self.rms_norm(vit_mlir, o_op, f"{layer_path}.post_attention_layernorm")
                new_op = top.AddOp(T(hidden_shape), [residual_op, o_op],
                                   loc=L(f"{layer_path}.attn_residual"),
                                   ip=ip).output

                # MLP
                residual_op = new_op
                new_op = self.rms_norm(vit_mlir, new_op, f"{layer_path}.pre_feedforward_layernorm")

                gate_op = self._clippable_linear(
                    vit_mlir, f"{layer_path}.mlp.gate_proj", new_op,
                    [embed_dim, vconfig.intermediate_size],
                    [batch_size, max_patches, vconfig.intermediate_size])
                act_op = self.activate(vit_mlir, gate_op, hidden_act, layer_path)
                up_op = self._clippable_linear(vit_mlir, f"{layer_path}.mlp.up_proj", new_op,
                                               [embed_dim, vconfig.intermediate_size],
                                               [batch_size, max_patches, vconfig.intermediate_size])
                new_op = top.MulOp(T([batch_size, max_patches, vconfig.intermediate_size]),
                                   [act_op, up_op],
                                   loc=L(f"{layer_path}.mlp.mul"),
                                   ip=ip).output
                down_op = self._clippable_linear(vit_mlir, f"{layer_path}.mlp.down_proj", new_op,
                                                 [vconfig.intermediate_size, embed_dim],
                                                 hidden_shape)
                down_op = self.rms_norm(vit_mlir, down_op,
                                        f"{layer_path}.post_feedforward_layernorm")
                new_op = top.AddOp(T(hidden_shape), [residual_op, down_op],
                                   loc=L(f"{layer_path}.mlp_residual"),
                                   ip=ip).output

            # ====== Pooler ======
            new_op = top.MulOp(T(hidden_shape), [new_op, valid_mask_3d],
                               loc=L("pooler_masked_fill"),
                               ip=ip).output
            k = pooling_kernel_size
            k_squared = k * k

            div_k_table = vit_mlir.create_weight_op("pooler.div_k_table",
                                                    [position_embedding_size + 1])
            max_x_div_k_table = vit_mlir.create_weight_op("pooler.max_x_div_k_table",
                                                          [position_embedding_size])
            identity = vit_mlir.create_weight_op("pooler.identity", [mm_tokens, mm_tokens])

            pos_x_f32 = top.CastOp(T([batch_size, max_patches]),
                                   pos_x_clamped,
                                   round_mode=StringAttr.get("TowardsZero"),
                                   to="F32",
                                   loc=L("pos_x_f32"),
                                   ip=ip).output
            pos_y_f32 = top.CastOp(T([batch_size, max_patches]),
                                   pos_y_clamped,
                                   round_mode=StringAttr.get("TowardsZero"),
                                   to="F32",
                                   loc=L("pos_y_f32"),
                                   ip=ip).output

            x_group = top.GatherOp(T([batch_size, max_patches]),
                                   div_k_table,
                                   pos_x_f32,
                                   axis=0,
                                   loc=L("x_group"),
                                   ip=ip).output
            y_group = top.GatherOp(T([batch_size, max_patches]),
                                   div_k_table,
                                   pos_y_f32,
                                   axis=0,
                                   loc=L("y_group"),
                                   ip=ip).output

            max_x_val = top.ReduceOp(T([batch_size, 1]),
                                     pos_x_clamped,
                                     axes=[1],
                                     keepdims=1,
                                     mode=StringAttr.get("ReduceMax"),
                                     loc=L("max_x"),
                                     ip=ip).output
            max_x_val_f32 = top.CastOp(T([batch_size, 1]),
                                       max_x_val,
                                       round_mode=StringAttr.get("TowardsZero"),
                                       to="F32",
                                       loc=L("max_x_f32"),
                                       ip=ip).output
            max_x_grp = top.GatherOp(T([batch_size, 1]),
                                     max_x_div_k_table,
                                     max_x_val_f32,
                                     axis=0,
                                     loc=L("max_x_group"),
                                     ip=ip).output
            max_x_grp_3d = top.ReshapeOp(T([batch_size, 1, 1]),
                                         max_x_grp,
                                         shape=[batch_size, 1, 1],
                                         loc=L("max_x_group_3d"),
                                         ip=ip).output
            y_group_3d = top.ReshapeOp(T([batch_size, max_patches, 1]),
                                       y_group,
                                       shape=[batch_size, -1, 1],
                                       loc=L("y_group_3d"),
                                       ip=ip).output
            y_scaled = top.MulOp(T([batch_size, max_patches, 1]), [y_group_3d, max_x_grp_3d],
                                 loc=L("y_scaled"),
                                 ip=ip).output

            x_group_3d = top.ReshapeOp(T([batch_size, max_patches, 1]),
                                       x_group,
                                       shape=[batch_size, -1, 1],
                                       loc=L("x_group_3d"),
                                       ip=ip).output
            kernel_idx = top.AddOp(T([batch_size, max_patches, 1]), [x_group_3d, y_scaled],
                                   loc=L("kernel_idx"),
                                   ip=ip).output
            kernel_idx_2d = top.ReshapeOp(T([batch_size, max_patches]),
                                          kernel_idx,
                                          shape=[batch_size, -1],
                                          loc=L("kernel_idx_2d"),
                                          ip=ip).output

            one_hot = top.GatherOp(T([batch_size, max_patches, mm_tokens]),
                                   identity,
                                   kernel_idx_2d,
                                   axis=0,
                                   loc=L("one_hot"),
                                   ip=ip).output
            weights = top.MulConstOp(T([batch_size, max_patches, mm_tokens]),
                                     one_hot,
                                     const_val=1.0 / k_squared,
                                     loc=L("weights"),
                                     ip=ip).output
            weights_t = top.PermuteOp(T([batch_size, mm_tokens, max_patches]),
                                      weights,
                                      order=[0, 2, 1],
                                      loc=L("weights_transpose"),
                                      ip=ip).output
            pooled_shape = [batch_size, mm_tokens, embed_dim]
            new_op = top.MatMulOp(T(pooled_shape),
                                  weights_t,
                                  new_op,
                                  vit_mlir.none_op,
                                  loc=L("pooler_matmul"),
                                  ip=ip).output
            root_hidden_size = embed_dim**0.5
            new_op = top.MulConstOp(T([batch_size, mm_tokens, embed_dim]),
                                    new_op,
                                    const_val=root_hidden_size,
                                    loc=L("pooler_scale"),
                                    ip=ip).output

            # ====== MM Projector ======
            new_op = self._rms_norm_no_scale(vit_mlir, new_op, mm_projector_norm)
            new_op = self.linear(vit_mlir, mm_projector_mm, new_op, [embed_dim, self.hidden_size],
                                 out_shape)

            vit_mlir.create_return_op([new_op])
            mlir_txt = vit_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # Image: batch=1, mm_tokens=280, max_patches=2520
        _gen_vit_variant("vit_image", 1, self.config.vision_soft_tokens_per_image,
                         "vit_image_top_weights.npz")
        # Video: batch=32, mm_tokens=70, max_patches=630
        _gen_vit_variant("vit_video", 32, 70, "vit_video_top_weights.npz")

    def gen_audio_mlir(self):
        import math
        tqdm.write(f"generate audio mlir ...")
        name = "audio"
        aconfig = self.config.audio_config

        hidden_size = aconfig.hidden_size
        num_heads = aconfig.num_attention_heads
        head_dim = hidden_size // num_heads
        ffn_dim = hidden_size * 4
        num_layers = aconfig.num_hidden_layers
        num_mel_bins = aconfig.hidden_size  # same as hidden_size for mel features = 128 (feature_size)
        output_proj_dims = aconfig.output_proj_dims
        hidden_act = aconfig.hidden_act
        eps = aconfig.rms_norm_eps
        residual_weight = aconfig.residual_weight

        # Audio attention parameters
        chunk_size = aconfig.attention_chunk_size
        context_left = aconfig.attention_context_left
        context_right = aconfig.attention_context_right
        max_past_horizon = context_left - 1
        max_future_horizon = context_right
        context_size = chunk_size + max_past_horizon + max_future_horizon
        softcap = aconfig.attention_logit_cap
        gradient_clipping = float(aconfig.gradient_clipping)

        # SubSampleConvProjection channels
        subsample_channels = aconfig.subsampling_conv_channels

        # Sequence length configuration
        audio_seq_len = self.audio_length
        max_audio_frames = audio_seq_len * 4
        num_blocks = (audio_seq_len + chunk_size - 1) // chunk_size  # = 3
        padded_seq_len = num_blocks * chunk_size  # = 36
        max_rel_pos = context_left - 1  # = 12
        pad_amount = padded_seq_len - audio_seq_len  # = 11

        # Tower path prefix
        audio_path = "model.audio_tower"
        tower_path = f"{audio_path}"
        subsample_path = f"{tower_path}.subsample_conv_projection"

        # create weights file
        audio_npz = "audio_tower_top_weights.npz"
        weights_dict = {}

        def save_weights():
            # SubSampleConvProjection
            for layer_idx in range(2):
                layer_path = f"{subsample_path}.layer{layer_idx}"
                conv_key = f"{layer_path}.conv.weight"
                weights_dict[conv_key] = self.model.read(conv_key)
                norm_key = f"{layer_path}.norm.weight"
                weights_dict[norm_key] = self.model.read(norm_key)
                # LayerNorm bias=False, create zeros bias for LayerNormOp
                weights_dict[f"{layer_path}.norm.bias"] = np.zeros(
                    subsample_channels[layer_idx] if layer_idx == 0 else subsample_channels[1],
                    dtype=np.float32)
            self.set_linear_weight(f"{subsample_path}.input_proj_linear", weights_dict)

            # RelPositionalEncoding: precompute position embeddings (used for rel_key_states, not stored in npz)
            pos_enc = Gemma4AudioRelPositionalEncodingPrecompute(aconfig)

            # Precompute per-layer attention data
            for idx in range(num_layers):
                layer_path = f"{tower_path}.layers.{idx}"
                attn_path = f"{layer_path}.self_attn"

                # ClippableLinear weights for attention
                for proj_name in ["q_proj", "k_proj", "v_proj", "post"]:
                    self._set_clippable_linear_weight(f"{attn_path}.{proj_name}", weights_dict)

                # relative_k_proj: compute rel_key_states for dynamic MatMul
                rel_k_weight_raw = self.model.read(f"{attn_path}.relative_k_proj.weight")
                rel_k_data = np.ascontiguousarray(np.transpose(rel_k_weight_raw, (1, 0)))

                # per_dim_scale: precompute softplus(per_dim_scale) * q_scale
                per_dim_scale_data = self.model.read(f"{attn_path}.per_dim_scale")
                q_scale = (head_dim**-0.5) / math.log(2)
                combined_scale = q_scale * torch.nn.functional.softplus(
                    torch.tensor(per_dim_scale_data, dtype=torch.float32)).numpy()
                weights_dict[f"{attn_path}.per_dim_scale_combined.weight"] = combined_scale.reshape(
                    [1, 1, 1, head_dim])

                # Compute rel_key_states for dynamic blocked attention
                rel_k_proj_out = np.matmul(pos_enc, rel_k_data)
                rel_k_proj_out = rel_k_proj_out.reshape(-1, num_heads, head_dim)
                rel_key_states = np.ascontiguousarray(
                    np.transpose(rel_k_proj_out,
                                 (1, 2, 0)).reshape(1, num_heads, head_dim,
                                                    max_rel_pos + 1)).astype(np.float32)
                weights_dict[f"{attn_path}.rel_key_states.weight"] = rel_key_states

                # Precompute 5D blocked attention mask [1, 1, num_blocks, chunk_size, context_size]
                blocked_mask = _compute_blocked_attn_mask(audio_seq_len, padded_seq_len, num_blocks,
                                                          chunk_size, context_size,
                                                          max_past_horizon, max_future_horizon)
                weights_dict[f"{attn_path}.blocked_attn_mask.weight"] = blocked_mask
                # Precompute mask_bias: invalid_val for masked positions, 0 for unmasked
                invalid_val = float(aconfig.attention_invalid_logits_value)
                mask_bias = np.where(blocked_mask == 0, invalid_val, 0.0).astype(np.float32)
                weights_dict[f"{attn_path}.blocked_attn_mask_bias.weight"] = mask_bias

                # k_scale constant
                k_scale = math.log(1 + math.e) / math.log(2)

                # Norms for attention path
                self.set_common_weight(f"{layer_path}.norm_pre_attn", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.norm_post_attn", weights_dict,
                                       self.rmsnorm_type)

                # LightConv1d
                lconv_path = f"{layer_path}.lconv1d"
                self.set_common_weight(f"{lconv_path}.pre_layer_norm", weights_dict,
                                       self.rmsnorm_type)
                self._set_clippable_linear_weight(f"{lconv_path}.linear_start", weights_dict)
                self._set_clippable_linear_weight(f"{lconv_path}.linear_end", weights_dict)
                conv1d_key = f"{lconv_path}.depthwise_conv1d.weight"
                _conv1d_data = self.model.read(conv1d_key)
                weights_dict[conv1d_key] = _conv1d_data
                self.set_common_weight(f"{lconv_path}.conv_norm", weights_dict, self.rmsnorm_type)

                # FeedForward1 and FeedForward2
                for ff_idx in [1, 2]:
                    ff_path = f"{layer_path}.feed_forward{ff_idx}"
                    self.set_common_weight(f"{ff_path}.pre_layer_norm", weights_dict,
                                           self.rmsnorm_type)
                    self.set_common_weight(f"{ff_path}.post_layer_norm", weights_dict,
                                           self.rmsnorm_type)
                    self._set_clippable_linear_weight(f"{ff_path}.ffw_layer_1", weights_dict)
                    self._set_clippable_linear_weight(f"{ff_path}.ffw_layer_2", weights_dict)

                # norm_out
                self.set_common_weight(f"{layer_path}.norm_out", weights_dict, self.rmsnorm_type)

            # output_proj (nn.Linear with bias)
            self.set_linear_weight(f"{tower_path}.output_proj", weights_dict)

            # embed_audio: RMSNorm(no_scale) + Linear
            embed_path = "model.embed_audio"
            weights_dict[f"{embed_path}.embedding_pre_projection_norm.weight"] = np.ones(
                output_proj_dims, dtype=np.float32)
            self.set_linear_weight(f"{embed_path}.embedding_projection", weights_dict)

            np.savez(audio_npz, **weights_dict)

        def Gemma4AudioRelPositionalEncodingPrecompute(config):
            """Precompute sinusoidal relative positional encoding following source code."""
            hidden_size = config.hidden_size
            min_timescale = 1.0
            max_timescale = 10000.0
            num_timescales = hidden_size // 2
            log_timescale_increment = math.log(max_timescale / min_timescale) / max(
                num_timescales - 1, 1)
            inv_timescales = min_timescale * np.exp(
                -log_timescale_increment * np.arange(num_timescales))
            # Use max_relative_position from config if available, otherwise default to 12
            max_rel_pos = getattr(config, 'max_relative_position', 12)
            position_ids = np.arange(max_rel_pos, -1, -1, dtype=np.float32)[..., None]
            scaled_time = position_ids * inv_timescales[None, :]
            pos_embed = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=-1)
            return pos_embed.astype(np.float32)  # [max_rel_pos+1, hidden_size]

        def _compute_blocked_attn_mask(real_seq_len, padded_seq_len, num_blocks, chunk_size,
                                       context_size, past_horizon, future_horizon):
            """5D blocked mask [1, 1, num_blocks, chunk_size, context_size].
            For each block b, each query q_in_block, each key position k_in_context:
            mask=1 if key is a valid real token AND within sliding window of query.
            """
            mask = np.zeros([1, 1, num_blocks, chunk_size, context_size], dtype=np.float32)
            for b in range(num_blocks):
                for q_in_block in range(chunk_size):
                    q_real = b * chunk_size + q_in_block
                    if q_real >= real_seq_len:
                        continue
                    for k_in_context in range(context_size):
                        k_padded48 = b * chunk_size + k_in_context
                        k_real = k_padded48 - past_horizon
                        if k_real >= 0 and k_real < real_seq_len:
                            if k_real >= q_real - past_horizon and k_real <= q_real + future_horizon:
                                mask[0, 0, b, q_in_block, k_in_context] = 1.0
            return mask

        save_weights()

        # === Generate MLIR ===
        # Input: [1, 1, feature_size(128), max_audio_frames]
        feature_size = aconfig.hidden_size if hasattr(aconfig, 'feature_size') else 128
        # Actually feature_size comes from config, but for Gemma4Audio it's always 128 (num_mel_bins)
        # Check config: audio_config doesn't have feature_size, it has hidden_size=1024
        # The input to audio model is [1, seq_len, 128] from the feature extractor
        # After unsqueeze(1): [1, 1, seq_len, 128] for Conv2D
        # feature_size for conv input = 128
        conv_in_channels = 1
        conv_feature_size = 128  # mel bins

        in_shape = [1, conv_in_channels, max_audio_frames, conv_feature_size]
        out_shape = [1, audio_seq_len, self.hidden_size]

        audio_mlir = MLIRImporter([in_shape], [out_shape],
                                  name,
                                  Platform.LLM, ["F32"],
                                  weight_file=f"../{audio_npz}")
        ip = audio_mlir.insert_point

        def T(shape: list):
            return audio_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, audio_mlir)

        in0_op = audio_mlir.create_input_op(L('input_features'), 0)

        # ====== SubSampleConvProjection ======
        new_op = in0_op  # [1, 1, max_audio_frames, conv_feature_size]

        # Get conv channel dimensions from config or use defaults
        conv0_out_channels = subsample_channels[0] if hasattr(subsample_channels,
                                                              '__getitem__') else 128
        conv1_out_channels = subsample_channels[1] if hasattr(subsample_channels,
                                                              '__getitem__') else 32

        # layer0: Conv2D(1→conv0_out_channels, 3x3, stride=2x2, pad=1x1, no bias)
        conv0_h = (max_audio_frames - 1) // 2 + 1
        conv0_w = (conv_feature_size - 1) // 2 + 1
        conv0_weight = audio_mlir.create_weight_op(f"{subsample_path}.layer0.conv.weight",
                                                   [conv0_out_channels, 1, 3, 3])
        conv0_op = top.ConvOp(T([1, conv0_out_channels, conv0_h, conv0_w]),
                              new_op,
                              conv0_weight,
                              audio_mlir.none_op,
                              kernel_shape=[3, 3],
                              strides=[2, 2],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              loc=L(f"{subsample_path}.layer0.conv"),
                              ip=ip).output
        # layer0: LayerNorm + ReLU
        # LayerNorm on NHWC format - need to permute
        conv0_op = top.PermuteOp(T([1, conv0_h, conv0_w, conv0_out_channels]),
                                 conv0_op,
                                 order=[0, 2, 3, 1],
                                 loc=L(f"{subsample_path}.layer0.permute"),
                                 ip=ip).output
        ln0_weight = audio_mlir.create_weight_op(f"{subsample_path}.layer0.norm.weight",
                                                 [conv0_out_channels])
        ln0_bias = audio_mlir.create_weight_op(f"{subsample_path}.layer0.norm.bias",
                                               [conv0_out_channels])
        conv0_op = top.LayerNormOp(T([1, conv0_h, conv0_w, conv0_out_channels]),
                                   conv0_op,
                                   ln0_weight,
                                   ln0_bias,
                                   normalized_shape=[conv0_out_channels],
                                   axis=3,
                                   eps=eps,
                                   loc=L(f"{subsample_path}.layer0.norm"),
                                   ip=ip).output
        # Permute back to NCHW
        conv0_op = top.PermuteOp(T([1, conv0_out_channels, conv0_h, conv0_w]),
                                 conv0_op,
                                 order=[0, 3, 1, 2],
                                 loc=L(f"{subsample_path}.layer0.permute_back"),
                                 ip=ip).output
        conv0_op = self.activate(audio_mlir, conv0_op, ActType.RELU, f"{subsample_path}.layer0")

        # layer1: Conv2D(conv0_out_channels→conv1_out_channels, 3x3, stride=2x2, pad=1x1, no bias)
        conv1_h = (conv0_h - 1) // 2 + 1
        conv1_w = (conv0_w - 1) // 2 + 1
        conv1_weight = audio_mlir.create_weight_op(f"{subsample_path}.layer1.conv.weight",
                                                   [conv1_out_channels, conv0_out_channels, 3, 3])
        conv1_op = top.ConvOp(T([1, conv1_out_channels, conv1_h, conv1_w]),
                              conv0_op,
                              conv1_weight,
                              audio_mlir.none_op,
                              kernel_shape=[3, 3],
                              strides=[2, 2],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              loc=L(f"{subsample_path}.layer1.conv"),
                              ip=ip).output
        # layer1: LayerNorm + ReLU
        conv1_op = top.PermuteOp(T([1, conv1_h, conv1_w, conv1_out_channels]),
                                 conv1_op,
                                 order=[0, 2, 3, 1],
                                 loc=L(f"{subsample_path}.layer1.permute"),
                                 ip=ip).output
        ln1_weight = audio_mlir.create_weight_op(f"{subsample_path}.layer1.norm.weight",
                                                 [conv1_out_channels])
        ln1_bias = audio_mlir.create_weight_op(f"{subsample_path}.layer1.norm.bias",
                                               [conv1_out_channels])
        conv1_op = top.LayerNormOp(T([1, conv1_h, conv1_w, conv1_out_channels]),
                                   conv1_op,
                                   ln1_weight,
                                   ln1_bias,
                                   normalized_shape=[conv1_out_channels],
                                   axis=3,
                                   eps=eps,
                                   loc=L(f"{subsample_path}.layer1.norm"),
                                   ip=ip).output
        # Permute back and reshape for input_proj_linear
        conv1_op = top.PermuteOp(T([1, conv1_out_channels, conv1_h, conv1_w]),
                                 conv1_op,
                                 order=[0, 3, 1, 2],
                                 loc=L(f"{subsample_path}.layer1.permute_back"),
                                 ip=ip).output
        conv1_op = self.activate(audio_mlir, conv1_op, ActType.RELU, f"{subsample_path}.layer1")
        # Reshape: [1, conv1_out_channels, conv1_h, conv1_w] → [1, audio_seq_len, proj_input_dim]
        proj_input_dim = conv1_out_channels * conv1_w
        new_op = top.PermuteOp(T([1, conv1_h, conv1_out_channels, conv1_w]),
                               conv1_op,
                               order=[0, 2, 3, 1],
                               loc=L(f"{subsample_path}.reshape_permute"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, audio_seq_len, proj_input_dim]),
                               new_op,
                               loc=L(f"{subsample_path}.reshape"),
                               ip=ip).output
        # input_proj_linear: [proj_input_dim, hidden_size] → [1, audio_seq_len, hidden_size]
        hidden_shape = [1, audio_seq_len, hidden_size]
        new_op = self.linear(audio_mlir, f"{subsample_path}.input_proj_linear", new_op,
                             [proj_input_dim, hidden_size], hidden_shape)

        # ====== AudioLayers ======
        k_scale = math.log(1 + math.e) / math.log(2)
        softcap_val = float(softcap)

        def audio_feed_forward(ff_idx, in_op, layer_path_str):
            """Gemma4AudioFeedForward: clamp → pre_norm → ffw1 → act → ffw2 → clamp → post_norm → *residual_weight → +residual"""
            residual = in_op
            ff_path = f"{layer_path_str}.feed_forward{ff_idx}"

            # Skip clamp since gradient_clipping=1e10 (effectively no clamping)

            # pre_layer_norm
            new_op = self.rms_norm(audio_mlir, in_op, f"{ff_path}.pre_layer_norm")

            # ffw_layer_1 (ClippableLinear)
            ffw1_path = f"{ff_path}.ffw_layer_1"
            new_op = self._clippable_linear(audio_mlir, ffw1_path, new_op, [hidden_size, ffn_dim],
                                            [1, audio_seq_len, ffn_dim])

            # activation (silu)
            new_op = self.activate(audio_mlir, new_op, hidden_act, ff_path)

            # ffw_layer_2 (ClippableLinear)
            ffw2_path = f"{ff_path}.ffw_layer_2"
            new_op = self._clippable_linear(audio_mlir, ffw2_path, new_op, [ffn_dim, hidden_size],
                                            hidden_shape)

            # Skip clamp again

            # post_layer_norm
            new_op = self.rms_norm(audio_mlir, new_op, f"{ff_path}.post_layer_norm")

            # * residual_weight (0.5)
            new_op = top.MulConstOp(T(hidden_shape),
                                    new_op,
                                    const_val=residual_weight,
                                    loc=L(f"{ff_path}.residual_scale"),
                                    ip=ip).output

            # + residual
            new_op = top.AddOp(T(hidden_shape), [residual, new_op],
                               loc=L(f"{ff_path}.residual_add"),
                               ip=ip).output
            return new_op

        def audio_light_conv1d(in_op, layer_path_str):
            """Gemma4AudioLightConv1d: pre_norm → linear_start → GLU → causal_conv1d → clamp → conv_norm → act → linear_end → +residual"""
            residual = in_op
            lconv_path = f"{layer_path_str}.lconv1d"

            # pre_layer_norm
            new_op = self.rms_norm(audio_mlir, in_op, f"{lconv_path}.pre_layer_norm")

            # linear_start (ClippableLinear, hidden_size → hidden_size*2=2048)
            new_op = self._clippable_linear(audio_mlir, f"{lconv_path}.linear_start", new_op,
                                            [hidden_size, hidden_size * 2],
                                            [1, audio_seq_len, hidden_size * 2])

            # GLU: F.glu(x, dim=-1) = sigmoid(x[..., half:]) * x[..., :half]
            # Input: [1, audio_seq_len, hidden_size*2] → split → [1, audio_seq_len, hidden_size] each
            new_op_shape = [1, audio_seq_len, hidden_size * 2]
            gate_shape = [1, audio_seq_len, hidden_size]
            value_op = top.SliceOp(T(gate_shape),
                                   new_op,
                                   audio_mlir.none_op,
                                   audio_mlir.none_op,
                                   audio_mlir.none_op,
                                   offset=[0, 0, 0],
                                   steps=[1, 1, 1],
                                   ends=gate_shape,
                                   axes=[],
                                   loc=L(f"{lconv_path}.value_slice"),
                                   ip=ip).output
            gate_op = top.SliceOp(T(gate_shape),
                                  new_op,
                                  audio_mlir.none_op,
                                  audio_mlir.none_op,
                                  audio_mlir.none_op,
                                  offset=[0, 0, hidden_size],
                                  steps=[1, 1, 1],
                                  ends=new_op_shape,
                                  axes=[],
                                  loc=L(f"{lconv_path}.gate_slice"),
                                  ip=ip).output
            gate_op = top.SigmoidOp(T(gate_shape),
                                    gate_op,
                                    loc=L(f"{lconv_path}.glu_sigmoid"),
                                    ip=ip).output
            new_op = top.MulOp(T(gate_shape), [gate_op, value_op],
                               loc=L(f"{lconv_path}.glu_mul"),
                               ip=ip).output

            # Causal Conv1d: depthwise_conv1d with left_pad
            # Input: [1, audio_seq_len, hidden_size] → transpose → [1, hidden_size, audio_seq_len]
            # Conv1d: groups=hidden_size, kernel_size=5, causal padding (left_pad=4)
            conv1d_in_shape = [1, hidden_size, audio_seq_len]
            new_op = top.PermuteOp(T(conv1d_in_shape),
                                   new_op,
                                   order=[0, 2, 1],
                                   loc=L(f"{lconv_path}.conv1d_permute_in"),
                                   ip=ip).output
            # Reshape to 4D for ConvOp: [1, hidden_size, 1, audio_seq_len]
            new_op = top.ReshapeOp(T([1, hidden_size, 1, audio_seq_len]),
                                   new_op,
                                   loc=L(f"{lconv_path}.conv1d_reshape_in"),
                                   ip=ip).output
            conv_kernel_size = aconfig.conv_kernel_size
            left_pad = conv_kernel_size - 1  # (kernel_size-1)*dilation+1-stride = 4 for kernel=5,dilation=1,stride=1
            conv_out_shape = [1, hidden_size, 1, audio_seq_len]
            conv1d_weight = audio_mlir.create_weight_op(f"{lconv_path}.depthwise_conv1d.weight",
                                                        [hidden_size, 1, 1, conv_kernel_size])
            conv1d_op = top.ConvOp(T(conv_out_shape),
                                   new_op,
                                   conv1d_weight,
                                   audio_mlir.none_op,
                                   kernel_shape=[1, conv_kernel_size],
                                   strides=[1, 1],
                                   group=hidden_size,
                                   pads=[0, left_pad, 0, 0],
                                   loc=L(f"{lconv_path}.depthwise_conv1d"),
                                   ip=ip).output
            # Reshape back: [1, hidden_size, 1, audio_seq_len] → [1, hidden_size, audio_seq_len]
            conv1d_op = top.ReshapeOp(T(conv1d_in_shape),
                                      conv1d_op,
                                      loc=L(f"{lconv_path}.conv1d_reshape_out"),
                                      ip=ip).output
            # Transpose back: [1, hidden_size, audio_seq_len] → [1, audio_seq_len, hidden_size]
            new_op = top.PermuteOp(T(hidden_shape),
                                   conv1d_op,
                                   order=[0, 2, 1],
                                   loc=L(f"{lconv_path}.conv1d_permute_out"),
                                   ip=ip).output

            # Skip clamp (gradient_clipping=1e10)

            # conv_norm (RMSNorm)
            new_op = self.rms_norm(audio_mlir, new_op, f"{lconv_path}.conv_norm")

            # activation (silu)
            new_op = self.activate(audio_mlir, new_op, hidden_act, lconv_path)

            # linear_end (ClippableLinear, hidden_size → hidden_size)
            new_op = self._clippable_linear(audio_mlir, f"{lconv_path}.linear_end", new_op,
                                            [hidden_size, hidden_size], hidden_shape)

            # + residual
            new_op = top.AddOp(T(hidden_shape), [residual, new_op],
                               loc=L(f"{lconv_path}.residual_add"),
                               ip=ip).output
            return new_op

        def audio_attention(in_op, layer_path_str):
            """Gemma4AudioAttention: blocked 5D chunked local attention with dynamic relative pos bias and softcapping."""
            attn_path = f"{layer_path_str}.self_attn"

            # q_proj, k_proj, v_proj (ClippableLinear)
            q_op = self._clippable_linear(audio_mlir, f"{attn_path}.q_proj", in_op,
                                          [hidden_size, hidden_size],
                                          [1, audio_seq_len, hidden_size])
            k_op = self._clippable_linear(audio_mlir, f"{attn_path}.k_proj", in_op,
                                          [hidden_size, hidden_size],
                                          [1, audio_seq_len, hidden_size])
            v_op = self._clippable_linear(audio_mlir, f"{attn_path}.v_proj", in_op,
                                          [hidden_size, hidden_size],
                                          [1, audio_seq_len, hidden_size])

            # Reshape to [1, seq_len, num_heads, head_dim]
            qkv_shape = [1, audio_seq_len, num_heads, head_dim]
            q_op = top.ReshapeOp(T(qkv_shape), q_op, loc=L(f"{attn_path}.q_reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(qkv_shape), k_op, loc=L(f"{attn_path}.k_reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(qkv_shape), v_op, loc=L(f"{attn_path}.v_reshape"), ip=ip).output

            # q *= per_dim_scale_combined (includes q_scale * softplus(per_dim_scale))
            scale_weight = audio_mlir.create_weight_op(f"{attn_path}.per_dim_scale_combined.weight",
                                                       [1, 1, 1, head_dim])
            q_op = top.MulOp(T(qkv_shape), [q_op, scale_weight],
                             loc=L(f"{attn_path}.q_scale"),
                             ip=ip).output

            # k *= k_scale
            k_op = top.MulConstOp(T(qkv_shape),
                                  k_op,
                                  const_val=k_scale,
                                  loc=L(f"{attn_path}.k_scale"),
                                  ip=ip).output

            # ===== Blocked attention =====
            # _convert_to_block(q): pad seq_len to padded_seq_len then reshape
            q_pad_shape = [1, padded_seq_len, num_heads, head_dim]
            q_padded = top.PadOp(
                T(q_pad_shape),
                q_op,
                # [left_dims, right_dims]: left=[0,0,0,0], right=[0,pad_amount,0,0]
                paddings=[0, 0, 0, 0, 0, pad_amount, 0, 0],
                val=0.0,
                mode=StringAttr.get("constant"),
                loc=L(f"{attn_path}.q_pad"),
                ip=ip).output
            q_blocked_shape = [1, num_blocks, chunk_size, num_heads, head_dim]
            q_blocked = top.ReshapeOp(T(q_blocked_shape),
                                      q_padded,
                                      loc=L(f"{attn_path}.q_block"),
                                      ip=ip).output

            # _extract_block_context(k/v): pad left by max_past_horizon, right by max_future_horizon+chunk_size-1
            # then extract overlapping context windows via slice + concat
            kv_left_pad = max_past_horizon
            kv_right_pad = max_future_horizon + chunk_size - 1
            kv_padded_len = audio_seq_len + kv_left_pad + kv_right_pad  # = 48
            k_pad_shape = [1, kv_padded_len, num_heads, head_dim]
            k_padded = top.PadOp(
                T(k_pad_shape),
                k_op,
                # left=[0,kv_left_pad,0,0], right=[0,kv_right_pad,0,0]
                paddings=[0, kv_left_pad, 0, 0, 0, kv_right_pad, 0, 0],
                val=0.0,
                mode=StringAttr.get("constant"),
                loc=L(f"{attn_path}.k_context_pad"),
                ip=ip).output
            v_padded = top.PadOp(
                T(k_pad_shape),
                v_op,
                # left=[0,kv_left_pad,0,0], right=[0,kv_right_pad,0,0]
                paddings=[0, kv_left_pad, 0, 0, 0, kv_right_pad, 0, 0],
                val=0.0,
                mode=StringAttr.get("constant"),
                loc=L(f"{attn_path}.v_context_pad"),
                ip=ip).output

            # Extract context windows for each block using SliceOp
            k_slices = []
            v_slices = []
            ctx_shape = [1, context_size, num_heads, head_dim]
            for b in range(num_blocks):
                offset_start = b * chunk_size
                offset_end = offset_start + context_size
                k_slice = top.SliceOp(T(ctx_shape),
                                      k_padded,
                                      audio_mlir.none_op,
                                      audio_mlir.none_op,
                                      audio_mlir.none_op,
                                      offset=[0, offset_start, 0, 0],
                                      steps=[1, 1, 1, 1],
                                      ends=[1, offset_end, num_heads, head_dim],
                                      axes=[],
                                      loc=L(f"{attn_path}.k_ctx_slice_{b}"),
                                      ip=ip).output
                v_slice = top.SliceOp(T(ctx_shape),
                                      v_padded,
                                      audio_mlir.none_op,
                                      audio_mlir.none_op,
                                      audio_mlir.none_op,
                                      offset=[0, offset_start, 0, 0],
                                      steps=[1, 1, 1, 1],
                                      ends=[1, offset_end, num_heads, head_dim],
                                      axes=[],
                                      loc=L(f"{attn_path}.v_ctx_slice_{b}"),
                                      ip=ip).output
                # Reshape each slice to [1, 1, context_size, num_heads, head_dim] for concat
                k_slice_r = top.ReshapeOp(T([1, 1, context_size, num_heads, head_dim]),
                                          k_slice,
                                          loc=L(f"{attn_path}.k_ctx_reshape_{b}"),
                                          ip=ip).output
                v_slice_r = top.ReshapeOp(T([1, 1, context_size, num_heads, head_dim]),
                                          v_slice,
                                          loc=L(f"{attn_path}.v_ctx_reshape_{b}"),
                                          ip=ip).output
                k_slices.append(k_slice_r)
                v_slices.append(v_slice_r)

            # Concat along axis 1 → [1, num_blocks, context_size, num_heads, head_dim]
            k_blocked_shape = [1, num_blocks, context_size, num_heads, head_dim]
            k_blocked = top.ConcatOp(T(k_blocked_shape),
                                     k_slices,
                                     axis=1,
                                     loc=L(f"{attn_path}.k_blocked_concat"),
                                     ip=ip).output
            v_blocked = top.ConcatOp(T(k_blocked_shape),
                                     v_slices,
                                     axis=1,
                                     loc=L(f"{attn_path}.v_blocked_concat"),
                                     ip=ip).output

            # Blocked q@k^T: permute q to [1, num_heads, num_blocks, chunk_size, head_dim]
            # and k to [1, num_heads, num_blocks, head_dim, context_size]
            q_perm_shape = [1, num_heads, num_blocks, chunk_size, head_dim]
            q_perm = top.PermuteOp(T(q_perm_shape),
                                   q_blocked,
                                   order=[0, 3, 1, 2, 4],
                                   loc=L(f"{attn_path}.q_perm"),
                                   ip=ip).output
            k_perm_shape = [1, num_heads, num_blocks, head_dim, context_size]
            k_perm = top.PermuteOp(T(k_perm_shape),
                                   k_blocked,
                                   order=[0, 3, 1, 4, 2],
                                   loc=L(f"{attn_path}.k_perm"),
                                   ip=ip).output
            v_perm_shape = [1, num_heads, num_blocks, context_size, head_dim]
            v_perm = top.PermuteOp(T(v_perm_shape),
                                   v_blocked,
                                   order=[0, 3, 1, 2, 4],
                                   loc=L(f"{attn_path}.v_perm"),
                                   ip=ip).output

            # Flatten batch dims for 4D MatMul: merge num_heads*num_blocks
            batch_flat = num_heads * num_blocks  # = 24
            q_flat = top.ReshapeOp(T([1, batch_flat, chunk_size, head_dim]),
                                   q_perm,
                                   loc=L(f"{attn_path}.q_flat"),
                                   ip=ip).output
            k_flat = top.ReshapeOp(T([1, batch_flat, head_dim, context_size]),
                                   k_perm,
                                   loc=L(f"{attn_path}.k_flat"),
                                   ip=ip).output

            # matrix_ac = q @ k^T → [1, batch_flat, chunk_size, context_size]
            matrix_ac_flat = top.MatMulOp(T([1, batch_flat, chunk_size, context_size]),
                                          q_flat,
                                          k_flat,
                                          audio_mlir.none_op,
                                          loc=L(f"{attn_path}.qk_matmul"),
                                          ip=ip).output
            # Reshape back to 5D: [1, num_heads, num_blocks, chunk_size, context_size]
            ac_5d_shape = [1, num_heads, num_blocks, chunk_size, context_size]
            matrix_ac = top.ReshapeOp(T(ac_5d_shape),
                                      matrix_ac_flat,
                                      loc=L(f"{attn_path}.matrix_ac"),
                                      ip=ip).output

            # ===== Dynamic relative position bias (matrix_bd) =====
            # rel_key_states weight: [1, num_heads, head_dim, max_rel_pos+1]
            rel_key_weight = audio_mlir.create_weight_op(f"{attn_path}.rel_key_states.weight",
                                                         [1, num_heads, head_dim, max_rel_pos + 1])
            # queries_flat: [1, num_heads, padded_seq_len, head_dim]
            queries_flat = top.ReshapeOp(T([1, num_heads, padded_seq_len, head_dim]),
                                         q_perm,
                                         loc=L(f"{attn_path}.queries_flat"),
                                         ip=ip).output
            # matrix_bd = queries_flat @ rel_key_states → [1, num_heads, padded_seq_len, max_rel_pos+1]
            max_rel_plus_1 = max_rel_pos + 1  # = 13
            matrix_bd_flat = top.MatMulOp(T([1, num_heads, padded_seq_len, max_rel_plus_1]),
                                          queries_flat,
                                          rel_key_weight,
                                          audio_mlir.none_op,
                                          loc=L(f"{attn_path}.matrix_bd"),
                                          ip=ip).output

            # ===== _rel_shift =====
            # Pad last dim from max_rel_pos+1 to context_size+1 on 4D tensor (avoid 5D PadOp)
            # matrix_bd_flat is [1, num_heads, padded_seq_len, max_rel_plus_1]
            # Pad dim3 (last) from max_rel_plus_1=13 to cs_plus_1=25
            cs_plus_1 = context_size + 1  # = 25
            matrix_bd_padded_flat = top.PadOp(
                T([1, num_heads, padded_seq_len, cs_plus_1]),
                matrix_bd_flat,
                paddings=[0, 0, 0, 0, 0, 0, 0, cs_plus_1 - max_rel_plus_1],
                val=0.0,
                mode=StringAttr.get("constant"),
                loc=L(f"{attn_path}.rel_shift_pad"),
                ip=ip).output
            # Reshape to 5D: [1, num_heads, num_blocks, chunk_size, cs_plus_1]
            matrix_bd_padded_5d = top.ReshapeOp(T([1, num_heads, num_blocks, chunk_size,
                                                   cs_plus_1]),
                                                matrix_bd_padded_flat,
                                                loc=L(f"{attn_path}.rel_shift_5d"),
                                                ip=ip).output
            # Reshape: [1, num_heads, num_blocks, chunk_size * (context_size+1)]
            cs_times_chunk_plus = chunk_size * cs_plus_1  # = 300
            matrix_bd_shift = top.ReshapeOp(T([1, num_heads, num_blocks, cs_times_chunk_plus]),
                                            matrix_bd_padded_5d,
                                            loc=L(f"{attn_path}.rel_shift_reshape1"),
                                            ip=ip).output
            # Slice: truncate to chunk_size * context_size
            cs_times_chunk = chunk_size * context_size  # = 288
            matrix_bd_trunc = top.SliceOp(T([1, num_heads, num_blocks, cs_times_chunk]),
                                          matrix_bd_shift,
                                          audio_mlir.none_op,
                                          audio_mlir.none_op,
                                          audio_mlir.none_op,
                                          offset=[0, 0, 0, 0],
                                          steps=[1, 1, 1, 1],
                                          ends=[1, num_heads, num_blocks, cs_times_chunk],
                                          axes=[],
                                          loc=L(f"{attn_path}.rel_shift_slice"),
                                          ip=ip).output
            # Reshape back: [1, num_heads, num_blocks, chunk_size, context_size]
            matrix_bd_final = top.ReshapeOp(T(ac_5d_shape),
                                            matrix_bd_trunc,
                                            loc=L(f"{attn_path}.matrix_bd_final"),
                                            ip=ip).output

            # ===== Combine + softcap + mask + softmax =====
            # attn_weights = matrix_ac + matrix_bd
            attn_weights = top.AddOp(T(ac_5d_shape), [matrix_ac, matrix_bd_final],
                                     loc=L(f"{attn_path}.ac_bd_add"),
                                     ip=ip).output

            # Softcapping: /softcap → tanh → *softcap
            attn_weights = top.MulConstOp(T(ac_5d_shape),
                                          attn_weights,
                                          const_val=1.0 / softcap_val,
                                          loc=L(f"{attn_path}.softcap_div"),
                                          ip=ip).output
            attn_weights = top.TanhOp(T(ac_5d_shape),
                                      attn_weights,
                                      loc=L(f"{attn_path}.softcap_tanh"),
                                      ip=ip).output
            attn_weights = top.MulConstOp(T(ac_5d_shape),
                                          attn_weights,
                                          const_val=softcap_val,
                                          loc=L(f"{attn_path}.softcap_mul"),
                                          ip=ip).output

            # Apply 5D blocked mask: attn * mask + mask_bias
            # mask: 1 for unmasked, 0 for masked; mask_bias: 0 for unmasked, invalid_val for masked
            # This avoids bf16 precision loss from large-value shift/mul/unshift
            mask_weight = audio_mlir.create_weight_op(f"{attn_path}.blocked_attn_mask.weight",
                                                      [1, 1, num_blocks, chunk_size, context_size])
            mask_bias = audio_mlir.create_weight_op(f"{attn_path}.blocked_attn_mask_bias.weight",
                                                    [1, 1, num_blocks, chunk_size, context_size])
            attn_masked = top.MulOp(T(ac_5d_shape), [attn_weights, mask_weight],
                                    loc=L(f"{attn_path}.mask_mul"),
                                    ip=ip).output
            attn_weights = top.AddOp(T(ac_5d_shape), [attn_masked, mask_bias],
                                     loc=L(f"{attn_path}.mask_add"),
                                     ip=ip).output

            # Softmax over context_size (axis=4 for 5D)
            attn_weights = top.SoftmaxOp(T(ac_5d_shape),
                                         attn_weights,
                                         axis=4,
                                         loc=L(f"{attn_path}.softmax"),
                                         ip=ip).output

            # ===== Blocked attn @ v MatMul =====
            # Flatten batch dims for 4D MatMul
            attn_flat = top.ReshapeOp(T([1, batch_flat, chunk_size, context_size]),
                                      attn_weights,
                                      loc=L(f"{attn_path}.attn_flat"),
                                      ip=ip).output
            v_flat = top.ReshapeOp(T([1, batch_flat, context_size, head_dim]),
                                   v_perm,
                                   loc=L(f"{attn_path}.v_flat"),
                                   ip=ip).output
            attn_out_flat = top.MatMulOp(T([1, batch_flat, chunk_size, head_dim]),
                                         attn_flat,
                                         v_flat,
                                         audio_mlir.none_op,
                                         loc=L(f"{attn_path}.attn_v_matmul"),
                                         ip=ip).output
            # Reshape to 5D: [1, num_heads, num_blocks, chunk_size, head_dim]
            attn_out_5d = top.ReshapeOp(T([1, num_heads, num_blocks, chunk_size, head_dim]),
                                        attn_out_flat,
                                        loc=L(f"{attn_path}.attn_out_5d"),
                                        ip=ip).output

            # ===== Reshape back to flat sequence =====
            # [1, num_heads, num_blocks, chunk_size, head_dim] → [1, num_blocks, chunk_size, num_heads, head_dim]
            # → [1, padded_seq_len, num_heads, head_dim] → [1, padded_seq_len, hidden_size]
            attn_out_perm = top.PermuteOp(T([1, num_blocks, chunk_size, num_heads, head_dim]),
                                          attn_out_5d,
                                          order=[0, 2, 3, 1, 4],
                                          loc=L(f"{attn_path}.out_perm"),
                                          ip=ip).output
            attn_out_padded = top.ReshapeOp(T([1, padded_seq_len, hidden_size]),
                                            attn_out_perm,
                                            loc=L(f"{attn_path}.out_reshape"),
                                            ip=ip).output
            # Slice to remove padding → [1, audio_seq_len, hidden_size]
            new_op = top.SliceOp(T(hidden_shape),
                                 attn_out_padded,
                                 audio_mlir.none_op,
                                 audio_mlir.none_op,
                                 audio_mlir.none_op,
                                 offset=[0, 0, 0],
                                 steps=[1, 1, 1],
                                 ends=[1, audio_seq_len, hidden_size],
                                 axes=[],
                                 loc=L(f"{attn_path}.out_slice"),
                                 ip=ip).output

            # post_proj (ClippableLinear)
            new_op = self._clippable_linear(audio_mlir, f"{attn_path}.post", new_op,
                                            [hidden_size, hidden_size], hidden_shape)
            return new_op

        for idx in range(num_layers):
            layer_path = f"{tower_path}.layers.{idx}"

            # 1. feed_forward1
            new_op = audio_feed_forward(1, new_op, layer_path)

            # 2. norm_pre_attn → attention → norm_post_attn → residual add
            residual_ff1 = new_op
            # Skip clamp (gradient_clipping=1e10)
            new_op = self.rms_norm(audio_mlir, new_op, f"{layer_path}.norm_pre_attn")
            new_op = audio_attention(new_op, layer_path)
            # Skip clamp again
            new_op = self.rms_norm(audio_mlir, new_op, f"{layer_path}.norm_post_attn")
            new_op = top.AddOp(T(hidden_shape), [residual_ff1, new_op],
                               loc=L(f"{layer_path}.attn_residual_add"),
                               ip=ip).output

            # 3. LightConv1d
            new_op = audio_light_conv1d(new_op, layer_path)

            # 4. feed_forward2
            new_op = audio_feed_forward(2, new_op, layer_path)

            # 5. norm_out (skip clamp since gradient_clipping=1e10)
            new_op = self.rms_norm(audio_mlir, new_op, f"{layer_path}.norm_out")

        # ====== output_proj ======
        # nn.Linear(hidden_size → output_proj_dims, bias=True)
        output_shape = [1, audio_seq_len, output_proj_dims]
        new_op = self.linear(audio_mlir, f"{tower_path}.output_proj", new_op,
                             [hidden_size, output_proj_dims], output_shape)

        # ====== embed_audio ======
        # RMSNorm(no_scale) → Linear(output_proj_dims → text_hidden_size)
        embed_path = "model.embed_audio"
        new_op = self._rms_norm_no_scale(audio_mlir, new_op,
                                         f"{embed_path}.embedding_pre_projection_norm")
        new_op = self.linear(audio_mlir, f"{embed_path}.embedding_projection", new_op,
                             [output_proj_dims, self.hidden_size], out_shape)

        audio_mlir.create_return_op([new_op])
        mlir_txt = audio_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)

    def _clippable_linear(self, mlir_gen, key_prefix, in_op, weight_shape, out_shape):
        """ClippableLinear: input_clamp → MatMul → output_clamp.
        Reads clip buffers from weight file; if -inf/+inf, skip clamping."""
        ip = mlir_gen.insert_point
        weight_key = key_prefix + ".linear.weight"
        weight_op = mlir_gen.create_weight_op(weight_key, weight_shape)

        # Check if we need input clamping
        needs_input_clamp = False
        needs_output_clamp = False
        input_min_key = key_prefix + ".input_min"
        input_max_key = key_prefix + ".input_max"
        output_min_key = key_prefix + ".output_min"
        output_max_key = key_prefix + ".output_max"

        # Read clip values from model
        input_min_val = None
        input_max_val = None
        output_min_val = None
        output_max_val = None

        if self.model.is_exist(input_min_key):
            input_min_val = float(self.model.read(input_min_key).item())
        if self.model.is_exist(input_max_key):
            input_max_val = float(self.model.read(input_max_key).item())
        if self.model.is_exist(output_min_key):
            output_min_val = float(self.model.read(output_min_key).item())
        if self.model.is_exist(output_max_key):
            output_max_val = float(self.model.read(output_max_key).item())

        if input_min_val is not None and input_min_val != float('-inf'):
            needs_input_clamp = True
        if output_min_val is not None and output_min_val != float('-inf'):
            needs_output_clamp = True

        def T(shape):
            return mlir_gen.get_tensor_type(shape)

        def L(name):
            return self.get_loc(name, mlir_gen)

        # Input clamping: ClipOp(min, max)
        if needs_input_clamp:
            in_op = top.ClipOp(T(list(in_op.type.shape)),
                               in_op,
                               min=input_min_val,
                               max=input_max_val,
                               loc=L(key_prefix + ".input_clamp"),
                               ip=ip).output

        # MatMul with no bias (ClippableLinear has no bias)
        # Determine if bias exists
        bias_key = key_prefix + ".linear.bias"
        bias_op = mlir_gen.none_op
        if self.model.is_exist(bias_key):
            bias_shape = [1] * (len(out_shape) - 1) + [out_shape[-1]]
            bias_op = mlir_gen.create_weight_op(bias_key, bias_shape)

        new_op = top.MatMulOp(T(out_shape), in_op, weight_op, bias_op, loc=L(key_prefix),
                              ip=ip).output

        # Output clamping
        if needs_output_clamp:
            new_op = top.ClipOp(T(out_shape),
                                new_op,
                                min=output_min_val,
                                max=output_max_val,
                                loc=L(key_prefix + ".output_clamp"),
                                ip=ip).output

        return new_op

    def _set_clippable_linear_weight(self, key_prefix, wd):
        """Read ClippableLinear weight (with transpose) and its clip buffers."""
        self.set_linear_weight(key_prefix + ".linear", wd)
        for suffix in ["input_min", "input_max", "output_min", "output_max"]:
            buf_key = f"{key_prefix}.{suffix}"
            if self.model.is_exist(buf_key):
                wd[buf_key] = self.model.read(buf_key)

    def compile_vit(self):
        if not self.do_vit:
            return
        for name in ["vit_image", "vit_video"]:
            model_path = f"{name}/{name}.bmodel"
            self.all_bmodels.append(model_path)
            if os.path.exists(model_path):
                print(f"{model_path} already exists. Skipping compilation.")
                continue
            deploy_args = [
                f'pushd {name} && ',
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--chip {self.chip}',
                f'--num_core {self.num_core}',
                f'--num_device {self.num_device}',
                f'--model {name}.bmodel',
                '--addr_mode basic',
            ]
            if self.half_precision_quantize == 'bf16' and self.vit_f16_out_bf16:
                deploy_args.append('--quantize f16')
                deploy_args.append('--quant_output_bf16')
            else:
                deploy_args.append(f'--quantize {self.half_precision_quantize}')
                deploy_args.append('--quant_output')
            if self.high_precision:
                deploy_args.append('--high_precision')
            if self.debug:
                deploy_args.append('--debug')
            deploy_args.append('&& popd')
            self.add_task(deploy_args, f"{name}.log")

    def compile_audio(self):
        name = "audio"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}', '--addr_mode basic',
            f'--model {name}.bmodel'
        ]
        deploy_args.append(f'--quantize {self.half_precision_quantize}')
        deploy_args.append('--quant_output')
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")
