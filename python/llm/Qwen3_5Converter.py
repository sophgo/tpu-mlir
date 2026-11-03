# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import math
from .LlmConverter import *
from typing_extensions import override
from .LlmInfo import *


class Qwen3_5Converter(LlmConverter):

    def __init__(self, args, config, loader=None):
        super().__init__(args, config, loader=loader)
        self.max_pixels = args.max_pixels
        self.do_vit = not getattr(args, 'llm_only', False)
        self.dynamic = True  # force dynamic
        self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM
        if self.do_vit:
            # vision config (sets self.pixel_multiple among others)
            self.init_vconfig()
            pm = self.pixel_multiple
            if self.max_pixels == 0 or self.max_pixels % (pm * pm) != 0:
                raise RuntimeError(
                    f"max_pixels values must be multiples of {pm}*{pm} and non-zero: {args.max_pixels}"
                )
        else:
            # llm_only: still need MRoPE position_ids shape [3, seq]
            self.mrope_section, _ = self._get_mrope_params()
            self.position_shape = [1, 3, self.max_input_length
                                   ] if self.use_insert else [3, self.max_input_length]

        # extern compiles
        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}

    def init_vconfig(self):
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.temporal_patch_size = self.vconfig.temporal_patch_size
        self.spatial_merge_size = self.vconfig.spatial_merge_size
        self.pixel_multiple = self.patch_size * self.spatial_merge_size
        self.in_channels = self.vconfig.in_channels
        self.depth = self.vconfig.depth
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        self.embed_dim = self.vconfig.hidden_size
        self.vnum_heads = self.vconfig.num_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [1, 3, self.max_input_length
                               ] if self.use_insert else [3, self.max_input_length]
        self.num_position_embeddings = self.vconfig.num_position_embeddings
        self.mrope_section = getattr(self.llm_config.rope_parameters, 'mrope_section', [11, 11, 10])
        self.vit_path = "model.visual"

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = QWEN3VL_INFO

    @override
    def rotary_embedding(self):
        from .transformers_compat import text_rotary_cos_sin
        cos, sin = text_rotary_cos_sin(self.llm_config, self.seq_length)
        end_dim = cos.shape[-1] // 2
        # half
        cos = cos[:, :, :end_dim]
        sin = sin[:, :, :end_dim]
        return cos, sin  #[seq, 1, 64]

    def _get_mrope_params(self):
        rope_params = getattr(self.llm_config, 'rope_parameters', None)
        if rope_params is None:
            mrope_section = [11, 11, 10]
            rope_theta = getattr(self.llm_config, 'rope_theta', 10000000.0)
        elif isinstance(rope_params, dict):
            mrope_section = rope_params.get('mrope_section', [11, 11, 10])
            rope_theta = rope_params.get('rope_theta',
                                         getattr(self.llm_config, 'rope_theta', 10000000.0))
        else:
            mrope_section = getattr(rope_params, 'mrope_section', [11, 11, 10])
            rope_theta = getattr(rope_params, 'rope_theta',
                                 getattr(self.llm_config, 'rope_theta', 10000000.0))
        return list(mrope_section), rope_theta

    def _compute_mrope_manually(self):
        mrope_section, rope_theta = self._get_mrope_params()
        self.mrope_section = mrope_section
        total_dim = sum(mrope_section)

        inv_freq = 1.0 / (rope_theta**(np.arange(0, total_dim, 2, dtype=np.float32) / total_dim))

        pos = np.arange(self.seq_length, dtype=np.float32)
        pos_3d = pos[None, :].repeat(3, axis=0)

        freqs_3d = np.zeros((3, self.seq_length, total_dim), dtype=np.float32)
        offset = 0
        for dim_idx, section_len in enumerate(mrope_section):
            if section_len == 0:
                continue
            inv_begin = offset // 2
            inv_end = (offset + section_len + 1) // 2
            section_inv_freq = inv_freq[inv_begin:inv_end]
            section_freqs = np.outer(pos_3d[dim_idx], section_inv_freq)
            section_freqs = np.repeat(section_freqs, 2, axis=1)[:, :section_len]
            freqs_3d[dim_idx, :, offset:offset + section_len] = section_freqs
            offset += section_len

        freqs_t = self.apply_interleaved_mrope(freqs_3d)

        cos = np.cos(freqs_t).astype(np.float32).reshape(self.seq_length, 1, -1)
        sin = np.sin(freqs_t).astype(np.float32).reshape(self.seq_length, 1, -1)
        return cos, sin

    def apply_interleaved_mrope(self, freqs):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def get_mrope_index(self):
        mrope_dim = sum(self.mrope_section)
        freqs = np.arange(0, 3 * mrope_dim, dtype=np.int32).reshape(3, 1, mrope_dim)
        freqs_t = self.apply_interleaved_mrope(freqs)  # [1, 32]
        freqs_t = np.tile(freqs_t, (1, 2))  # [1, 64]
        return freqs_t.astype(np.float32)

    def mrope(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        cos_dim = self.cos.shape[-1]
        mrope_dim = sum(self.mrope_section) * 2
        weight_op = mlir_gen.create_weight_op(name + ".weight", [self.seq_length, 1, cos_dim])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, cos_dim]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, cos_dim, 1, dim]),
                               in_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * cos_dim, 1, dim]),
                               new_op,
                               shape=[3 * cos_dim, 1, -1],
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, mrope_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, mrope_dim, 1, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, mrope_dim]),
                               new_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        # cos & sin MROPE
        q_shape = list(q_op.type.shape)
        k_shape = list(k_op.type.shape)
        seqlen = q_shape[1]
        cos_op = self.mrope(mlir_gen, pos_op, rotary_cos)
        sin_op = self.mrope(mlir_gen, pos_op, rotary_sin)
        mrope_dim = sum(self.mrope_section) * 2
        pass_dim = q_shape[-1] - mrope_dim
        # q rotary
        q_rot_shape = [1, seqlen, self.num_attention_heads, mrope_dim]
        q_pass_shape = [1, seqlen, self.num_attention_heads, pass_dim]
        q_rot = top.SliceOp(mlir_gen.get_tensor_type(q_rot_shape),
                            q_op,
                            mlir_gen.none_op,
                            mlir_gen.none_op,
                            mlir_gen.none_op,
                            offset=[0, 0, 0, 0],
                            steps=[1, 1, 1, 1],
                            ends=q_rot_shape,
                            loc=self.get_loc("q_proj.slice", mlir_gen),
                            ip=mlir_gen.insert_point).output
        q_pass = top.SliceOp(mlir_gen.get_tensor_type(q_pass_shape),
                             q_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             offset=[0, 0, 0, mrope_dim],
                             steps=[1, 1, 1, 1],
                             ends=q_shape,
                             loc=self.get_loc("q_proj.slice2", mlir_gen),
                             ip=mlir_gen.insert_point).output
        q_embed = top.RopeOp(mlir_gen.get_tensor_type(q_rot_shape),
                             q_rot,
                             sin_op,
                             cos_op,
                             rope_mode=StringAttr.get("contiguous_halves"),
                             loc=self.get_loc("q_proj.rope", mlir_gen),
                             ip=mlir_gen.insert_point).output
        q_op = top.ConcatOp(mlir_gen.get_tensor_type(q_shape), [q_embed, q_pass],
                            axis=3,
                            loc=self.get_loc("q_proj", mlir_gen),
                            ip=mlir_gen.insert_point).output
        # k rotary
        k_rot_shape = [1, seqlen, self.num_key_value_heads, mrope_dim]
        k_pass_shape = [1, seqlen, self.num_key_value_heads, pass_dim]
        k_rot = top.SliceOp(mlir_gen.get_tensor_type(k_rot_shape),
                            k_op,
                            mlir_gen.none_op,
                            mlir_gen.none_op,
                            mlir_gen.none_op,
                            offset=[0, 0, 0, 0],
                            steps=[1, 1, 1, 1],
                            ends=k_rot_shape,
                            loc=self.get_loc("k_proj.slice", mlir_gen),
                            ip=mlir_gen.insert_point).output
        k_pass = top.SliceOp(mlir_gen.get_tensor_type(k_pass_shape),
                             k_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             offset=[0, 0, 0, mrope_dim],
                             steps=[1, 1, 1, 1],
                             ends=k_shape,
                             loc=self.get_loc("k_proj.slice2", mlir_gen),
                             ip=mlir_gen.insert_point).output
        k_embed = top.RopeOp(mlir_gen.get_tensor_type(k_rot_shape),
                             k_rot,
                             sin_op,
                             cos_op,
                             rope_mode=StringAttr.get("contiguous_halves"),
                             loc=self.get_loc("k_proj.rope", mlir_gen),
                             ip=mlir_gen.insert_point).output
        k_op = top.ConcatOp(mlir_gen.get_tensor_type(k_shape), [k_embed, k_pass],
                            axis=3,
                            loc=self.get_loc("k_cache", mlir_gen),
                            ip=mlir_gen.insert_point).output
        return q_op, k_op

    def vision_rotary(self):
        from .transformers_compat import vision_rotary_cos_sin
        head_dim = self.vconfig.hidden_size // self.vnum_heads
        return vision_rotary_cos_sin(head_dim // 2, self.num_patches)

    def gen_vit_mlir(self):
        tqdm.write(f"generate vit  mlir ...")
        name = f"vit"
        patches = self.num_patches
        # create weights file
        vit_npz = f"vit_top_weights.npz"
        patch_embed = f"{self.vit_path}.patch_embed.proj"
        pos_embed = f"{self.vit_path}.pos_embed"
        rotary_cos = f"{self.vit_path}.rotary.cos"
        rotary_sin = f"{self.vit_path}.rotary.sin"
        merger_norm = f"{self.vit_path}.merger.norm"
        linear_fc1 = f"{self.vit_path}.merger.linear_fc1"
        linear_fc2 = f"{self.vit_path}.merger.linear_fc2"

        def save_weights():
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + ".weight": cos,
                rotary_sin + ".weight": sin,
            }
            data = self.model.read(patch_embed + ".weight").reshape(self.embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            weights_dict[patch_embed + ".weight"] = data
            weights_dict[patch_embed + ".bias"] = self.model.read(patch_embed + ".bias")
            self.set_common_weight(pos_embed, weights_dict)  # fast_pos_embed_interpolate
            # merger
            self.set_common_weight(merger_norm, weights_dict)
            use_float_vit = isinstance(self.loader, GGUFModelHandle)
            self.vit_gguf_float = use_float_vit
            if use_float_vit:

                def _load_vit_float_linear(path, weights_dict, with_bias=True):
                    data = self.model.read(path + ".weight")
                    data = np.ascontiguousarray(np.transpose(data, (1, 0)))
                    weights_dict[path + ".weight"] = data
                    if with_bias:
                        bias = self.model.read(path + ".bias")
                        weights_dict[path + ".bias"] = bias

                _load_vit_float_linear(linear_fc1, weights_dict)
                _load_vit_float_linear(linear_fc2, weights_dict)
            else:
                self.set_linear_weight(linear_fc1, weights_dict)
                self.set_linear_weight(linear_fc2, weights_dict)
            for i in range(self.depth):
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm2", weights_dict)
                if use_float_vit:
                    _load_vit_float_linear(f"{self.vit_path}.blocks.{i}.attn.proj",
                                           weights_dict,
                                           with_bias=True)
                    _load_vit_float_linear(f"{self.vit_path}.blocks.{i}.mlp.linear_fc1",
                                           weights_dict)
                    _load_vit_float_linear(f"{self.vit_path}.blocks.{i}.mlp.linear_fc2",
                                           weights_dict)
                else:
                    self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.proj", weights_dict)
                    self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc1",
                                           weights_dict)
                    self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc2",
                                           weights_dict)

                # split qkv
                # self.set_linear_weight(f"visual.blocks.{i}.attn.qkv", weights_dict)
                weight = self.model.read(f"{self.vit_path}.blocks.{i}.attn.qkv.weight").reshape(
                    3 * self.embed_dim, self.embed_dim)
                bias = self.model.read(f"{self.vit_path}.blocks.{i}.attn.qkv.bias").reshape(
                    3 * self.embed_dim)
                q_w = weight[:self.embed_dim, :]
                k_w = weight[self.embed_dim:2 * self.embed_dim, :]
                v_w = weight[2 * self.embed_dim:, :]
                q_b = bias[:self.embed_dim]
                k_b = bias[self.embed_dim:2 * self.embed_dim]
                v_b = bias[2 * self.embed_dim:]
                q_w = np.ascontiguousarray(np.transpose(q_w, (1, 0)))
                k_w = np.ascontiguousarray(np.transpose(k_w, (1, 0)))
                v_w = np.ascontiguousarray(np.transpose(v_w, (1, 0)))
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.q.weight"] = q_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.k.weight"] = k_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.v.weight"] = v_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.q.bias"] = q_b
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.k.bias"] = k_b
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.v.bias"] = v_b
            # save weights
            np.savez(vit_npz, **weights_dict)

        save_weights()

        # create mlir file
        in_shape = [patches, self.patch_dim]
        position_shape = [patches, 2]
        out_dim = patches // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.hidden_size]
        input_shapes = [in_shape, position_shape, [patches, 4], [patches, 4, 1]]
        input_types = ['F32', 'INT32', 'INT32', 'F32']
        out_num = 1

        vit_mlir = MLIRImporter(
            input_shapes,
            [out_shape] * out_num,
            "vit",  # all vit use the same name
            self.platform,
            input_types,
            weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point

        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        def vision_block(id: int, in_op, cos_op, sin_op, mask_op):
            norm1 = f"{self.vit_path}.blocks.{id}.norm1"
            attn_q = f"{self.vit_path}.blocks.{id}.attn.q"
            attn_k = f"{self.vit_path}.blocks.{id}.attn.k"
            attn_v = f"{self.vit_path}.blocks.{id}.attn.v"
            attn_proj = f"{self.vit_path}.blocks.{id}.attn.proj"
            norm2 = f"{self.vit_path}.blocks.{id}.norm2"

            def vision_attention(in_op):
                norm1_op = self.layer_norm(vit_mlir, in_op, norm1, eps=1e-6)
                hidden_shape = [patches, self.embed_dim]
                q_op = self.linear(vit_mlir,
                                   attn_q,
                                   norm1_op, [self.embed_dim, self.embed_dim],
                                   hidden_shape,
                                   force_bias=True)
                k_op = self.linear(vit_mlir,
                                   attn_k,
                                   norm1_op, [self.embed_dim, self.embed_dim],
                                   hidden_shape,
                                   force_bias=True)
                v_op = self.linear(vit_mlir,
                                   attn_v,
                                   norm1_op, [self.embed_dim, self.embed_dim],
                                   hidden_shape,
                                   force_bias=True)
                qk_shape = [1, patches, self.vnum_heads, self.vhead_dim]
                qk_reshape = [1, -1, self.vnum_heads, self.vhead_dim]
                q_op = top.ReshapeOp(T(qk_shape),
                                     q_op,
                                     shape=qk_reshape,
                                     loc=L(attn_q + ".reshape"),
                                     ip=ip).output
                k_op = top.ReshapeOp(T(qk_shape),
                                     k_op,
                                     shape=qk_reshape,
                                     loc=L(attn_k + ".reshape"),
                                     ip=ip).output
                v_op = top.ReshapeOp(T(qk_shape),
                                     v_op,
                                     shape=qk_reshape,
                                     loc=L(attn_v + ".reshape"),
                                     ip=ip).output
                q_op = top.RopeOp(T(qk_shape),
                                  q_op,
                                  sin_op,
                                  cos_op,
                                  force_f32=True,
                                  rope_mode=StringAttr.get("contiguous_halves"),
                                  loc=L(attn_q + ".rotary"),
                                  ip=ip).output
                k_op = top.RopeOp(T(qk_shape),
                                  k_op,
                                  sin_op,
                                  cos_op,
                                  force_f32=True,
                                  rope_mode=StringAttr.get("contiguous_halves"),
                                  loc=L(attn_k + ".rotary"),
                                  ip=ip).output
                fa_op = top.FAttentionOp(T(qk_shape),
                                         q_op,
                                         k_op,
                                         v_op,
                                         mask_op,
                                         vit_mlir.none_op,
                                         scale=self.vhead_dim**-0.5,
                                         batch=1,
                                         q_head=self.vnum_heads,
                                         kv_head=self.vnum_heads,
                                         dim=self.vhead_dim,
                                         mq=patches,
                                         mk=patches,
                                         keep_dims=True,
                                         loc=L(f"{self.vit_path}.blocks.{id}.fattention"),
                                         ip=ip).output
                fa_op = top.ReshapeOp(T(hidden_shape),
                                      fa_op,
                                      shape=[-1, self.embed_dim],
                                      loc=L(f"{self.vit_path}.blocks.{id}.fattention.reshape"),
                                      ip=ip).output
                out_op = self.linear(vit_mlir,
                                     attn_proj,
                                     fa_op, [self.embed_dim, self.embed_dim],
                                     [patches, self.embed_dim],
                                     force_bias=True)
                out_op = top.AddOp(T(hidden_shape), [in_op, out_op],
                                   loc=L(attn_proj + ".add"),
                                   ip=ip).output
                return out_op

            def vision_mlp(in_op):
                in_shape = [patches, self.embed_dim]
                linear_fc1 = f"{self.vit_path}.blocks.{id}.mlp.linear_fc1"
                linear_fc2 = f"{self.vit_path}.blocks.{id}.mlp.linear_fc2"

                new_op = self.layer_norm(vit_mlir, in_op, norm2, eps=1e-6)

                fc1_op = self.linear(vit_mlir, linear_fc1, new_op,
                                     [self.embed_dim, self.vintermediate_size],
                                     [patches, self.vintermediate_size])
                act_op = self.activate(vit_mlir, fc1_op, self.vconfig.hidden_act, linear_fc1)
                fc2_op = self.linear(vit_mlir, linear_fc2, act_op,
                                     [self.vintermediate_size, self.embed_dim],
                                     [patches, self.embed_dim])
                new_op = top.AddOp(T(in_shape), [in_op, fc2_op], loc=L(linear_fc2 + ".add"),
                                   ip=ip).output
                return new_op

            in_op = vision_attention(in_op)
            in_op = vision_mlp(in_op)
            return in_op

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        in1_op = vit_mlir.create_input_op(L('position_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('pos_idx'),
                                          2)  # by fast_pos_embed_interpolate, need to be reordered
        in3_op = vit_mlir.create_input_op(L('pos_weight'),
                                          3)  # by fast_pos_embed_interpolate, need to be reordered
        in4_op = vit_mlir.none_op
        new_weight = vit_mlir.create_weight_op(patch_embed + ".weight",
                                               [self.patch_dim, self.embed_dim])
        new_bias = vit_mlir.create_weight_op(patch_embed + ".bias", [1, self.embed_dim])
        new_op = top.MatMulOp(T([patches, self.embed_dim]),
                              in0_op,
                              new_weight,
                              new_bias,
                              loc=L(patch_embed),
                              ip=ip).output
        # fast_pos_embed_interpolate
        new_weight = vit_mlir.create_weight_op(pos_embed + ".weight",
                                               [self.num_position_embeddings, self.embed_dim])
        pos_idx_gather = top.GatherOp(T([patches, 4, self.embed_dim]),
                                      new_weight,
                                      in2_op,
                                      axis=0,
                                      loc=L(pos_embed),
                                      ip=ip).output
        pos_mul = top.MulOp(T([patches, 4, self.embed_dim]), [pos_idx_gather, in3_op],
                            loc=L(pos_embed + ".mul"),
                            ip=ip).output
        pos_sum = top.ReduceOp(T([patches, self.embed_dim]),
                               pos_mul,
                               axes=[1],
                               keepdims=0,
                               mode=StringAttr.get("ReduceSum"),
                               loc=L(pos_embed + ".sum"),
                               ip=ip).output
        new_op = top.AddOp(T([patches, self.embed_dim]), [new_op, pos_sum],
                           loc=L(pos_embed + ".add"),
                           ip=ip).output
        # rotary embedding
        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight",
                                               [self.num_patches, self.vhead_dim // 4])
        cos_op = top.GatherOp(T([patches, 2, self.vhead_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_cos),
                              ip=ip).output
        cos_op = top.ReshapeOp(T([1, patches, 1, self.vhead_dim // 2]),
                               cos_op,
                               shape=[1, -1, 1, self.vhead_dim // 2],
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        cos_op = top.TileOp(T([1, patches, 1, self.vhead_dim]),
                            cos_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_cos + ".tile"),
                            ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight",
                                               [self.num_patches, self.vhead_dim // 4])
        sin_op = top.GatherOp(T([patches, 2, self.vhead_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_sin),
                              ip=ip).output
        sin_op = top.ReshapeOp(T([1, patches, 1, self.vhead_dim // 2]),
                               sin_op,
                               shape=[1, -1, 1, self.vhead_dim // 2],
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        sin_op = top.TileOp(T([1, patches, 1, self.vhead_dim]),
                            sin_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_sin + ".tile"),
                            ip=ip).output
        for id in range(self.depth):
            new_op = vision_block(id, new_op, cos_op, sin_op, in4_op)

        def patch_merger(in_op, fc1_path: str, fc2_path: str, norm_path: str, shuffle: bool):
            out_dim = self.embed_dim * (self.spatial_merge_size**2)
            in_dim = patches // (self.spatial_merge_size**2)
            if shuffle:
                in_op = top.ReshapeOp(T([in_dim, out_dim]),
                                      in_op,
                                      shape=[-1, out_dim],
                                      loc=L(fc1_path + ".preshape"),
                                      ip=ip).output
            new_op = self.layer_norm(vit_mlir, in_op, norm_path, eps=1e-6)
            if not shuffle:
                new_op = top.ReshapeOp(T([in_dim, out_dim]),
                                       new_op,
                                       shape=[-1, out_dim],
                                       loc=L(fc1_path + ".preshape"),
                                       ip=ip).output
            new_op = self.linear(vit_mlir, fc1_path, new_op, [out_dim, out_dim], [in_dim, out_dim])
            new_op = self.activate(vit_mlir, new_op, ActType.GELU, fc1_path)
            new_op = self.linear(vit_mlir, fc2_path, new_op, [out_dim, self.hidden_size],
                                 [in_dim, self.hidden_size])
            return new_op

        ret_ops = []
        new_op = patch_merger(new_op, linear_fc1, linear_fc2, merger_norm, False)
        ret_ops.append(new_op)

        vit_mlir.create_return_op(ret_ops)
        self.save_mlir_module(vit_mlir, name)

    # ------------------------------------------------------------------
    # Full-attention layer building blocks. Shared by the trunk full-attention
    # layers (prefill / decode / MTP verification nets) and the MTP draft head
    # (`mtp.layers.0.`), which is structurally identical to a trunk
    # full-attention layer (gated q, q/k norm, partial MRoPE, dense MLP).
    # ------------------------------------------------------------------
    def _fa_paths(self, prefix: str) -> dict:
        w = self.model_info.weights
        paths = {
            "prefix": prefix,
            "input_ln": prefix + w[LlmList.INPUT_LN],
            "q_proj": prefix + w[LlmList.Q_PROJ],
            "q_norm": prefix + w[LlmList.Q_NORM],
            "k_proj": prefix + w[LlmList.K_PROJ],
            "k_norm": prefix + w[LlmList.K_NORM],
            "v_proj": prefix + w[LlmList.V_PROJ],
            "o_proj": prefix + w[LlmList.O_PROJ],
            "post_attn_ln": prefix + w[LlmList.POST_ATTN_LN],
            "mlp_gate": prefix + w[LlmList.MLP_GATE],
            "mlp_up": prefix + w[LlmList.MLP_UP],
            "mlp_down": prefix + w[LlmList.MLP_DOWN],
        }
        if self.llm_type in [LlmType.QWEN3_5_MOE]:
            for key, name in [("shared_gate", LlmList.SHARED_GATE),
                              ("shared_expert_gate", LlmList.SHARED_EXPERT_GATE),
                              ("shared_expert_up", LlmList.SHARED_EXPERT_UP),
                              ("shared_expert_down", LlmList.SHARED_EXPERT_DOWN),
                              ("gate", LlmList.GATE), ("experts_gate", LlmList.EXPERTS_GATE),
                              ("experts_up", LlmList.EXPERTS_UP),
                              ("experts_down", LlmList.EXPERTS_DOWN)]:
                paths[key] = prefix + w[name]
        return paths

    def _fa_is_moe(self, idx) -> bool:
        # idx is None for the MTP draft layer, which is dense unless the
        # checkpoint stores MoE weights for it (e.g. Qwen3.5 MoE MTP head).
        return (self.llm_type in [LlmType.QWEN3_5_MOE] and idx is not None
                and (idx not in self.mlp_only_layers) and (self.num_experts > 0)
                and ((idx + 1) % self.decoder_sparse_step == 0))

    def _fa_is_moe_mtp(self) -> bool:
        # Qwen3.5 MoE checkpoints give the MTP draft layer the same MoE MLP
        # as the trunk; detect it from the safetensors index.
        if self.llm_type not in [LlmType.QWEN3_5_MOE] or self.num_experts <= 0:
            return False
        idx_file = os.path.join(self.model_path, "model.safetensors.index.json")
        if not os.path.exists(idx_file):
            return False
        try:
            with open(idx_file) as f:
                wm = json.load(f).get("weight_map", {})
        except Exception:
            return False
        w = self.model_info.weights
        mtp_prefix = w.get(LlmList.MTP_LAYERS, "mtp.layers")
        gate_suffix = w.get(LlmList.GATE, "mlp.gate")
        return (mtp_prefix + ".0." + gate_suffix + ".weight") in wm

    def _fa_save_weights(self, paths: dict, weight_dict: dict, idx):
        self.set_common_weight(paths["input_ln"], weight_dict, self.rmsnorm_type)
        # q_proj split if not do lora
        self.set_linear_weight(paths["q_proj"], weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(paths["k_proj"], weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(paths["v_proj"], weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(paths["o_proj"], weight_dict, do_lora=self.do_lora)
        self.set_common_weight(paths["q_norm"], weight_dict, self.rmsnorm_type)
        self.set_common_weight(paths["k_norm"], weight_dict, self.rmsnorm_type)
        self.set_common_weight(paths["post_attn_ln"], weight_dict, self.rmsnorm_type)
        if idx is not None:
            is_moe = self._fa_is_moe(idx)
        else:
            is_moe = self._fa_is_moe_mtp()
        if is_moe:
            self._set_moe_expert_weights(weight_dict, paths["shared_gate"],
                                         paths["shared_expert_gate"], paths["shared_expert_up"],
                                         paths["shared_expert_down"], paths["gate"],
                                         paths["experts_gate"], paths["experts_up"],
                                         paths["experts_down"])
        else:
            self.set_linear_weight(paths["mlp_gate"], weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(paths["mlp_up"], weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(paths["mlp_down"], weight_dict, do_lora=self.do_lora)

    def _fa_mlp(self, mlir_gen, input_shape, in_op, paths: dict, idx, do_norm: bool, norm_path: str,
                is_last: bool):
        """post_attention_layernorm -> MLP (dense or MoE) -> residual add
        [-> final norm]. ``is_last`` reproduces the trunk naming: the last
        layer names its residual add after mlp_down so that "output_states" is
        reserved for the final norm."""
        ip = mlir_gen.insert_point
        batch = input_shape[0]
        len = input_shape[1]
        new_op = self.rms_norm(mlir_gen, in_op, paths["post_attn_ln"])
        mlp_gate, mlp_up, mlp_down = paths["mlp_gate"], paths["mlp_up"], paths["mlp_down"]

        if idx is not None:
            is_moe = self._fa_is_moe(idx)
        else:
            is_moe = self._fa_is_moe_mtp()
        if is_moe:
            down_op = self.moe(mlir_gen,
                               paths["shared_gate"],
                               paths["shared_expert_gate"],
                               paths["shared_expert_up"],
                               paths["shared_expert_down"],
                               paths["gate"],
                               paths["experts_gate"],
                               paths["experts_up"],
                               paths["experts_down"],
                               new_op,
                               len,
                               self.hidden_act,
                               num_split_fused_moe=1,
                               do_lora=self.do_lora)
        else:
            if not self.fused_mlp:
                gate_op = self.linear(mlir_gen,
                                      mlp_gate,
                                      new_op, [self.hidden_size, self.intermediate_size],
                                      [batch, len, self.intermediate_size],
                                      do_lora=self.do_lora)
                act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
                up_op = self.linear(mlir_gen,
                                    mlp_up,
                                    new_op, [self.hidden_size, self.intermediate_size],
                                    [batch, len, self.intermediate_size],
                                    do_lora=self.do_lora)
                new_op = top.MulOp(mlir_gen.get_tensor_type([batch, len, self.intermediate_size]),
                                   [act_op, up_op],
                                   loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                                   ip=ip).output
                down_op = self.linear(mlir_gen,
                                      mlp_down,
                                      new_op, [self.intermediate_size, self.hidden_size],
                                      input_shape,
                                      do_lora=self.do_lora)
            else:
                # TODO: support multi batch
                down_op = self.mlp(mlir_gen,
                                   mlp_gate,
                                   mlp_up,
                                   mlp_down,
                                   new_op,
                                   len,
                                   self.hidden_size,
                                   self.intermediate_size,
                                   self.hidden_act,
                                   do_lora=self.do_lora)

        last_name = "output_states"
        if self._fa_is_moe(idx):
            new_name = last_name if not is_last else f"{paths['experts_down']}.add"
        else:
            new_name = last_name if not is_last else f"{mlp_down}.add"
        new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                           loc=self.get_loc(new_name, mlir_gen),
                           ip=ip).output
        if do_norm:
            new_op = self.rms_norm(mlir_gen, new_op, norm_path, last_name)
        return new_op

    def _fa_attention(self,
                      mlir_gen,
                      paths: dict,
                      in0_op,
                      pos_op,
                      input_len: int,
                      batch: int,
                      mask_op,
                      mask_size: int,
                      kv_mode: str,
                      kv_len: int = 0,
                      hist_k=None,
                      hist_v=None):
        """input_layernorm -> gated q / k / v proj -> q/k norm -> MRoPE ->
        [history concat/insert] -> FAttention -> sigmoid gate -> o_proj ->
        residual add.

        kv_mode:
          "none"   : attend over the ``input_len`` new tokens only (prefill).
          "concat" : attend over ``kv_len`` history tokens + ``input_len`` new
                     tokens (prefill with history / decode / MTP verify).
          "insert" : batch>1 decode; the new token is inserted at kv_len-1.

        Returns (attn_residual_out, k, v) with k/v being the new tokens' K/V
        ``[batch, input_len, kv_heads, head_dim]`` (what the runtime stores).
        """
        T = mlir_gen.get_tensor_type
        L = lambda name: self.get_loc(name, mlir_gen)
        ip = mlir_gen.insert_point
        q_proj, k_proj, v_proj = paths["q_proj"], paths["k_proj"], paths["v_proj"]
        o_proj, mlp_gate = paths["o_proj"], paths["mlp_gate"]
        input_shape = [batch, input_len, self.hidden_size]
        q_dim = self.num_attention_heads * self.head_dim
        q_shape = [batch, input_len, self.num_attention_heads, self.head_dim]
        kv_shape = [batch, input_len, self.num_key_value_heads, self.head_dim]

        ln_op = self.rms_norm(mlir_gen, in0_op, paths["input_ln"])
        # q_proj
        ## For Qwen3.5, q_proj is fused with gate, so the output dim is 2x q_dim, and will be splitted later
        ## TODO: maybe we can split weight, but need to consider the case of lora
        q_gate_op = self.linear(
            mlir_gen,
            q_proj,
            ln_op,
            [self.hidden_size, q_dim * 2],  # q_proj with gate, so output dim is 2x q_dim
            [batch, input_len, q_dim * 2],
            do_lora=self.do_lora)
        q_gate_op = top.ReshapeOp(T([batch, input_len, self.num_attention_heads,
                                     2 * self.head_dim]),
                                  q_gate_op,
                                  shape=[batch, -1, self.num_attention_heads, 2 * self.head_dim],
                                  loc=L(q_proj + ".reshape"),
                                  ip=ip).output
        q_op = top.SliceOp(T(q_shape),
                           q_gate_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[0, 0, 0, 0],
                           steps=[1, 1, 1, 1],
                           ends=[batch, input_len, self.num_attention_heads, self.head_dim],
                           loc=L(q_proj + ".q_slice"),
                           ip=ip).output
        gate_op = top.SliceOp(T(q_shape),
                              q_gate_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, self.head_dim],
                              steps=[1, 1, 1, 1],
                              ends=[batch, input_len, self.num_attention_heads, 2 * self.head_dim],
                              loc=L(q_proj + ".gate_slice"),
                              ip=ip).output
        gate_op = top.SigmoidOp(T(q_shape), gate_op, loc=L(mlp_gate + ".gate_sigmoid"),
                                ip=ip).output
        # k_proj / v_proj
        k_op = self.linear(mlir_gen,
                           k_proj,
                           ln_op, [self.hidden_size, self.kv_dim], [batch, input_len, self.kv_dim],
                           do_lora=self.do_lora)
        v_op = self.linear(mlir_gen,
                           v_proj,
                           ln_op, [self.hidden_size, self.kv_dim], [batch, input_len, self.kv_dim],
                           do_lora=self.do_lora)
        k_op = top.ReshapeOp(T(kv_shape),
                             k_op,
                             shape=[batch, -1, self.num_key_value_heads, self.head_dim],
                             loc=L(k_proj + ".reshape"),
                             ip=ip).output
        v_op = top.ReshapeOp(T(kv_shape),
                             v_op,
                             shape=[batch, -1, self.num_key_value_heads, self.head_dim],
                             loc=L("v_cache"),
                             ip=ip).output
        q_op = self.rms_norm(mlir_gen, q_op, paths["q_norm"])
        k_op = self.rms_norm(mlir_gen, k_op, paths["k_norm"])
        # rotary cos/sin (MRoPE)
        q_op, k_op = self.apply_rotary_pos(mlir_gen, pos_op, q_op, k_op, "rotary_cos", "rotary_sin")
        new_k, new_v = k_op, v_op
        # ======= kv history =========
        if kv_mode == "none":
            mk = input_len
        elif kv_mode == "concat":
            mk = kv_len + input_len
            k_op = top.ConcatOp(T([batch, mk, self.num_key_value_heads, self.head_dim]),
                                [hist_k, k_op],
                                axis=1,
                                only_merge=True,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([batch, mk, self.num_key_value_heads, self.head_dim]),
                                [hist_v, v_op],
                                axis=1,
                                only_merge=True,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
        elif kv_mode == "insert":
            mk = kv_len
            k_op = top.InsertOp(T([batch, kv_len, self.num_key_value_heads, self.head_dim]),
                                hist_k,
                                rhs=k_op,
                                axis=1,
                                offset=kv_len - 1,
                                loc=L(k_proj + ".insert"),
                                ip=ip).output
            v_op = top.InsertOp(T([batch, kv_len, self.num_key_value_heads, self.head_dim]),
                                hist_v,
                                rhs=v_op,
                                axis=1,
                                offset=kv_len - 1,
                                loc=L(v_proj + ".insert"),
                                ip=ip).output
        else:
            raise ValueError(f"unknown kv_mode: {kv_mode}")
        # ======= fattention =========
        fa_op = top.FAttentionOp(T([batch, input_len, q_dim]),
                                 q_op,
                                 k_op,
                                 v_op,
                                 mask_op,
                                 mlir_gen.none_op,
                                 scale=self.head_dim**-0.5,
                                 batch=batch,
                                 q_head=self.num_attention_heads,
                                 kv_head=self.num_key_value_heads,
                                 dim=self.head_dim,
                                 mq=input_len,
                                 mk=mk,
                                 keep_dims=False,
                                 mask_size=mask_size,
                                 loc=L(paths["prefix"] + "fattention"),
                                 ip=ip).output
        gate_op = top.ReshapeOp(T([batch, input_len, q_dim]),
                                gate_op,
                                shape=[batch, -1, q_dim],
                                loc=L(mlp_gate + ".gate_reshape"),
                                ip=ip).output
        fa_op = top.MulOp(T([batch, input_len, q_dim]), [fa_op, gate_op],
                          loc=L(mlp_gate + ".gate_mul"),
                          ip=ip).output
        o_op = self.linear(mlir_gen,
                           o_proj,
                           fa_op, [q_dim, self.hidden_size],
                           input_shape,
                           do_lora=self.do_lora)
        o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
        return o_op, new_k, new_v

    def gen_block_full_attn_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} full attention mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        paths = self._fa_paths(TOP_PATH)
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.num_device < 2 and idx == self.num_layers - 1
        is_last = idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"
        # ===== save weight ===========
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.save_small_attn_mask_weight(weight_dict)
        self._fa_save_weights(paths, weight_dict, idx)
        if do_norm:
            self.set_common_weight(norm, weight_dict, self.rmsnorm_type)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weight_keys.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        # create block mlir
        def gen_block_by_length(name: str, input_len: int, with_history: bool = False):
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = input_len
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]
            max_kv_len = input_len if not with_history else self.seq_length + input_len
            mask_shape = [1, 1, input_len, max_kv_len]
            input_shapes = [input_shape, id_shape
                            ] if self.use_small_mask() else [input_shape, id_shape, mask_shape]
            input_types = ["F32", "INT32"] if self.use_small_mask() else ["F32", "INT32", "F32"]
            if with_history:
                history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
                input_shapes.append(history_shape)  # k cache
                input_shapes.append(history_shape)  # v cache
                input_types.append("F32")
                input_types.append("F32")
            block_mlir = MLIRImporter(input_shapes, [input_shape, kv_shape, kv_shape],
                                      name,
                                      self.platform,
                                      input_types,
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")
            L = lambda name: self.get_loc(name, block_mlir)

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2) if not self.use_small_mask() \
                else None
            in3_op = in4_op = None
            if with_history:
                index = 3 if not self.use_small_mask() else 2
                in3_op = block_mlir.create_input_op(L("history_k"), index)
                in4_op = block_mlir.create_input_op(L("history_v"), index + 1)
            mask_op, mask_size = self.get_fattention_mask_op(block_mlir, in2_op)
            o_op, k_op, v_op = self._fa_attention(block_mlir,
                                                  paths,
                                                  in0_op,
                                                  in1_op,
                                                  input_len,
                                                  1,
                                                  mask_op,
                                                  mask_size,
                                                  kv_mode="concat" if with_history else "none",
                                                  kv_len=self.seq_length,
                                                  hist_k=in3_op,
                                                  hist_v=in4_op)
            # ========== mlp =============
            new_op = self._fa_mlp(block_mlir, input_shape, o_op, paths, idx, do_norm, norm, is_last)
            block_mlir.create_return_op([new_op, k_op, v_op])
            self.save_mlir_module(block_mlir, name)

        def gen_block():
            gen_block_by_length(f"block_{idx}", self.max_input_length)

        def gen_block_kv():
            gen_block_by_length(f"block_kv_{idx}", self.max_input_length, with_history=True)
            return

        def gen_block_cache_by_length(kv_length: int, stage_idx: int = 0):
            name = f"block_cache_{idx}"
            input_shape = [self.batch, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            mask_len = kv_length if self.use_insert else kv_length + 1
            if self.use_insert:
                id_shape[0] = self.batch
            id_shape[-1] = 1
            mask_shape = [self.batch, 1, 1, mask_len]
            history_shape = [self.batch, kv_length, self.num_key_value_heads, self.head_dim]

            kv_shape = [self.batch, 1, self.num_key_value_heads, self.head_dim]
            output_shapes = [input_shape] if self.use_insert else [input_shape, kv_shape, kv_shape]
            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                output_shapes,
                name,
                self.platform, ["F32", "INT32", "F32", "F32", "F32"],
                lora_rank=self.lora_rank,
                weight_file=f"../{weight_file}")
            L = lambda name: self.get_loc(name, block_mlir)

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            # no use - placeholder to match weight file from prefill block
            self.create_decode_mask_placeholder(block_mlir)
            o_op, k_op, v_op = self._fa_attention(block_mlir,
                                                  paths,
                                                  in0_op,
                                                  in1_op,
                                                  1,
                                                  self.batch,
                                                  in2_op,
                                                  0,
                                                  kv_mode="insert" if self.use_insert else "concat",
                                                  kv_len=kv_length,
                                                  hist_k=in3_op,
                                                  hist_v=in4_op)
            # ========== mlp =============
            new_op = self._fa_mlp(block_mlir, input_shape, o_op, paths, idx, do_norm, norm, is_last)
            return_ops = [new_op] if self.use_insert else [new_op, k_op, v_op]
            block_mlir.create_return_op(return_ops)
            if stage_idx == 0:
                self.save_mlir_module(block_mlir, name)
            else:
                self.save_mlir_module(block_mlir, f"{name}_{stage_idx}")

        def gen_block_cache():
            # largest
            gen_block_cache_by_length(self.seq_length)
            if len(self.decode_chunk_list) > 0:
                for stage_idx, kv_length in enumerate(self.decode_chunk_list):
                    gen_block_cache_by_length(kv_length, stage_idx + 1)

        def gen_block_verify():
            # MTP verification: K = draft_len + 1 tokens decoded in one launch
            # against the full history (static shapes, explicit mask like
            # block_cache). K/V of the K new tokens are returned so the runtime
            # can store them at history_length.. (only the accepted prefix is
            # kept: later rounds overwrite the rest).
            K = self.mtp_k
            name = f"block_verify_{idx}"
            input_shape = [1, K, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = K
            mask_shape = [1, 1, K, self.seq_length + K]
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            kv_shape = [1, K, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                self.platform, ["F32", "INT32", "F32", "F32", "F32"],
                lora_rank=self.lora_rank,
                weight_file=f"../{weight_file}")
            L = lambda name: self.get_loc(name, block_mlir)

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            self.create_decode_mask_placeholder(block_mlir)
            o_op, k_op, v_op = self._fa_attention(block_mlir,
                                                  paths,
                                                  in0_op,
                                                  in1_op,
                                                  K,
                                                  1,
                                                  in2_op,
                                                  0,
                                                  kv_mode="concat",
                                                  kv_len=self.seq_length,
                                                  hist_k=in3_op,
                                                  hist_v=in4_op)
            new_op = self._fa_mlp(block_mlir, input_shape, o_op, paths, idx, do_norm, norm, is_last)
            block_mlir.create_return_op([new_op, k_op, v_op])
            self.save_mlir_module(block_mlir, name)

        gen_block()
        gen_block_cache()
        if self.use_history_kv:
            gen_block_kv()
        if self.do_mtp:
            gen_block_verify()

    def gen_block_linear_attn_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} linear attention mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        if self.llm_type in [LlmType.QWEN3_5_MOE]:
            shared_gate = TOP_PATH + self.model_info.weights[LlmList.SHARED_GATE]
            shared_expert_gate = TOP_PATH + self.model_info.weights[LlmList.SHARED_EXPERT_GATE]
            shared_expert_up = TOP_PATH + self.model_info.weights[LlmList.SHARED_EXPERT_UP]
            shared_expert_down = TOP_PATH + self.model_info.weights[LlmList.SHARED_EXPERT_DOWN]
            gate = TOP_PATH + self.model_info.weights[LlmList.GATE]
            experts_gate = TOP_PATH + self.model_info.weights[LlmList.EXPERTS_GATE]
            experts_up = TOP_PATH + self.model_info.weights[LlmList.EXPERTS_UP]
            experts_down = TOP_PATH + self.model_info.weights[LlmList.EXPERTS_DOWN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        A_log = TOP_PATH + "linear_attn.A_log"
        conv1d = TOP_PATH + "linear_attn.conv1d"
        dt_bias = TOP_PATH + "linear_attn.dt_bias"
        in_proj_a = TOP_PATH + "linear_attn.in_proj_a"
        in_proj_a_bias = TOP_PATH + "linear_attn.in_proj_a.bias"
        in_proj_b = TOP_PATH + "linear_attn.in_proj_b"
        in_proj_qkv = TOP_PATH + "linear_attn.in_proj_qkv"
        in_proj_z = TOP_PATH + "linear_attn.in_proj_z"
        linear_norm = TOP_PATH + "linear_attn.norm"
        out_proj = TOP_PATH + "linear_attn.out_proj"
        weight_file = f"block_{idx}_top_weights.npz"
        A_log_data = self.model.read(A_log)
        if not isinstance(self.loader, GGUFModelHandle):
            A_log_data = -np.exp(A_log_data)
        chunk_size = 64

        weight_dict = {A_log + ".weight": A_log_data}
        self.set_common_weight(input_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(post_attn_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(conv1d, weight_dict)
        self.set_common_weight(dt_bias, weight_dict)
        self.set_linear_weight(in_proj_a, weight_dict, do_lora=self.do_lora)
        weight_dict[in_proj_a_bias] = self.model.read(dt_bias)
        self.set_linear_weight(in_proj_b, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(in_proj_qkv, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(in_proj_z, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(linear_norm, weight_dict,
                               WeightType.RMSNORM)  # not zero centered norm
        self.set_linear_weight(out_proj, weight_dict, do_lora=self.do_lora)
        if self.llm_type in [LlmType.QWEN3_5_MOE] and (idx not in self.mlp_only_layers) and (
                self.num_experts > 0) and ((idx + 1) % self.decoder_sparse_step == 0):
            self._set_moe_expert_weights(weight_dict, shared_gate, shared_expert_gate,
                                         shared_expert_up, shared_expert_down, gate, experts_gate,
                                         experts_up, experts_down)
        else:
            self.set_linear_weight(mlp_gate, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(mlp_up, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(mlp_down, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(norm, weight_dict, self.rmsnorm_type)
        # eye
        eye_key = TOP_PATH + "linear_attn.eye"
        weight_dict[eye_key] = np.eye(chunk_size, dtype=np.float32)
        ones = np.ones((chunk_size, chunk_size), dtype=np.float32)
        # triu_mask
        triu_mask_key = TOP_PATH + "linear_attn.triu_mask"
        triu_mask = np.triu(ones, k=0)
        weight_dict[triu_mask_key] = triu_mask
        # strict triu_mask
        strict_triu_mask_key = TOP_PATH + "linear_attn.strict_triu_mask"
        strict_triu_mask = np.triu(ones, k=1)
        weight_dict[strict_triu_mask_key] = strict_triu_mask
        # tril_mask
        tril_mask_key = TOP_PATH + "linear_attn.tril_mask"
        tril_mask = np.tril(ones, k=0)
        weight_dict[tril_mask_key] = tril_mask
        np.savez(weight_file, **weight_dict)

        num_v_heads = self.llm_config.linear_num_value_heads
        num_k_heads = self.llm_config.linear_num_key_heads
        head_k_dim = self.llm_config.linear_key_head_dim
        head_v_dim = self.llm_config.linear_value_head_dim
        if head_k_dim != head_v_dim:
            raise ValueError(
                f"linear attention with different key/value head dim is not supported, but got {head_k_dim} and {head_v_dim}"
            )
        conv_kernel_size = self.llm_config.linear_conv_kernel_dim
        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = key_dim * 2 + value_dim
        scale = 1 / (head_v_dim**0.5)

        do_norm = self.num_device < 2 and idx == self.num_layers - 1

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            batch = input_shape[0]
            len = input_shape[1]
            new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            if self.llm_type in [LlmType.QWEN3_5_MOE] and (idx not in self.mlp_only_layers) and (
                    self.num_experts > 0) and ((idx + 1) % self.decoder_sparse_step == 0):
                down_op = self.moe(mlir_gen,
                                   shared_gate,
                                   shared_expert_gate,
                                   shared_expert_up,
                                   shared_expert_down,
                                   gate,
                                   experts_gate,
                                   experts_up,
                                   experts_down,
                                   new_op,
                                   len,
                                   self.hidden_act,
                                   num_split_fused_moe=1,
                                   do_lora=self.do_lora)
            else:
                if not self.fused_mlp:
                    gate_op = self.linear(mlir_gen,
                                          mlp_gate,
                                          new_op, [self.hidden_size, self.intermediate_size],
                                          [batch, len, self.intermediate_size],
                                          do_lora=self.do_lora)
                    act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
                    up_op = self.linear(mlir_gen,
                                        mlp_up,
                                        new_op, [self.hidden_size, self.intermediate_size],
                                        [batch, len, self.intermediate_size],
                                        do_lora=self.do_lora)
                    new_op = top.MulOp(mlir_gen.get_tensor_type(
                        [batch, len, self.intermediate_size]), [act_op, up_op],
                                       loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                                       ip=ip).output
                    down_op = self.linear(mlir_gen,
                                          mlp_down,
                                          new_op, [self.intermediate_size, self.hidden_size],
                                          input_shape,
                                          do_lora=self.do_lora)
                else:
                    # TODO: support multi batch
                    down_op = self.mlp(mlir_gen,
                                       mlp_gate,
                                       mlp_up,
                                       mlp_down,
                                       new_op,
                                       len,
                                       self.hidden_size,
                                       self.intermediate_size,
                                       self.hidden_act,
                                       do_lora=self.do_lora)

            last_name = "output_states"
            if self.llm_type in [LlmType.QWEN3_5_MOE] and (idx not in self.mlp_only_layers) and (
                    self.num_experts > 0) and ((idx + 1) % self.decoder_sparse_step == 0):
                new_name = last_name if idx != self.num_layers - 1 else f"{experts_down}.add"
            else:
                new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        def gen_block_by_length(name: str, input_len: int, with_history: bool = False):
            input_shape = [1, input_len, self.hidden_size]
            conv_shape = [1, conv_dim, conv_kernel_size]
            recurrent_shape = [1, num_v_heads, head_v_dim, head_v_dim]
            input_shapes = [input_shape, recurrent_shape
                            ] if not with_history else [input_shape, recurrent_shape, conv_shape]
            input_types = ["F32", "F32"] if not with_history else ["F32", "F32", "F32"]
            output_shapes = [input_shape, conv_shape]
            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      self.platform,
                                      input_types,
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("recurrent_states"), 1)
            in2_op = block_mlir.create_input_op(L("conv_states"),
                                                2) if with_history else block_mlir.none_op
            in0_norm_op = self.rms_norm(block_mlir, in0_op, input_ln)
            mixed_qkv_op = self.linear(block_mlir,
                                       in_proj_qkv,
                                       in0_norm_op, [self.hidden_size, conv_dim],
                                       [1, input_len, conv_dim],
                                       do_lora=self.do_lora)
            z_op = self.linear(block_mlir,
                               in_proj_z,
                               in0_norm_op, [self.hidden_size, value_dim],
                               [1, input_len, value_dim],
                               do_lora=self.do_lora)
            z_op = self.activate(block_mlir, z_op, ActType.SILU, in_proj_z)
            b_op = self.linear(block_mlir,
                               in_proj_b,
                               in0_norm_op, [self.hidden_size, num_v_heads],
                               [1, input_len, num_v_heads],
                               do_lora=self.do_lora)
            a_op = self.linear(block_mlir,
                               in_proj_a,
                               in0_norm_op, [self.hidden_size, num_v_heads],
                               [1, input_len, num_v_heads],
                               force_bias=True,
                               do_lora=self.do_lora)
            qkv_dim = input_len
            mixed_qkv_op = top.PermuteOp(T([1, conv_dim, qkv_dim]),
                                         mixed_qkv_op,
                                         order=[0, 2, 1],
                                         loc=L(in_proj_qkv + ".permute"),
                                         ip=ip).output
            if with_history:
                qkv_dim = conv_kernel_size + input_len
                mixed_qkv_op = top.ConcatOp(T([1, conv_dim, qkv_dim]), [in2_op, mixed_qkv_op],
                                            axis=2,
                                            loc=L("concat_conv_states"),
                                            ip=ip).output
            conv_state = top.SliceOp(T([1, conv_dim, conv_kernel_size]),
                                     mixed_qkv_op,
                                     block_mlir.none_op,
                                     block_mlir.none_op,
                                     block_mlir.none_op,
                                     offset=[0, 0, -conv_kernel_size],
                                     steps=[1, 1, 1],
                                     ends=[1, conv_dim, qkv_dim],
                                     loc=L("conv_states"),
                                     ip=ip).output
            return_ops = [conv_state]
            # conv1d to conv2d
            mixed_qkv_op = top.ReshapeOp(T([1, conv_dim, 1, qkv_dim]),
                                         mixed_qkv_op,
                                         shape=[1, conv_dim, 1, -1],
                                         loc=L(in_proj_qkv + ".reshape_to_conv2d"),
                                         ip=ip).output
            weight_op = block_mlir.create_weight_op(conv1d + ".weight",
                                                    [conv_dim, 1, 1, conv_kernel_size])
            conv_op = top.ConvOp(T([1, conv_dim, 1, qkv_dim]),
                                 mixed_qkv_op,
                                 weight_op,
                                 block_mlir.none_op,
                                 kernel_shape=[1, conv_kernel_size],
                                 strides=[1, 1],
                                 group=conv_dim,
                                 pads=[0, conv_kernel_size - 1, 0, 0],
                                 loc=L(conv1d),
                                 ip=ip).output
            if with_history:
                conv_op = top.SliceOp(T([1, conv_dim, 1, input_len]),
                                      conv_op,
                                      block_mlir.none_op,
                                      block_mlir.none_op,
                                      block_mlir.none_op,
                                      offset=[0, 0, 0, conv_kernel_size],
                                      steps=[1, 1, 1, 1],
                                      ends=[1, conv_dim, 1, qkv_dim],
                                      loc=L("conv_output_slice"),
                                      ip=ip).output
            conv_op = self.activate(block_mlir, conv_op, ActType.SILU, conv1d)
            conv_op = top.ReshapeOp(T([1, num_k_heads * 2 + num_v_heads, head_v_dim, input_len]),
                                    conv_op,
                                    shape=[1, num_k_heads * 2 + num_v_heads, head_v_dim, -1],
                                    loc=L(conv1d + ".reshape_after_conv"),
                                    ip=ip).output
            q_op = top.SliceOp(T([1, num_k_heads, head_k_dim, input_len]),
                               conv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, 0, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=[1, num_k_heads, head_k_dim, input_len],
                               loc=L(in_proj_qkv + ".q_slice_after_conv"),
                               ip=ip).output
            k_op = top.SliceOp(T([1, num_k_heads, head_k_dim, input_len]),
                               conv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, num_k_heads, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=[1, num_k_heads * 2, head_k_dim, input_len],
                               loc=L(in_proj_qkv + ".k_slice_after_conv"),
                               ip=ip).output
            v_op = top.SliceOp(T([1, num_v_heads, head_v_dim, input_len]),
                               conv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, num_k_heads * 2, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=[1, num_k_heads * 2 + num_v_heads, head_v_dim, input_len],
                               loc=L(in_proj_qkv + ".v_slice_after_conv"),
                               ip=ip).output
            q_op = top.PermuteOp(T([1, input_len, num_k_heads, head_k_dim]),
                                 q_op,
                                 order=[0, 3, 1, 2],
                                 loc=L(in_proj_qkv + ".q_permute_after_conv"),
                                 ip=ip).output
            k_op = top.PermuteOp(T([1, input_len, num_k_heads, head_k_dim]),
                                 k_op,
                                 order=[0, 3, 1, 2],
                                 loc=L(in_proj_qkv + ".k_permute_after_conv"),
                                 ip=ip).output
            v_op = top.PermuteOp(T([1, input_len, num_v_heads, head_v_dim]),
                                 v_op,
                                 order=[0, 3, 1, 2],
                                 loc=L(in_proj_qkv + ".v_permute_after_conv"),
                                 ip=ip).output
            beta_op = top.SigmoidOp(T([1, input_len, num_v_heads]),
                                    b_op,
                                    loc=L(in_proj_b + ".sigmoid"),
                                    ip=ip).output
            # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            a_op = top.SoftplusOp(T([1, input_len, num_v_heads]),
                                  a_op,
                                  loc=L(in_proj_a + ".softplus"),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(A_log + ".weight", [1, 1, num_v_heads])
            g_op = top.MulOp(T([1, input_len, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".mul"),
                             ip=ip).output

            # ================= chunk_gated_delta_rule ==================
            triu_mask_op = block_mlir.create_weight_op(triu_mask_key, [chunk_size, chunk_size])
            strict_triu_mask_op = block_mlir.create_weight_op(strict_triu_mask_key,
                                                              [chunk_size, chunk_size])
            tril_mask_op = block_mlir.create_weight_op(tril_mask_key, [chunk_size, chunk_size])
            eye_op = block_mlir.create_weight_op(eye_key, [chunk_size, chunk_size])
            core_attn_out = top.ChunkGatedDeltaRuleOp(T([1, input_len, num_v_heads, head_v_dim]),
                                                      q_op,
                                                      k_op,
                                                      v_op,
                                                      g_op,
                                                      beta_op,
                                                      in1_op,
                                                      triu_mask_op,
                                                      strict_triu_mask_op,
                                                      tril_mask_op,
                                                      eye_op,
                                                      num_k_heads=num_k_heads,
                                                      num_v_heads=num_v_heads,
                                                      d=head_v_dim,
                                                      chunk_size=chunk_size,
                                                      use_qk_l2norm=True,
                                                      scale=scale,
                                                      loc=L(TOP_PATH + "chunk_gated_delta_rule"),
                                                      ip=ip).attn_out

            # RmsNormGated
            core_attn_op = self.rms_norm(block_mlir,
                                         core_attn_out,
                                         linear_norm,
                                         eps=self.rms_norm_eps)
            # -> [1, input_len, value_dim]
            core_attn_op = top.ReshapeOp(T([1, input_len, value_dim]),
                                         core_attn_op,
                                         shape=[1, -1, value_dim],
                                         loc=L(in_proj_qkv + ".core_attn_reshape"),
                                         ip=ip).output
            core_attn_op = top.MulOp(T([1, input_len, value_dim]), [core_attn_op, z_op],
                                     loc=L(in_proj_z + ".core_attn_mul"),
                                     ip=ip).output
            o_op = self.linear(block_mlir,
                               out_proj,
                               core_attn_op, [value_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(out_proj + ".add"), ip=ip).output
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            self.save_mlir_module(block_mlir, name)

        def gen_block():
            name = f"block_{idx}"
            gen_block_by_length(name, self.max_input_length, self.use_history_kv)
            return

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [self.batch, 1, self.hidden_size]
            conv_state_shape = [self.batch, conv_dim, conv_kernel_size]
            recurrent_state_shape = [self.batch, num_v_heads, head_v_dim, head_v_dim]
            block_mlir = MLIRImporter([input_shape, conv_state_shape, recurrent_state_shape],
                                      [input_shape, conv_state_shape],
                                      name,
                                      self.platform, ["F32", "F32", "F32"],
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("conv_state"), 1)
            in2_op = block_mlir.create_input_op(L("recurrent_state"), 2)
            # eye_key no use, just the same with block
            block_mlir.create_weight_op(triu_mask_key, [chunk_size, chunk_size], placeholder="Auto")
            block_mlir.create_weight_op(strict_triu_mask_key, [chunk_size, chunk_size],
                                        placeholder="Auto")
            block_mlir.create_weight_op(tril_mask_key, [chunk_size, chunk_size], placeholder="Auto")
            block_mlir.create_weight_op(eye_key, [chunk_size, chunk_size], placeholder="Auto")
            return_ops = []
            in0_norm_op = self.rms_norm(block_mlir, in0_op, input_ln)
            # z/a/b
            z_op = self.linear(block_mlir,
                               in_proj_z,
                               in0_norm_op, [self.hidden_size, value_dim],
                               [self.batch, 1, value_dim],
                               do_lora=self.do_lora)
            a_op = self.linear(block_mlir,
                               in_proj_a,
                               in0_norm_op, [self.hidden_size, num_v_heads],
                               [self.batch, 1, num_v_heads],
                               force_bias=True,
                               do_lora=self.do_lora)
            b_op = self.linear(block_mlir,
                               in_proj_b,
                               in0_norm_op, [self.hidden_size, num_v_heads],
                               [self.batch, 1, num_v_heads],
                               do_lora=self.do_lora)
            # q_proj
            mixed_qkv_op = self.linear(block_mlir,
                                       in_proj_qkv,
                                       in0_norm_op, [self.hidden_size, conv_dim],
                                       [self.batch, 1, conv_dim],
                                       do_lora=self.do_lora)
            mixed_qkv_op = top.ReshapeOp(T([self.batch, conv_dim, 1]),
                                         mixed_qkv_op,
                                         loc=L(in_proj_qkv + ".reshape"),
                                         ip=ip).output
            # use_precomputed_states, causal_conv1d_update
            mixed_qkv_op = top.ConcatSliceOp(T([self.batch, conv_dim, conv_kernel_size]),
                                             in1_op,
                                             mixed_qkv_op,
                                             axis=2,
                                             loc=L(conv1d + ".concat"),
                                             ip=ip).output
            return_ops.append(mixed_qkv_op)  # new conv state
            weight_op = block_mlir.create_weight_op(conv1d + ".weight",
                                                    [1, conv_dim, conv_kernel_size])
            mixed_qkv_op = top.MulOp(T([self.batch, conv_dim, conv_kernel_size]),
                                     [mixed_qkv_op, weight_op],
                                     loc=L(conv1d + ".mul"),
                                     ip=ip).output
            mixed_qkv_op = top.ReduceOp(T([self.batch, conv_dim]),
                                        mixed_qkv_op,
                                        axes=[2],
                                        keepdims=False,
                                        mode=StringAttr.get("ReduceSum"),
                                        loc=L(conv1d + ".reduce_sum"),
                                        ip=ip).output

            beta_op = top.SigmoidOp(T([self.batch, 1, num_v_heads]),
                                    b_op,
                                    loc=L(in_proj_b + ".sigmoid"),
                                    ip=ip).output
            # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            a_op = top.SoftplusOp(T([self.batch, 1, num_v_heads]),
                                  a_op,
                                  loc=L(in_proj_a + ".softplus"),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(A_log + ".weight", [1, 1, num_v_heads])
            g_op = top.MulOp(T([self.batch, 1, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".mul"),
                             ip=ip).output
            mixed_qkv_op = self.activate(block_mlir, mixed_qkv_op, ActType.SILU, conv1d)
            q_op = top.SliceOp(T([1, key_dim]),
                               mixed_qkv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, 0],
                               steps=[1, 1],
                               ends=[1, key_dim],
                               loc=L(in_proj_qkv + ".q_slice"),
                               ip=ip).output
            k_op = top.SliceOp(T([1, key_dim]),
                               mixed_qkv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, key_dim],
                               steps=[1, 1],
                               ends=[1, key_dim * 2],
                               loc=L(in_proj_qkv + ".k_slice"),
                               ip=ip).output
            v_op = top.SliceOp(T([1, value_dim]),
                               mixed_qkv_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, key_dim * 2],
                               steps=[1, 1],
                               ends=[1, conv_dim],
                               loc=L(in_proj_qkv + ".v_slice"),
                               ip=ip).output
            # ========== recurreent_gated_delta_rule =================
            # new_state is NoneType: the kernel updates in2_op (recurrent state)
            # in place, which is what --same_addr relies on.
            core_attn_op = top.RecurrentGatedDeltaRuleOp(
                T([self.batch, 1, num_v_heads, head_v_dim]),
                T(None),
                q_op,
                k_op,
                v_op,
                g_op,
                beta_op,
                in2_op,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                d=head_v_dim,
                use_qk_l2norm=True,
                scale=scale,
                loc=L(TOP_PATH + "recurrent_gated_delta_rule"),
                ip=ip).attn_out
            # RmsNormGated
            core_attn_op = self.rms_norm(block_mlir,
                                         core_attn_op,
                                         linear_norm,
                                         eps=self.rms_norm_eps)
            core_attn_op = top.ReshapeOp(T([self.batch, 1, value_dim]),
                                         core_attn_op,
                                         loc=L(TOP_PATH + ".core_attn_reshape"),
                                         ip=ip).output
            z_op = self.activate(block_mlir, z_op, ActType.SILU, in_proj_z)
            o_op = top.MulOp(T([self.batch, 1, value_dim]), [core_attn_op, z_op],
                             loc=L(TOP_PATH + ".core_attn_mul"),
                             ip=ip).output
            o_op = self.linear(block_mlir,
                               out_proj,
                               o_op, [value_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(out_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            self.save_mlir_module(block_mlir, name)

        def gen_block_verify():
            # MTP verification for a linear-attention layer: K = draft_len + 1
            # tokens in one launch. Unlike block_cache the state inputs are NOT
            # updated in place; instead the conv window / recurrent state after
            # each of the K tokens is returned so the runtime can commit the one
            # matching the accepted prefix (d2d copy) and discard the rest.
            K = self.mtp_k
            name = f"block_verify_{idx}"
            input_shape = [1, K, self.hidden_size]
            conv_state_shape = [1, conv_dim, conv_kernel_size]
            recurrent_state_shape = [1, num_v_heads, head_v_dim, head_v_dim]
            conv_all_shape = [K, conv_dim, conv_kernel_size]
            recurrent_all_shape = [K, num_v_heads, head_v_dim, head_v_dim]
            block_mlir = MLIRImporter([input_shape, conv_state_shape, recurrent_state_shape],
                                      [input_shape, conv_all_shape, recurrent_all_shape],
                                      name,
                                      self.platform, ["F32", "F32", "F32"],
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)
            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("conv_state"), 1)
            in2_op = block_mlir.create_input_op(L("recurrent_state"), 2)
            # chunk masks are unused here; keep them as placeholders so the
            # weight file layout matches block / block_cache
            block_mlir.create_weight_op(triu_mask_key, [chunk_size, chunk_size], placeholder="Auto")
            block_mlir.create_weight_op(strict_triu_mask_key, [chunk_size, chunk_size],
                                        placeholder="Auto")
            block_mlir.create_weight_op(tril_mask_key, [chunk_size, chunk_size], placeholder="Auto")
            block_mlir.create_weight_op(eye_key, [chunk_size, chunk_size], placeholder="Auto")

            in0_norm_op = self.rms_norm(block_mlir, in0_op, input_ln)
            mixed_qkv_op = self.linear(block_mlir,
                                       in_proj_qkv,
                                       in0_norm_op, [self.hidden_size, conv_dim], [1, K, conv_dim],
                                       do_lora=self.do_lora)
            z_op = self.linear(block_mlir,
                               in_proj_z,
                               in0_norm_op, [self.hidden_size, value_dim], [1, K, value_dim],
                               do_lora=self.do_lora)
            z_op = self.activate(block_mlir, z_op, ActType.SILU, in_proj_z)
            b_op = self.linear(block_mlir,
                               in_proj_b,
                               in0_norm_op, [self.hidden_size, num_v_heads], [1, K, num_v_heads],
                               do_lora=self.do_lora)
            a_op = self.linear(block_mlir,
                               in_proj_a,
                               in0_norm_op, [self.hidden_size, num_v_heads], [1, K, num_v_heads],
                               force_bias=True,
                               do_lora=self.do_lora)
            # ---- causal conv over [conv_state | K new tokens] ----
            qkv_dim = conv_kernel_size + K
            mixed_qkv_op = top.PermuteOp(T([1, conv_dim, K]),
                                         mixed_qkv_op,
                                         order=[0, 2, 1],
                                         loc=L(in_proj_qkv + ".permute"),
                                         ip=ip).output
            mixed_qkv_op = top.ConcatOp(T([1, conv_dim, qkv_dim]), [in1_op, mixed_qkv_op],
                                        axis=2,
                                        loc=L("concat_conv_states"),
                                        ip=ip).output
            # conv window after each new token j: columns [j+1, j+1+kernel)
            conv_states = []
            for j in range(K):
                conv_states.append(
                    top.SliceOp(T([1, conv_dim, conv_kernel_size]),
                                mixed_qkv_op,
                                block_mlir.none_op,
                                block_mlir.none_op,
                                block_mlir.none_op,
                                offset=[0, 0, j + 1],
                                steps=[1, 1, 1],
                                ends=[1, conv_dim, j + 1 + conv_kernel_size],
                                loc=L(f"conv_states_step{j}"),
                                ip=ip).output)
            conv_states_all = top.ConcatOp(T(conv_all_shape),
                                           conv_states,
                                           axis=0,
                                           loc=L("conv_states_all"),
                                           ip=ip).output
            # conv1d to conv2d
            mixed_qkv_op = top.ReshapeOp(T([1, conv_dim, 1, qkv_dim]),
                                         mixed_qkv_op,
                                         shape=[1, conv_dim, 1, -1],
                                         loc=L(in_proj_qkv + ".reshape_to_conv2d"),
                                         ip=ip).output
            weight_op = block_mlir.create_weight_op(conv1d + ".weight",
                                                    [conv_dim, 1, 1, conv_kernel_size])
            conv_op = top.ConvOp(T([1, conv_dim, 1, qkv_dim]),
                                 mixed_qkv_op,
                                 weight_op,
                                 block_mlir.none_op,
                                 kernel_shape=[1, conv_kernel_size],
                                 strides=[1, 1],
                                 group=conv_dim,
                                 pads=[0, conv_kernel_size - 1, 0, 0],
                                 loc=L(conv1d),
                                 ip=ip).output
            conv_op = top.SliceOp(T([1, conv_dim, 1, K]),
                                  conv_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  offset=[0, 0, 0, conv_kernel_size],
                                  steps=[1, 1, 1, 1],
                                  ends=[1, conv_dim, 1, qkv_dim],
                                  loc=L("conv_output_slice"),
                                  ip=ip).output
            conv_op = self.activate(block_mlir, conv_op, ActType.SILU, conv1d)
            conv_op = top.ReshapeOp(T([1, num_k_heads * 2 + num_v_heads, head_v_dim, K]),
                                    conv_op,
                                    shape=[1, num_k_heads * 2 + num_v_heads, head_v_dim, -1],
                                    loc=L(conv1d + ".reshape_after_conv"),
                                    ip=ip).output

            def slice_heads(off, num, tag):
                op = top.SliceOp(T([1, num, head_v_dim, K]),
                                 conv_op,
                                 block_mlir.none_op,
                                 block_mlir.none_op,
                                 block_mlir.none_op,
                                 offset=[0, off, 0, 0],
                                 steps=[1, 1, 1, 1],
                                 ends=[1, off + num, head_v_dim, K],
                                 loc=L(in_proj_qkv + f".{tag}_slice_after_conv"),
                                 ip=ip).output
                return top.PermuteOp(T([1, K, num, head_v_dim]),
                                     op,
                                     order=[0, 3, 1, 2],
                                     loc=L(in_proj_qkv + f".{tag}_permute_after_conv"),
                                     ip=ip).output

            q_all = slice_heads(0, num_k_heads, "q")
            k_all = slice_heads(num_k_heads, num_k_heads, "k")
            v_all = slice_heads(num_k_heads * 2, num_v_heads, "v")
            beta_all = top.SigmoidOp(T([1, K, num_v_heads]),
                                     b_op,
                                     loc=L(in_proj_b + ".sigmoid"),
                                     ip=ip).output
            # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            a_op = top.SoftplusOp(T([1, K, num_v_heads]),
                                  a_op,
                                  loc=L(in_proj_a + ".softplus"),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(A_log + ".weight", [1, 1, num_v_heads])
            g_all = top.MulOp(T([1, K, num_v_heads]), [a_op, weight_op],
                              loc=L(in_proj_a + ".mul"),
                              ip=ip).output

            def slice_token(op, j, shape, tag):
                ends = list(shape)
                ends[1] = j + 1
                offset = [0] * len(shape)
                offset[1] = j
                return top.SliceOp(T(shape),
                                   op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=offset,
                                   steps=[1] * len(shape),
                                   ends=ends,
                                   loc=L(f"{tag}_step{j}"),
                                   ip=ip).output

            # ---- chained single-step recurrent rule with explicit state outputs ----
            state_op = in2_op
            attn_outs, states = [], []
            for j in range(K):
                q_j = slice_token(q_all, j, [1, 1, num_k_heads, head_k_dim], "q")
                k_j = slice_token(k_all, j, [1, 1, num_k_heads, head_k_dim], "k")
                v_j = slice_token(v_all, j, [1, 1, num_v_heads, head_v_dim], "v")
                g_j = slice_token(g_all, j, [1, 1, num_v_heads], "g")
                beta_j = slice_token(beta_all, j, [1, 1, num_v_heads], "beta")
                rec = top.RecurrentGatedDeltaRuleOp(
                    T([1, 1, num_v_heads, head_v_dim]),
                    T(recurrent_state_shape),
                    q_j,
                    k_j,
                    v_j,
                    g_j,
                    beta_j,
                    state_op,
                    num_k_heads=num_k_heads,
                    num_v_heads=num_v_heads,
                    d=head_v_dim,
                    use_qk_l2norm=True,
                    scale=scale,
                    loc=self.get_loc([
                        TOP_PATH + f"recurrent_gated_delta_rule_step{j}",
                        f"recurrent_states_step{j}"
                    ], block_mlir),
                    ip=ip)
                attn_outs.append(rec.attn_out)
                states.append(rec.new_state)
                state_op = rec.new_state
            recurrent_states_all = top.ConcatOp(T(recurrent_all_shape),
                                                states,
                                                axis=0,
                                                loc=L("recurrent_states_all"),
                                                ip=ip).output
            core_attn_op = top.ConcatOp(T([1, K, num_v_heads, head_v_dim]),
                                        attn_outs,
                                        axis=1,
                                        loc=L(TOP_PATH + "recurrent_gated_delta_rule"),
                                        ip=ip).output
            # RmsNormGated
            core_attn_op = self.rms_norm(block_mlir,
                                         core_attn_op,
                                         linear_norm,
                                         eps=self.rms_norm_eps)
            core_attn_op = top.ReshapeOp(T([1, K, value_dim]),
                                         core_attn_op,
                                         shape=[1, -1, value_dim],
                                         loc=L(in_proj_qkv + ".core_attn_reshape"),
                                         ip=ip).output
            core_attn_op = top.MulOp(T([1, K, value_dim]), [core_attn_op, z_op],
                                     loc=L(in_proj_z + ".core_attn_mul"),
                                     ip=ip).output
            o_op = self.linear(block_mlir,
                               out_proj,
                               core_attn_op, [value_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(out_proj + ".add"), ip=ip).output
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op, conv_states_all, recurrent_states_all])
            self.save_mlir_module(block_mlir, name)

        gen_block()
        gen_block_cache()
        if self.do_mtp:
            gen_block_verify()

    # ------------------------------------------------------------------
    # MTP (NextN) draft head
    # ------------------------------------------------------------------
    def gen_mtp_mlir(self):
        """Generate the MTP draft-head nets.

        Semantics (llama.cpp / vLLM Qwen3-Next MTP):
            x = fc(concat([rms(embeds, pre_fc_norm_embedding),
                           rms(hidden, pre_fc_norm_hidden)], -1))
            x = full_attention_layer(x)          # mtp.layers.0 (gated q, q/k norm, MRoPE)
            out = rms(x, mtp.norm)               # -> shared lm_head
        ``hidden`` is the trunk's final-norm output at position t (the same
        tensor fed to lm_head) and ``embeds`` is the embedding of the token at
        position t+1; the output predicts the token at position t+2.

        Nets:
          mtp_block       : prefill over L<=max_input_length positions (dynamic)
          mtp_block_cache : one draft step against the MTP KV history (io_alone)
        """
        tqdm.write("generate mtp_block / mtp_block_cache mlir ...")
        w = self.model_info.weights
        TOP_PATH = f"{w[LlmList.MTP_LAYERS]}.0."
        paths = self._fa_paths(TOP_PATH)
        fc = w[LlmList.MTP_FC]
        enorm = w[LlmList.MTP_PRE_FC_NORM_EMBED]
        hnorm = w[LlmList.MTP_PRE_FC_NORM_HIDDEN]
        mtp_norm = w[LlmList.MTP_NORM]
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"
        # ===== save weight ===========
        weight_file = "mtp_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.save_small_attn_mask_weight(weight_dict)
        self.set_common_weight(enorm, weight_dict, self.rmsnorm_type)
        self.set_common_weight(hnorm, weight_dict, self.rmsnorm_type)
        self.set_common_weight(mtp_norm, weight_dict, self.rmsnorm_type)
        self.set_linear_weight(fc, weight_dict)
        self._fa_save_weights(paths, weight_dict, None)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weight_keys.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        def mtp_body(block_mlir, hidden_op, embeds_op, pos_op, L_len, mask_op, mask_size, kv_mode,
                     kv_len, hist_k, hist_v):
            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)
            ip = block_mlir.insert_point
            input_shape = [1, L_len, self.hidden_size]
            h_op = self.rms_norm(block_mlir, hidden_op, hnorm)
            e_op = self.rms_norm(block_mlir, embeds_op, enorm)
            # embedding first, hidden second (vLLM / llama.cpp order)
            cat_op = top.ConcatOp(T([1, L_len, 2 * self.hidden_size]), [e_op, h_op],
                                  axis=2,
                                  loc=L(fc + ".concat"),
                                  ip=ip).output
            x_op = self.linear(block_mlir, fc, cat_op, [2 * self.hidden_size, self.hidden_size],
                               input_shape)
            o_op, k_op, v_op = self._fa_attention(block_mlir,
                                                  paths,
                                                  x_op,
                                                  pos_op,
                                                  L_len,
                                                  1,
                                                  mask_op,
                                                  mask_size,
                                                  kv_mode=kv_mode,
                                                  kv_len=kv_len,
                                                  hist_k=hist_k,
                                                  hist_v=hist_v)
            new_op = self._fa_mlp(block_mlir,
                                  input_shape,
                                  o_op,
                                  paths,
                                  None,
                                  do_norm=False,
                                  norm_path="",
                                  is_last=True)
            new_op = self.rms_norm(block_mlir, new_op, mtp_norm, "output_states")
            return [new_op, k_op, v_op]

        def gen_mtp_block():
            name = "mtp_block"
            L_len = self.max_input_length
            input_shape = [1, L_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = L_len
            kv_shape = [1, L_len, self.num_key_value_heads, self.head_dim]
            mask_shape = [1, 1, L_len, L_len]
            if self.use_small_mask():
                input_shapes = [input_shape, input_shape, id_shape]
                input_types = ["F32", "F32", "INT32"]
            else:
                input_shapes = [input_shape, input_shape, id_shape, mask_shape]
                input_types = ["F32", "F32", "INT32", "F32"]
            block_mlir = MLIRImporter(input_shapes, [input_shape, kv_shape, kv_shape],
                                      name,
                                      self.platform,
                                      input_types,
                                      weight_file=f"../{weight_file}")
            L = lambda name: self.get_loc(name, block_mlir)
            in0_op = block_mlir.create_input_op(L("hidden_states"), 0)
            in1_op = block_mlir.create_input_op(L("input_embeds"), 1)
            in2_op = block_mlir.create_input_op(L("position_ids"), 2)
            in3_op = block_mlir.create_input_op(L("attention_mask"), 3) \
                if not self.use_small_mask() else None
            mask_op, mask_size = self.get_fattention_mask_op(block_mlir, in3_op)
            outs = mtp_body(block_mlir, in0_op, in1_op, in2_op, L_len, mask_op, mask_size, "none",
                            0, None, None)
            block_mlir.create_return_op(outs)
            self.save_mlir_module(block_mlir, name)

        def gen_mtp_block_cache():
            name = "mtp_block_cache"
            input_shape = [1, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            mask_shape = [1, 1, 1, self.seq_length + 1]
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter(
                [input_shape, input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                self.platform, ["F32", "F32", "INT32", "F32", "F32", "F32"],
                weight_file=f"../{weight_file}")
            L = lambda name: self.get_loc(name, block_mlir)
            in0_op = block_mlir.create_input_op(L("hidden_states"), 0)
            in1_op = block_mlir.create_input_op(L("input_embeds"), 1)
            in2_op = block_mlir.create_input_op(L("position_ids"), 2)
            in3_op = block_mlir.create_input_op(L("attention_mask"), 3)
            in4_op = block_mlir.create_input_op(L("history_k"), 4)
            in5_op = block_mlir.create_input_op(L("history_v"), 5)
            self.create_decode_mask_placeholder(block_mlir)
            outs = mtp_body(block_mlir, in0_op, in1_op, in2_op, 1, in3_op, 0, "concat",
                            self.seq_length, in4_op, in5_op)
            block_mlir.create_return_op(outs)
            self.save_mlir_module(block_mlir, name)

        gen_mtp_block()
        gen_mtp_block_cache()

    @override
    def gen_block_mlir(self, idx: int):
        if self.llm_config.layer_types[idx] == "full_attention":
            self.gen_block_full_attn_mlir(idx)
        elif self.llm_config.layer_types[idx] == "linear_attention":
            self.gen_block_linear_attn_mlir(idx)
        else:
            raise ValueError(
                f"Unsupported block type {self.llm_config.layer_types[idx]} at index {idx}")

    @override
    def compile_block_cache(self, layer_id):
        if self.llm_config.layer_types[layer_id] == "full_attention":
            return super().compile_block_cache(layer_id)
        name = f"block_cache_{layer_id}"
        if self.register_bmodel(name):
            return
        self.submit_deploy_task(
            name,
            [
                f'--quantize {self.quantize}',
                f'--q_group_size {self.q_group_size}',
                '--quant_input',
                '--quant_output',
                '--same_addr 0:0,1:1',
            ],
            symmetric=self._get_block_symmetric(layer_id),
            addr_mode='io_alone',
        )

    @override
    def compile_block_cache_chunk(self, layer_id):
        # decode chunk only applies to full_attention layers; linear attention
        # layers use recurrent/conv state (no kv-length), so skip them.
        if self.llm_config.layer_types[layer_id] == "full_attention":
            return super().compile_block_cache_chunk(layer_id)
        return

    @override
    def compile_block_kv(self, layer_id):
        if self.llm_config.layer_types[layer_id] != "full_attention":
            return
        super().compile_block_kv(layer_id)
