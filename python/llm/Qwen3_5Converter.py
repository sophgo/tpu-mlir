# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import math
from .LlmConverter import *
from typing_extensions import override


class Qwen3_5Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_pixels = args.max_pixels
        if self.max_pixels == 0 or self.max_pixels % (32 * 32) != 0:
            raise RuntimeError(
                f"max_pixels values must be multiples of 32*32 and non-zero: {args.max_pixels}")
        self.do_vit = True
        self.dynamic = True  # force dynamic
        self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM
        # vision config
        self.init_vconfig()
        self.vit_path = "model.visual"

        # extern compiles
        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}

    def init_vconfig(self):
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.temporal_patch_size = self.vconfig.temporal_patch_size
        self.spatial_merge_size = self.vconfig.spatial_merge_size
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

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = QWEN3VL_INFO

    @override
    def rotary_embedding(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
        rotary_embed = Qwen2VLRotaryEmbedding(self.llm_config)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(
            1, 1, self.seq_length).expand(3, 1, self.seq_length)
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        cos, sin = rotary_embed(x, position_ids)
        cos = cos[0].reshape(self.seq_length, 1, -1)
        sin = sin[0].reshape(self.seq_length, 1, -1)
        assert (cos.shape[-1] == self.head_dim)
        assert (sin.shape[-1] == self.head_dim)
        # half
        cos = cos[:, :, :self.head_dim // 2]
        sin = sin[:, :, :self.head_dim // 2]
        return cos.numpy(), sin.numpy()  #[seq, 1, 64]

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
        freqs = np.arange(0, 3 * self.head_dim // 2,
                          dtype=np.int32).reshape(3, 1, self.head_dim // 2)
        freqs_t = self.apply_interleaved_mrope(freqs)  # [1, 64]
        freqs_t = np.tile(freqs_t, (1, 2))  # [1, 128]
        return freqs_t.astype(np.float32)

    def mrope_batch(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, self.head_dim // 2])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([self.batch, 3, dim, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, self.batch, dim]),
                               in_op,
                               order=[1, 3, 0, 2],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, self.batch, dim]),
                               new_op,
                               shape=[3 * self.head_dim // 2, self.batch, -1],
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, self.batch, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([self.batch, dim, 1, self.head_dim]),
                               new_op,
                               order=[2, 3, 0, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    def mrope(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, 1, dim]),
                               in_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, 1, dim]),
                               new_op,
                               shape=[3 * self.head_dim // 2, 1, -1],
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, 1, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                               new_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        # cos & sin MROPE
        is_decode = pos_op.type.shape[-1] == 1
        if is_decode and self.use_insert:
            cos_op = self.mrope_batch(mlir_gen, pos_op, rotary_cos)
            sin_op = self.mrope_batch(mlir_gen, pos_op, rotary_sin)
        else:
            cos_op = self.mrope(mlir_gen, pos_op, rotary_cos)
            sin_op = self.mrope(mlir_gen, pos_op, rotary_sin)
        q_op_shape = q_op.type.shape
        q_op = top.RopeOp(mlir_gen.get_tensor_type(q_op_shape),
                          q_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("q_proj", mlir_gen),
                          ip=mlir_gen.insert_point).output
        k_op_shape = k_op.type.shape
        k_op = top.RopeOp(mlir_gen.get_tensor_type(k_op_shape),
                          k_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("k_cache", mlir_gen),
                          ip=mlir_gen.insert_point).output
        # q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")
        # k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")
        return q_op, k_op

    def vision_rotary(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
        head_dim = self.vconfig.hidden_size // self.vnum_heads
        rotary_embed = VisionRotaryEmbedding(head_dim // 2)
        freqs = rotary_embed(self.num_patches)
        return freqs.cos().numpy(), freqs.sin().numpy()

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
            self.set_linear_weight(linear_fc1, weights_dict)
            self.set_linear_weight(linear_fc2, weights_dict)
            for i in range(self.depth):
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm2", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc1", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc2", weights_dict)

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
            Platform.LLM,
            input_types,
            weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

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
        mlir_txt = vit_mlir.print_module()
        if not os.path.exists(name):
            os.mkdir(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

    def gen_block_full_attn_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} full attention mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
        q_norm = TOP_PATH + self.model_info.weights[LlmList.Q_NORM]
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
        k_norm = TOP_PATH + self.model_info.weights[LlmList.K_NORM]
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.num_device < 2 and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict, self.rmsnorm_type)
        # q_proj split if not do lora
        self.set_linear_weight(q_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(k_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(v_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(o_proj, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(q_norm, weight_dict, self.rmsnorm_type)
        self.set_common_weight(k_norm, weight_dict, self.rmsnorm_type)
        self.set_common_weight(post_attn_ln, weight_dict, self.rmsnorm_type)
        self.set_linear_weight(mlp_gate, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_up, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_down, weight_dict, do_lora=self.do_lora)
        if do_norm:
            self.set_common_weight(norm, weight_dict, self.rmsnorm_type)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weights.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            batch = input_shape[0]
            len = input_shape[1]
            new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            if not self.use_mlp:
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
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        # create block mlir
        def gen_block_by_length(name: str, input_len: int):
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = input_len
            mask_shape = [1, 1, input_len, input_len]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter([input_shape, id_shape, mask_shape],
                                      [input_shape, kv_shape, kv_shape],
                                      name,
                                      Platform.LLM, ["F32", "INT32", "F32"],
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

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

            # q_proj
            ## For Qwen3.5, q_proj is fused with gate, so the output dim is 2x q_dim, and will be splitted later
            ## TODO: maybe we can split weight, but need to consider the case of lora
            q_dim = self.num_attention_heads * self.head_dim
            q_gate_op = self.linear(
                block_mlir,
                q_proj,
                ln_op,
                [self.hidden_size, q_dim * 2],  # q_proj with gate, so output dim is 2x q_dim
                [1, input_len, q_dim * 2],
                do_lora=self.do_lora)
            q_op = top.SliceOp(T([1, input_len, q_dim]),
                               q_gate_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, 0, 0],
                               steps=[1, 1, 1],
                               ends=[1, input_len, q_dim],
                               loc=self.get_loc(q_proj + ".q_slice", block_mlir),
                               ip=ip).output
            gate_op = top.SliceOp(T([1, input_len, q_dim]),
                                  q_gate_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  offset=[0, 0, q_dim],
                                  steps=[1, 1, 1],
                                  ends=[1, input_len, q_dim * 2],
                                  loc=self.get_loc(q_proj + ".gate_slice", block_mlir),
                                  ip=ip).output

            gate_op = top.SigmoidOp(T([1, input_len, q_dim]),
                                    gate_op,
                                    loc=L(mlp_gate + ".gate_sigmoid"),
                                    ip=ip).output

            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)

            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, self.head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape),
                                 k_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L(k_proj + ".reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape),
                                 v_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L("v_cache"),
                                 ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)
            k_op = self.rms_norm(block_mlir, k_op, k_norm)

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),
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
                                     mq=input_len,
                                     mk=input_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            fa_op = top.MulOp(T([1, input_len, q_dim]), [fa_op, gate_op],
                              loc=L(mlp_gate + ".gate_mul"),
                              ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            target = os.path.join(name, f"{name}.mlir")
            with open(target, "w") as f:
                f.write(mlir_txt)

        def gen_block():
            name = f"block_{idx}"
            if self.share_prompt:
                name = f"block_prompt_{idx}"
                gen_block_by_length(name, self.max_prefill_kv_length)
                return

            gen_block_by_length(name, self.max_input_length)
            return

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [self.batch, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            mask_len = self.seq_length if self.use_insert else self.seq_length + 1
            if self.use_insert:
                id_shape[0] = self.batch
            id_shape[-1] = 1
            mask_shape = [self.batch, 1, 1, mask_len]
            history_shape = [self.batch, self.seq_length, self.num_key_value_heads, self.head_dim]

            q_shape = [self.batch, 1, self.num_attention_heads, self.head_dim]
            kv_shape = [self.batch, 1, self.num_key_value_heads, self.head_dim]
            output_shapes = [input_shape] if self.use_insert else [input_shape, kv_shape, kv_shape]
            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                output_shapes,
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                lora_rank=self.lora_rank,
                weight_file=f"../{weight_file}")

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
            q_gate_op = self.linear(block_mlir,
                                    q_proj,
                                    ln_op, [self.hidden_size, q_dim * 2],
                                    [self.batch, 1, q_dim * 2],
                                    do_lora=self.do_lora)
            q_op = top.SliceOp(T([self.batch, 1, q_dim]),
                               q_gate_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, 0, 0],
                               steps=[1, 1, 1],
                               ends=[self.batch, 1, q_dim],
                               loc=self.get_loc(q_proj + ".q_slice", block_mlir),
                               ip=ip).output
            gate_op = top.SliceOp(T([self.batch, 1, q_dim]),
                                  q_gate_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  offset=[0, 0, q_dim],
                                  steps=[1, 1, 1],
                                  ends=[self.batch, 1, q_dim * 2],
                                  loc=self.get_loc(q_proj + ".gate_slice", block_mlir),
                                  ip=ip).output
            gate_op = top.SigmoidOp(T([self.batch, 1, q_dim]),
                                    gate_op,
                                    loc=L(mlp_gate + ".gate_sigmoid"),
                                    ip=ip).output
            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [self.batch, 1, self.kv_dim],
                               do_lora=self.do_lora)
            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [self.batch, 1, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)
            k_op = self.rms_norm(block_mlir, k_op, k_norm)
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            if not self.use_insert:
                return_ops.append(k_op)
                return_ops.append(v_op)
            # ====== kv concat ========
            if not self.use_insert:
                k_op = top.ConcatOp(T(
                    [1, self.seq_length + 1, self.num_key_value_heads, self.head_dim]),
                                    [in3_op, k_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(k_proj + ".concat"),
                                    ip=ip).output
                v_op = top.ConcatOp(T(
                    [1, self.seq_length + 1, self.num_key_value_heads, self.head_dim]),
                                    [in4_op, v_op],
                                    axis=1,
                                    only_merge=True,
                                    loc=L(v_proj + ".concat"),
                                    ip=ip).output
            else:
                k_op = top.InsertOp(T(
                    [self.batch, self.seq_length, self.num_key_value_heads, self.head_dim]),
                                    in3_op,
                                    rhs=k_op,
                                    axis=1,
                                    offset=self.seq_length - 1,
                                    loc=L(k_proj + ".insert"),
                                    ip=ip).output
                v_op = top.InsertOp(T(
                    [self.batch, self.seq_length, self.num_key_value_heads, self.head_dim]),
                                    in4_op,
                                    rhs=v_op,
                                    axis=1,
                                    offset=self.seq_length - 1,
                                    loc=L(v_proj + ".insert"),
                                    ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([self.batch, 1, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=self.batch,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=1,
                                     mk=mask_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            fa_op = top.MulOp(T([self.batch, 1, q_dim]), [fa_op, gate_op],
                              loc=L(mlp_gate + ".gate_mul"),
                              ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_block_with_kv():
            # Generate block with kv cache related operations
            name = f"block_{idx}"
            input_len = self.max_input_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            max_kv_len = self.max_prefill_kv_length + input_len
            mask_shape = [1, 1, input_len, max_kv_len]
            history_shape = [1, self.max_prefill_kv_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]

            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                lora_rank=self.lora_rank,
                weight_file=f"../{weight_file}")

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
            q_gate_op = self.linear(block_mlir,
                                    q_proj,
                                    ln_op, [self.hidden_size, q_dim * 2], [1, input_len, q_dim * 2],
                                    do_lora=self.do_lora)
            q_op = top.SliceOp(T([1, input_len, q_dim]),
                               q_gate_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               block_mlir.none_op,
                               offset=[0, 0, 0],
                               steps=[1, 1, 1],
                               ends=[1, input_len, q_dim],
                               loc=self.get_loc(q_proj + ".q_slice", block_mlir),
                               ip=ip).output
            gate_op = top.SliceOp(T([1, input_len, q_dim]),
                                  q_gate_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  block_mlir.none_op,
                                  offset=[0, 0, q_dim],
                                  steps=[1, 1, 1],
                                  ends=[1, input_len, q_dim * 2],
                                  loc=self.get_loc(q_proj + ".gate_slice", block_mlir),
                                  ip=ip).output

            gate_op = top.SigmoidOp(T([1, input_len, q_dim]),
                                    gate_op,
                                    loc=L(mlp_gate + ".gate_sigmoid"),
                                    ip=ip).output
            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, self.head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape),
                                 k_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L(k_proj + ".reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape),
                                 v_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L("v_cache"),
                                 ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, q_norm)
            k_op = self.rms_norm(block_mlir, k_op, k_norm)
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in3_op, k_op],
                                axis=1,
                                only_merge=True,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in4_op, v_op],
                                axis=1,
                                only_merge=True,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),
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
                                     mq=input_len,
                                     mk=max_kv_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            fa_op = top.MulOp(T([self.batch, 1, q_dim]), [fa_op, gate_op],
                              loc=L(mlp_gate + ".gate_mul"),
                              ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        if self.use_block_with_kv:
            gen_block_with_kv()
        else:
            gen_block()
        if self.share_prompt:
            gen_block()
        gen_block_cache()

    def gen_block_linear_attn_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} linear attention mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        A_log = TOP_PATH + "linear_attn.A_log"
        conv1d = TOP_PATH + "linear_attn.conv1d"
        dt_bias = TOP_PATH + "linear_attn.dt_bias"
        in_proj_a = TOP_PATH + "linear_attn.in_proj_a"
        in_proj_b = TOP_PATH + "linear_attn.in_proj_b"
        in_proj_qkv = TOP_PATH + "linear_attn.in_proj_qkv"
        in_proj_z = TOP_PATH + "linear_attn.in_proj_z"
        linear_norm = TOP_PATH + "linear_attn.norm"
        out_proj = TOP_PATH + "linear_attn.out_proj"
        weight_file = f"block_{idx}_top_weights.npz"
        A_log_data = self.model.read(A_log)
        A_log_data = -np.exp(A_log_data)

        weight_dict = {A_log + ".weight": A_log_data}
        self.set_common_weight(input_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(post_attn_ln, weight_dict, self.rmsnorm_type)
        self.set_common_weight(conv1d, weight_dict)
        self.set_common_weight(dt_bias, weight_dict)
        self.set_linear_weight(in_proj_a, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(in_proj_b, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(in_proj_qkv, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(in_proj_z, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(linear_norm, weight_dict,
                               WeightType.RMSNORM)  # not zero centered norm
        self.set_linear_weight(out_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_gate, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_up, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_down, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(norm, weight_dict, self.rmsnorm_type)
        eye_key = TOP_PATH + "linear_attn.eye"
        weight_dict[eye_key] = np.eye(64, dtype=np.float32)
        np.savez(weight_file, **weight_dict)

        chunk_size = 64
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

            if not self.use_mlp:
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
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        def gen_block_by_length(name: str, input_len: int):
            input_shape = [1, input_len, self.hidden_size]
            conv_shape = [1, conv_dim, conv_kernel_size]
            recurrent_shape = [1, num_v_heads, head_v_dim, head_v_dim]
            block_mlir = MLIRImporter([input_shape, recurrent_shape],
                                      [input_shape, conv_shape, recurrent_shape],
                                      name,
                                      Platform.LLM, ["F32", "F32"],
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("recurrent_states"), 1)
            in0_op = self.rms_norm(block_mlir, in0_op, input_ln)
            mixed_qkv_op = self.linear(block_mlir,
                                       in_proj_qkv,
                                       in0_op, [self.hidden_size, conv_dim],
                                       [1, input_len, conv_dim],
                                       do_lora=self.do_lora)
            z_op = self.linear(block_mlir,
                               in_proj_z,
                               in0_op, [self.hidden_size, value_dim], [1, input_len, value_dim],
                               do_lora=self.do_lora)
            b_op = self.linear(block_mlir,
                               in_proj_b,
                               in0_op, [self.hidden_size, num_v_heads], [1, input_len, num_v_heads],
                               do_lora=self.do_lora)
            a_op = self.linear(block_mlir,
                               in_proj_a,
                               in0_op, [self.hidden_size, num_v_heads], [1, input_len, num_v_heads],
                               do_lora=self.do_lora)
            mixed_qkv_op = top.PermuteOp(T([1, conv_dim, input_len]),
                                         mixed_qkv_op,
                                         order=[0, 2, 1],
                                         loc=L(in_proj_qkv + ".permute"),
                                         ip=ip).output
            conv_state = top.SliceOp(T([1, conv_dim, conv_kernel_size]),
                                     mixed_qkv_op,
                                     block_mlir.none_op,
                                     block_mlir.none_op,
                                     block_mlir.none_op,
                                     offset=[0, 0, input_len - conv_kernel_size],
                                     steps=[1, 1, 1],
                                     ends=[1, conv_dim, input_len],
                                     loc=L(conv1d + ".conv_state_slice"),
                                     ip=ip).output
            return_ops = [conv_state]
            # conv1d to conv2d
            mixed_qkv_op = top.ReshapeOp(T([1, conv_dim, 1, input_len]),
                                         mixed_qkv_op,
                                         shape=[1, conv_dim, 1, input_len],
                                         loc=L(in_proj_qkv + ".reshape_to_conv2d"),
                                         ip=ip).output
            weight_op = block_mlir.create_weight_op(conv1d + ".weight",
                                                    [conv_dim, 1, 1, conv_kernel_size])
            conv_op = top.ConvOp(T([1, conv_dim, 1, input_len]),
                                 mixed_qkv_op,
                                 weight_op,
                                 block_mlir.none_op,
                                 kernel_shape=[1, conv_kernel_size],
                                 strides=[1, 1],
                                 group=conv_dim,
                                 pads=[0, conv_kernel_size - 1, 0, 0],
                                 loc=L(conv1d),
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
            q_op = top.PermuteOp(T([1, num_k_heads, input_len, head_k_dim]),
                                 q_op,
                                 order=[0, 1, 3, 2],
                                 loc=L(in_proj_qkv + ".q_permute_after_conv"),
                                 ip=ip).output
            k_op = top.PermuteOp(T([1, num_k_heads, input_len, head_k_dim]),
                                 k_op,
                                 order=[0, 1, 3, 2],
                                 loc=L(in_proj_qkv + ".k_permute_after_conv"),
                                 ip=ip).output
            v_op = top.PermuteOp(T([1, num_v_heads, input_len, head_v_dim]),
                                 v_op,
                                 order=[0, 1, 3, 2],
                                 loc=L(in_proj_qkv + ".v_permute_after_conv"),
                                 ip=ip).output
            beta_op = top.SigmoidOp(T([1, input_len, num_v_heads]),
                                    b_op,
                                    loc=L(in_proj_b + ".sigmoid"),
                                    ip=ip).output
            beta_op = top.PermuteOp(T([1, num_v_heads, input_len]),
                                    beta_op,
                                    order=[0, 2, 1],
                                    loc=L(in_proj_b + ".beta_permute"),
                                    ip=ip).output
            # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            weight_op = block_mlir.create_weight_op(dt_bias, [1, 1, num_v_heads])
            a_op = top.AddOp(T([1, input_len, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".add"),
                             ip=ip).output
            a_op = top.SoftplusOp(T([1, input_len, num_v_heads]),
                                  a_op,
                                  loc=L(in_proj_a + ".softplus"),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(A_log + ".weight", [1, 1, num_v_heads])
            g_op = top.MulOp(T([1, input_len, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".mul"),
                             ip=ip).output
            g_op = top.PermuteOp(T([1, num_v_heads, input_len]),
                                 g_op,
                                 order=[0, 2, 1],
                                 loc=L(in_proj_a + ".g_permute"),
                                 ip=ip).output
            # ================= chunk_gated_delta_rule ==================
            eye_op = block_mlir.create_weight_op(eye_key, [1, 1, 1, chunk_size, chunk_size])
            outputs = top.ChunkGatedDeltaRuleOp(T([1, input_len, num_v_heads, head_v_dim]),
                                                T([1, num_v_heads, head_v_dim, head_v_dim]),
                                                q_op,
                                                k_op,
                                                v_op,
                                                g_op,
                                                beta_op,
                                                in1_op,
                                                eye_op,
                                                chunk_size=chunk_size,
                                                scale=scale,
                                                loc=L(TOP_PATH + "chunk_gated_delta_rule"),
                                                ip=ip)
            core_attn_out = outputs.attn_out
            new_recurrent_state = outputs.new_recurrent_state
            return_ops.append(new_recurrent_state)

            # RmsNormGated
            core_attn_op = self.rms_norm(block_mlir,
                                         core_attn_out,
                                         linear_norm,
                                         eps=self.rms_norm_eps)
            # -> [1, input_len, value_dim]
            core_attn_op = top.ReshapeOp(T([1, input_len, value_dim]),
                                         core_attn_op,
                                         shape=[1, input_len, value_dim],
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
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            target = os.path.join(name, f"{name}.mlir")
            with open(target, "w") as f:
                f.write(mlir_txt)

        def gen_block():
            name = f"block_{idx}"
            gen_block_by_length(name, self.max_input_length)
            return

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [self.batch, 1, self.hidden_size]
            conv_state_shape = [self.batch, conv_dim, conv_kernel_size]
            recurrent_state_shape = [self.batch, num_v_heads, head_v_dim, head_v_dim]
            block_mlir = MLIRImporter([input_shape, conv_state_shape, recurrent_state_shape],
                                      [input_shape, conv_state_shape, recurrent_state_shape],
                                      name,
                                      Platform.LLM, ["F32", "F32", "F32"],
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("conv_state"), 1)
            in2_op = block_mlir.create_input_op(L("recurrent_state"), 2)
            return_ops = []
            in0_op = self.rms_norm(block_mlir, in0_op, input_ln)
            # eye_key no use, just the same with block
            block_mlir.create_weight_op(eye_key, [1, 1, 1, chunk_size, chunk_size],
                                        placeholder=True)
            # q_proj
            mixed_qkv_op = self.linear(block_mlir,
                                       in_proj_qkv,
                                       in0_op, [self.hidden_size, conv_dim],
                                       [self.batch, 1, conv_dim],
                                       do_lora=self.do_lora)
            z_op = self.linear(block_mlir,
                               in_proj_z,
                               in0_op, [self.hidden_size, value_dim], [self.batch, 1, value_dim],
                               do_lora=self.do_lora)
            b_op = self.linear(block_mlir,
                               in_proj_b,
                               in0_op, [self.hidden_size, num_v_heads],
                               [self.batch, 1, num_v_heads],
                               do_lora=self.do_lora)
            a_op = self.linear(block_mlir,
                               in_proj_a,
                               in0_op, [self.hidden_size, num_v_heads],
                               [self.batch, 1, num_v_heads],
                               do_lora=self.do_lora)

            mixed_qkv_op = top.ReshapeOp(T([self.batch, conv_dim, 1]),
                                         mixed_qkv_op,
                                         loc=L(in_proj_qkv + ".reshape"),
                                         ip=ip).output
            z_op = top.ReshapeOp(T([self.batch, num_v_heads, 1, head_v_dim]),
                                 z_op,
                                 shape=[self.batch, num_v_heads, 1, head_v_dim],
                                 loc=L(in_proj_z + ".reshape"),
                                 ip=ip).output
            # use_precomputed_states, causal_conv1d_update
            mixed_qkv_op = top.ConcatOp(T([self.batch, conv_dim, conv_kernel_size + 1]),
                                        [in1_op, mixed_qkv_op],
                                        axis=2,
                                        only_merge=False,
                                        loc=L(conv1d + ".concat"),
                                        ip=ip).output
            mixed_qkv_op = top.SliceOp(T([self.batch, conv_dim, conv_kernel_size]),
                                       mixed_qkv_op,
                                       block_mlir.none_op,
                                       block_mlir.none_op,
                                       block_mlir.none_op,
                                       offset=[0, 0, 1],
                                       steps=[1, 1, 1],
                                       ends=[self.batch, conv_dim, conv_kernel_size + 1],
                                       loc=L(conv1d + ".slice"),
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
            q_op = top.ReshapeOp(T([self.batch, 1, num_k_heads, head_k_dim]),
                                 q_op,
                                 shape=[self.batch, 1, num_k_heads, head_k_dim],
                                 loc=L(in_proj_qkv + ".q_reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T([self.batch, 1, num_k_heads, head_k_dim]),
                                 k_op,
                                 shape=[self.batch, 1, num_k_heads, head_k_dim],
                                 loc=L(in_proj_qkv + ".k_reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T([self.batch, num_v_heads, 1, head_v_dim]),
                                 v_op,
                                 shape=[self.batch, 1, num_v_heads, head_v_dim],
                                 loc=L(in_proj_qkv + ".v_reshape"),
                                 ip=ip).output
            beta_op = top.SigmoidOp(T([self.batch, 1, num_v_heads]),
                                    b_op,
                                    loc=L(in_proj_b + ".sigmoid"),
                                    ip=ip).output
            beta_op = top.ReshapeOp(T([self.batch, num_v_heads, 1, 1]),
                                    beta_op,
                                    shape=[self.batch, num_v_heads, 1, 1],
                                    loc=L(in_proj_b + ".reshape"),
                                    ip=ip).output
            # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            weight_op = block_mlir.create_weight_op(dt_bias, [1, 1, num_v_heads])
            a_op = top.AddOp(T([self.batch, 1, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".add"),
                             ip=ip).output
            a_op = top.SoftplusOp(T([self.batch, 1, num_v_heads]),
                                  a_op,
                                  loc=L(in_proj_a + ".softplus"),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(A_log + ".weight", [1, 1, num_v_heads])
            g_op = top.MulOp(T([self.batch, 1, num_v_heads]), [a_op, weight_op],
                             loc=L(in_proj_a + ".mul"),
                             ip=ip).output
            # ========== recurreent_gated_delta_rule =================
            q_op = self.l2norm(block_mlir, q_op, in_proj_qkv + ".q_l2norm", eps=1e-6)
            k_op = self.l2norm(block_mlir, k_op, in_proj_qkv + ".k_l2norm", eps=1e-6)
            q_op = top.MulConstOp(T([self.batch, 1, num_k_heads, head_k_dim]),
                                  q_op,
                                  const_val=scale,
                                  loc=L(in_proj_qkv + ".q_scale"),
                                  ip=ip).output
            g_op = top.ExpOp(T([self.batch, 1, num_v_heads]),
                             g_op,
                             loc=L(in_proj_a + ".exp"),
                             ip=ip).output
            g_op = top.ReshapeOp(T([self.batch, num_v_heads, 1, 1]),
                                 g_op,
                                 shape=[self.batch, num_v_heads, 1, 1],
                                 loc=L(in_proj_a + ".g_reshape"),
                                 ip=ip).output
            recurrent_op = top.MulOp(T([self.batch, num_v_heads, head_v_dim, head_v_dim]),
                                     [in2_op, g_op],
                                     loc=L(in_proj_a + ".recurrent"),
                                     ip=ip).output
            if num_k_heads < num_v_heads:
                # repeat interleave q_op and k_op to match value heads
                times = num_v_heads // num_k_heads
                q_op = top.TileOp(T([self.batch, 1, num_k_heads, times * head_k_dim]),
                                  q_op,
                                  tile=[1, 1, 1, times],
                                  loc=L(in_proj_qkv + ".q_tile"),
                                  ip=ip).output

                k_op = top.TileOp(T([self.batch, 1, num_k_heads, times * head_k_dim]),
                                  k_op,
                                  tile=[1, 1, 1, times],
                                  loc=L(in_proj_qkv + ".k_tile"),
                                  ip=ip).output

            q_op = top.ReshapeOp(T([self.batch, num_v_heads, 1, head_k_dim]),
                                 q_op,
                                 shape=[self.batch, num_v_heads, 1, head_k_dim],
                                 loc=L(in_proj_qkv + ".q_reshape2"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T([self.batch, num_v_heads, head_k_dim, 1]),
                                 k_op,
                                 shape=[self.batch, num_v_heads, head_k_dim, 1],
                                 loc=L(in_proj_qkv + ".k_reshape2"),
                                 ip=ip).output
            kv_mem_op = top.MulOp(T([self.batch, num_v_heads, head_v_dim, head_v_dim]),
                                  [recurrent_op, k_op],
                                  loc=L(in_proj_a + ".kv_mem"),
                                  ip=ip).output
            kv_mem_op = top.ReduceOp(T([self.batch, num_v_heads, 1, head_v_dim]),
                                     kv_mem_op,
                                     axes=[2],
                                     keepdims=True,
                                     mode=StringAttr.get("ReduceSum"),
                                     loc=L(in_proj_a + ".kv_mem_reduce"),
                                     ip=ip).output
            delta_op = top.SubOp(T([self.batch, num_v_heads, 1, head_v_dim]), [v_op, kv_mem_op],
                                 loc=L(in_proj_qkv + ".delta"),
                                 ip=ip).output
            delta_op = top.MulOp(T([self.batch, num_v_heads, 1, head_v_dim]), [delta_op, beta_op],
                                 loc=L(in_proj_b + ".delta_mul"),
                                 ip=ip).output
            delta_op = top.MulOp(T([self.batch, num_v_heads, head_k_dim, head_v_dim]),
                                 [delta_op, k_op],
                                 loc=L(in_proj_b + ".delta_k_mul"),
                                 ip=ip).output
            recurrent_op = top.AddOp(T([self.batch, num_v_heads, head_v_dim, head_v_dim]),
                                     [recurrent_op, delta_op],
                                     loc=L(in_proj_a + ".recurrent_update"),
                                     ip=ip).output
            return_ops.append(recurrent_op)  # new recurrent state
            core_attn_op = top.MatMulOp(T([self.batch, num_v_heads, 1, head_v_dim]),
                                        q_op,
                                        recurrent_op,
                                        block_mlir.none_op,
                                        loc=L("core_attention"),
                                        ip=ip).output
            # RmsNormGated
            core_attn_op = self.rms_norm(block_mlir,
                                         core_attn_op,
                                         linear_norm,
                                         eps=self.rms_norm_eps)
            core_attn_op = top.MulOp(T([self.batch, num_v_heads, 1, head_v_dim]),
                                     [core_attn_op, z_op],
                                     loc=L(in_proj_z + ".core_attn_mul"),
                                     ip=ip).output
            o_op = top.ReshapeOp(T([self.batch, 1, value_dim]),
                                 core_attn_op,
                                 shape=[self.batch, 1, value_dim],
                                 loc=L(out_proj + ".reshape"),
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
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        gen_block()
        gen_block_cache()

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
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return

        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}', '--quant_input',
            '--quant_output', f'--chip {self.chip}', '--addr_mode io_alone',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append(f'--same_addr 0:0,1:1,2:2')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")
