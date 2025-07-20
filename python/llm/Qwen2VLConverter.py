# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Qwen2VLConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_pixels = args.max_pixels
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        if args.max_pixels % (28 * 28) != 0:
            raise RuntimeError(
                "max_pixels is not a multiple of 28*28, please set max_pixels to a value that is a multiple of 28*28."
            )
        self.do_vit = True
        # vision config
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.temporal_patch_size = self.vconfig.temporal_patch_size
        self.spatial_merge_size = self.vconfig.spatial_merge_size
        self.in_channels = self.vconfig.in_chans
        self.depth = self.vconfig.depth
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        self.embed_dim = self.vconfig.embed_dim
        self.vnum_heads = self.vconfig.num_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.position_shape = [3, self.max_input_length]

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_type = LlmType.QWEN2

    @override
    def rotary_embedding(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
        rotary_embed = Qwen2VLRotaryEmbedding(self.config)
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

    def mrope(self, mlir_gen, in_op, name: str):
        mrope_section = getattr(self.config.rope_scaling, 'mrope_section', [16, 24, 24])
        in_shape = in_op.type.shape
        t_dim, h_dim, w_dim = mrope_section  # 16,24,24
        dim = in_shape[1]
        # slice cos_op = [1, dim 1, t_dim] + [1, dim, 1, h_dim] + [1, dim, 1, w_dim]
        t_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, t_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[0, 0, 0, 0],
                           steps=[1, 1, 1, 1],
                           ends=[1, dim, 1, t_dim],
                           loc=self.get_loc(name + ".slice.t", mlir_gen),
                           ip=mlir_gen.insert_point).output
        h_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, h_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[1, 0, 0, t_dim],
                           steps=[1, 1, 1, 1],
                           ends=[2, dim, 1, t_dim + h_dim],
                           loc=self.get_loc(name + ".slice.h", mlir_gen),
                           ip=mlir_gen.insert_point).output
        w_op = top.SliceOp(mlir_gen.get_tensor_type([1, dim, 1, w_dim]),
                           in_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           mlir_gen.none_op,
                           offset=[2, 0, 0, t_dim + h_dim],
                           steps=[1, 1, 1, 1],
                           ends=[3, dim, 1, t_dim + h_dim + w_dim],
                           loc=self.get_loc(name + ".slice.w", mlir_gen),
                           ip=mlir_gen.insert_point).output
        concat_op = top.ConcatOp(mlir_gen.get_tensor_type([1, dim, 1, t_dim + h_dim + w_dim]),
                                 [t_op, h_op, w_op],
                                 axis=3,
                                 loc=self.get_loc(name + ".concat", mlir_gen),
                                 ip=mlir_gen.insert_point).output
        tile_op = top.TileOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                             concat_op,
                             tile=[1, 1, 1, 2],
                             loc=self.get_loc(name + ".tile", mlir_gen),
                             ip=mlir_gen.insert_point).output
        return tile_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        dim = pos_op.type.shape[-1]

        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        cos_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_cos, mlir_gen),
                              ip=mlir_gen.insert_point).output
        # cos MROPE
        cos_op = self.mrope(mlir_gen, cos_op, rotary_cos)
        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        sin_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_sin, mlir_gen),
                              ip=mlir_gen.insert_point).output
        # sin MROPE
        sin_op = self.mrope(mlir_gen, sin_op, rotary_sin)
        # ===== q_proj rotary ========
        q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")

        # ===== k_proj rotary ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")
        return q_op, k_op

    def vision_rotary(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
        head_dim = self.vconfig.embed_dim // self.vnum_heads
        rotary_embed = VisionRotaryEmbedding(head_dim // 2)
        freqs = rotary_embed(self.num_patches)
        return freqs.cos().numpy(), freqs.sin().numpy()

    def vision_block(self, vit_mlir, id: int, in_op, cos_op, sin_op, mask_op):
        norm1 = f"visual.blocks.{id}.norm1"
        attn_q = f"visual.blocks.{id}.attn.q"
        attn_k = f"visual.blocks.{id}.attn.k"
        attn_v = f"visual.blocks.{id}.attn.v"
        attn_proj = f"visual.blocks.{id}.attn.proj"
        norm2 = f"visual.blocks.{id}.norm2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_attention(in_op):
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, eps=1e-6)
            hidden_shape = [self.num_patches, self.embed_dim]
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
            qk_shape = [1, self.num_patches, self.vnum_heads, self.vhead_dim]
            q_op = top.ReshapeOp(T(qk_shape), q_op, loc=L(attn_q + ".reshape"), ip=ip).output

            k_op = top.ReshapeOp(T(qk_shape), k_op, loc=L(attn_k + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(qk_shape), v_op, loc=L(attn_v + ".reshape"), ip=ip).output
            q_op = self.rotary_pos(vit_mlir, q_op, cos_op, sin_op, attn_q + ".rotary")
            k_op = self.rotary_pos(vit_mlir, k_op, cos_op, sin_op, attn_k + ".rotary")
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
                                     mq=self.num_patches,
                                     mk=self.num_patches,
                                     loc=L(f"visual.blocks.{id}.fattention"),
                                     ip=ip).output
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  loc=L(f"visual.blocks.{id}.fattention.reshape"),
                                  ip=ip).output
            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 fa_op, [self.embed_dim, self.embed_dim],
                                 [self.num_patches, self.embed_dim],
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        def vision_mlp(in_op):
            in_shape = [self.num_patches, self.embed_dim]
            mlp_fc1 = f"visual.blocks.{id}.mlp.fc1"
            mlp_fc2 = f"visual.blocks.{id}.mlp.fc2"
            mlp_dim = int(self.vconfig.mlp_ratio * self.embed_dim)
            new_op = self.layer_norm(vit_mlir, in_op, norm2, eps=1e-6)
            fc1_op = self.linear(vit_mlir, mlp_fc1, new_op, [self.embed_dim, mlp_dim],
                                 [self.num_patches, mlp_dim])
            # quick gelu
            act_op = self.activate(vit_mlir, fc1_op, self.vconfig.hidden_act, mlp_fc1)
            up_op = self.linear(vit_mlir, mlp_fc2, act_op, [mlp_dim, self.embed_dim],
                                [self.num_patches, self.embed_dim])
            new_op = top.AddOp(T(in_shape), [in_op, up_op], loc=L(mlp_fc2 + ".add"), ip=ip).output
            return new_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        patch_embed = "visual.patch_embed.proj"
        rotary_cos = "visual.rotary.cos"
        rotary_sin = "visual.rotary.sin"
        merger_ln_q = "visual.merger.ln_q"
        merger_mlp0 = "visual.merger.mlp.0"
        merger_mlp2 = "visual.merger.mlp.2"

        def save_weights():
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + ".weight": cos,
                rotary_sin + ".weight": sin,
            }
            data = self.model.read(patch_embed + ".weight").reshape(self.embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            weights_dict[patch_embed + ".weight"] = data
            self.set_common_weight(merger_ln_q, weights_dict)
            self.set_linear_weight(merger_mlp0, weights_dict)
            self.set_linear_weight(merger_mlp2, weights_dict)
            for i in range(self.depth):
                self.set_common_weight(f"visual.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"visual.blocks.{i}.norm2", weights_dict)
                self.set_linear_weight(f"visual.blocks.{i}.attn.proj", weights_dict)
                self.set_linear_weight(f"visual.blocks.{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(f"visual.blocks.{i}.mlp.fc2", weights_dict)
                # split qkv
                # self.set_linear_weight(f"visual.blocks.{i}.attn.qkv", weights_dict)
                weight = self.model.read(f"visual.blocks.{i}.attn.qkv.weight").reshape(
                    3 * self.embed_dim, self.embed_dim)
                bias = self.model.read(f"visual.blocks.{i}.attn.qkv.bias").reshape(3 *
                                                                                   self.embed_dim)
                q_w = weight[:self.embed_dim, :]
                k_w = weight[self.embed_dim:2 * self.embed_dim, :]
                v_w = weight[2 * self.embed_dim:, :]
                q_b = bias[:self.embed_dim]
                k_b = bias[self.embed_dim:2 * self.embed_dim]
                v_b = bias[2 * self.embed_dim:]
                q_w = np.ascontiguousarray(np.transpose(q_w, (1, 0)))
                k_w = np.ascontiguousarray(np.transpose(k_w, (1, 0)))
                v_w = np.ascontiguousarray(np.transpose(v_w, (1, 0)))
                weights_dict[f"visual.blocks.{i}.attn.q.weight"] = q_w
                weights_dict[f"visual.blocks.{i}.attn.k.weight"] = k_w
                weights_dict[f"visual.blocks.{i}.attn.v.weight"] = v_w
                weights_dict[f"visual.blocks.{i}.attn.q.bias"] = q_b
                weights_dict[f"visual.blocks.{i}.attn.k.bias"] = k_b
                weights_dict[f"visual.blocks.{i}.attn.v.bias"] = v_b
            # save weights
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.patch_dim]
        position_shape = [self.num_patches, 2]
        mask_shape = [1, 1, self.num_patches, self.num_patches]
        out_dim = self.num_patches // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.hidden_size]
        input_shapes = [in_shape, position_shape, mask_shape]
        input_types = ['F32', 'INT32', 'F32']

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                "vit",
                                Platform.LLM,
                                input_types,
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        in1_op = vit_mlir.create_input_op(L('position_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('full_attn_mask'), 2)

        new_weight = vit_mlir.create_weight_op(patch_embed + ".weight",
                                               [self.patch_dim, self.embed_dim])
        new_op = top.MatMulOp(T([self.num_patches, self.embed_dim]),
                              in0_op,
                              new_weight,
                              vit_mlir.none_op,
                              loc=L(patch_embed),
                              ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight", [self.num_patches, 20])
        cos_op = top.GatherOp(T([self.num_patches, 2, 20]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_cos),
                              ip=ip).output
        cos_op = top.ReshapeOp(T([1, self.num_patches, 1, 40]),
                               cos_op,
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        cos_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),
                            cos_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_cos + ".tile"),
                            ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight", [self.num_patches, 20])
        sin_op = top.GatherOp(T([self.num_patches, 2, 20]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_sin),
                              ip=ip).output
        sin_op = top.ReshapeOp(T([1, self.num_patches, 1, 40]),
                               sin_op,
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        sin_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),
                            sin_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_sin + ".tile"),
                            ip=ip).output
        for id in range(self.depth):
            new_op = self.vision_block(vit_mlir, id, new_op, cos_op, sin_op, in2_op)

        # merge
        new_op = self.layer_norm(vit_mlir, new_op, merger_ln_q, eps=1e-6)
        out_dim = self.embed_dim * (self.spatial_merge_size**2)
        in_dim = self.num_patches // (self.spatial_merge_size**2)
        new_op = top.ReshapeOp(T([in_dim, out_dim]), new_op, loc=L(merger_ln_q + ".reshape"),
                               ip=ip).output
        new_op = self.linear(vit_mlir, merger_mlp0, new_op, [out_dim, out_dim], [in_dim, out_dim])
        new_op = self.activate(vit_mlir, new_op, ActType.GELU, merger_mlp0)
        new_op = self.linear(vit_mlir, merger_mlp2, new_op, [out_dim, self.hidden_size],
                             [in_dim, self.hidden_size])
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()
