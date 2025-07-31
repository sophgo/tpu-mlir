# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
import torch.nn as nn
import torch.nn.functional as F


# copied from modeling_glm4v.py
class Glm4vVisionEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        # self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, lengths, image_shapes, h_coords, w_coords, pos_embed_weight) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            # embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        # pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0,
                                            hidden_size,
                                            device=device,
                                            dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (pos_embed_weight.view(orig_size, orig_size, hidden_size).permute(
                2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32))

            # Calculate target dimensions for each patch
            target_h = torch.cat([
                image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))
            ]).to(device=device, dtype=torch.float32)
            target_w = torch.cat([
                image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))
            ]).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(pos_embed_2d,
                                                    grid,
                                                    mode="bicubic",
                                                    align_corners=False,
                                                    padding_mode="border")

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            # adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype)
            adapted_pos_embed = adapted_pos_embed_fp32.to(torch.bfloat16).to(torch.float32)

        # Add adapted position encoding to embeddings
        # embeddings = embeddings + adapted_pos_embed
        return adapted_pos_embed


class GLM4VConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_pixels = args.max_pixels
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        if args.max_pixels % (28 * 28) != 0:
            raise RuntimeError(
                "max_pixels is not a multiple of 28*28, please set max_pixels to a value that is a multiple of 28*28."
            )
        self.max_shape = args.max_shape
        if args.max_shape == None:
            raise RuntimeError("max_shape is None, please set max_pixels to value like 672,896")
        if args.max_shape[0] % 28 != 0 or args.max_shape[1] % 28 != 0:
            raise RuntimeError(
                "shape[0] and shape[1] is not a multiple of 28, please set each dim size to a value that is a multiple of 28"
            )

        self.do_vit = True
        # vision config, e.g. GLMv4.1-9B
        self.vision_config = config.vision_config
        self.rms_eps = self.config.rms_norm_eps  # 1e-5
        self.image_size = self.vision_config.image_size  # 336
        self.patch_size = self.vision_config.patch_size  # 14
        self.temporal_patch_size = self.vision_config.temporal_patch_size
        self.spatial_merge_size = self.vision_config.spatial_merge_size  #2
        self.vit_depth = self.vision_config.depth  # 24
        self.vit_embed_dim = self.vision_config.hidden_size  # 1536
        self.vit_hidden_size = self.vision_config.out_hidden_size  # 4096
        self.vit_num_heads = self.vision_config.num_heads  # 12
        self.vit_head_dim = self.vit_embed_dim // self.vit_num_heads  # 128
        self.vit_intermediate_size = self.vision_config.intermediate_size  # 13696
        self.position_shape = [3, self.max_input_length]  # mrope position shape

        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.in_channels = self.vision_config.in_channels
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size

        self.embeddings = Glm4vVisionEmbeddings(self.vision_config)
        self.grid_thw = [[
            1, self.max_shape[0] // self.patch_size, self.max_shape[1] // self.patch_size
        ]]

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GLM4V_INFO
        self.llm_type = LlmType.GLM4V

    def rot_pos(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    # q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    # k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)
    def rotary_pos_half(self, mlir_gen, in_op, cos_op, sin_op, out_name: str):
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
    def rotary_embedding(self):
        from transformers.models.glm4v.modeling_glm4v import Glm4vTextRotaryEmbedding
        rotary_embed = Glm4vTextRotaryEmbedding(self.config)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(
            1, 1, self.seq_length).expand(3, 1, self.seq_length)
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        cos, sin = rotary_embed(x, position_ids)
        cos = cos[0].reshape(self.seq_length, 1, -1)
        sin = sin[0].reshape(self.seq_length, 1, -1)
        partial_rotary_factor = getattr(self.config, 'partial_rotary_factor', 1)
        assert (cos.shape[-1] == self.head_dim * partial_rotary_factor)
        assert (sin.shape[-1] == self.head_dim * partial_rotary_factor)
        # half
        cos = cos[:, :, :cos.shape[-1] // 2]
        sin = sin[:, :, :sin.shape[-1] // 2]
        self.rot_dim = cos.shape[-1]
        # cos.repeat_interleave(2, dim=-1)
        # sin.repeat_interleave(2, dim=-1)
        return cos.numpy(), sin.numpy()  # [seq, 1, 64]

    def mrope(self, mlir_gen, in_op, name: str):
        # in_op weight has been repeat_interleaved
        mrope_section = getattr(self.config.rope_scaling, 'mrope_section', [8, 12, 12])
        # mrope_section = [x * 2 for x in mrope_section]
        in_shape = in_op.type.shape
        t_dim, h_dim, w_dim = mrope_section  # 16, 24, 24
        dim = in_shape[1]
        # slice t_rope: input[0, dim, 1, :t_dim]
        # slice h_rope: input[1, dim, 1, t_dim:t_dim+h_dim]
        # slice w_rope: input[2, dim, 1, t_dim+h_dim:]
        # cos_op = [1, dim 1, t_dim] + [1, dim, 1, h_dim] + [1, dim, 1, w_dim]
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

        # [fix] don't use tile, it is repeat_interleave, not repeat
        # repeat_interleave: [0, 1, 2, 3] -> [0, 0, 1, 1, 2, 2, 3, 3]
        # repeat: [0, 1, 2, 3] -> [0, 1, 2, 3, 0, 1, 2, 3]
        # tile_op = top.TileOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
        #                      concat_op,
        #                      tile=[1, 1, 1, 2],
        #                      loc=self.get_loc(name + ".tile", mlir_gen),
        #                      ip=mlir_gen.insert_point).output
        return concat_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        dim = pos_op.type.shape[-1]
        rot_dim = self.rot_dim
        # cos MROPE
        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight", [self.seq_length, 1, rot_dim])
        cos_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, rot_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_cos, mlir_gen),
                              ip=mlir_gen.insert_point).output
        cos_op = self.mrope(mlir_gen, cos_op, rotary_cos)

        # sin MROPE
        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight", [self.seq_length, 1, rot_dim])
        sin_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, rot_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_sin, mlir_gen),
                              ip=mlir_gen.insert_point).output
        sin_op = self.mrope(mlir_gen, sin_op, rotary_sin)

        # ===== q_proj rotary ========
        q_op = self.rotary_pos_half(mlir_gen, q_op, cos_op, sin_op, "q_proj")
        # ===== k_proj rotary ========
        k_op = self.rotary_pos_half(mlir_gen, k_op, cos_op, sin_op, "k_cache")

        return q_op, k_op

    def vision_rotary(self):
        from transformers.models.glm4v.modeling_glm4v import Glm4vVisionRotaryEmbedding
        rotary_embed = Glm4vVisionRotaryEmbedding(self.vit_head_dim // 2)
        freqs = rotary_embed(self.num_patches)
        return freqs.cos().numpy(), freqs.sin().numpy()

    # modeling_glm4v.py:Glm4vVisionBlock
    def vision_block(self, vit_mlir, idx: int, in_op, cos_op, sin_op, mask_op):
        norm1 = f"model.visual.blocks.{idx}.norm1"
        attn_q = f"model.visual.blocks.{idx}.attn.q"
        attn_k = f"model.visual.blocks.{idx}.attn.k"
        attn_v = f"model.visual.blocks.{idx}.attn.v"
        attn_proj = f"model.visual.blocks.{idx}.attn.proj"
        norm2 = f"model.visual.blocks.{idx}.norm2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        # modeling_glm4v.py:Glm4vVisionAttention
        def vision_attention(in_op):
            norm1_op = self.rms_norm(vit_mlir, in_op, norm1, eps=self.rms_eps)
            hidden_shape = [self.num_patches, self.vit_embed_dim]
            weight_shape = [self.vit_embed_dim, self.vit_embed_dim]
            q_op = self.linear(vit_mlir,
                               attn_q,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=False)
            k_op = self.linear(vit_mlir,
                               attn_k,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=False)
            v_op = self.linear(vit_mlir,
                               attn_v,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=False)
            qkv_shape = [1, self.num_patches, self.vit_num_heads, self.vit_head_dim]
            q_op = top.ReshapeOp(T(qkv_shape), q_op, loc=L(attn_q + '.reshape'), ip=ip).output
            k_op = top.ReshapeOp(T(qkv_shape), k_op, loc=L(attn_k + '.reshape'), ip=ip).output
            v_op = top.ReshapeOp(T(qkv_shape), v_op, loc=L(attn_v + '.reshape'), ip=ip).output

            q_op = self.rotary_pos(vit_mlir, q_op, cos_op, sin_op, attn_q + '.rotary')
            k_op = self.rotary_pos(vit_mlir, k_op, cos_op, sin_op, attn_k + '.rotary')
            fa_op = top.FAttentionOp(T(qkv_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     mask_op,
                                     vit_mlir.none_op,
                                     scale=self.vit_head_dim**-0.5,
                                     batch=1,
                                     q_head=self.vit_num_heads,
                                     kv_head=self.vit_num_heads,
                                     dim=self.vit_head_dim,
                                     mq=self.num_patches,
                                     mk=self.num_patches,
                                     loc=L(f"visual.blocks.{idx}.fattention"),
                                     ip=ip).output
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  loc=L(f"visual.blocks.{idx}.fattention.reshape"),
                                  ip=ip).output
            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 fa_op,
                                 weight_shape,
                                 hidden_shape,
                                 force_bias=False)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + '.add'),
                               ip=ip).output
            return out_op

        # modeling_glm4v.py:Glm4VisionMlp
        def vision_mlp(in_op):
            in_shape = [self.num_patches, self.vit_embed_dim]
            up_shape = [self.num_patches, self.vit_hidden_size]
            mlp_fc1 = f"model.visual.blocks.{idx}.mlp.gate_proj"
            mlp_fc2 = f"model.visual.blocks.{idx}.mlp.up_proj"
            mlp_fc3 = f"model.visual.blocks.{idx}.mlp.down_proj"
            rms_op = self.rms_norm(vit_mlir, in_op, norm2, eps=self.rms_eps)
            gate_op = self.linear(vit_mlir, mlp_fc1, rms_op,
                                  [self.vit_embed_dim, self.vit_hidden_size], up_shape)

            act_op = self.activate(vit_mlir, gate_op, self.config.hidden_act, mlp_fc1)
            up_op = self.linear(vit_mlir, mlp_fc2, rms_op,
                                [self.vit_embed_dim, self.vit_hidden_size], up_shape)
            mul_op = top.MulOp(T(up_shape), [act_op, up_op], loc=L(mlp_fc2 + '.mul'), ip=ip).output

            down_op = self.linear(vit_mlir, mlp_fc3, mul_op,
                                  [self.vit_hidden_size, self.vit_embed_dim], in_shape)
            add_op = top.AddOp(T(in_shape), [in_op, down_op], loc=L(mlp_fc3 + '.add'), ip=ip).output
            return add_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    def set_qkv_weight(self, top_path: str, weight_dict: dict):
        qweight_path = top_path + '.attn.qkv.qweight'
        is_quant = self.quant_mode is not None and self.model.is_exist(qweight_path)
        if not is_quant:
            weight_path = top_path + '.attn.qkv.weight'
            q_weight_path = top_path + '.attn.q.weight'
            k_weight_path = top_path + '.attn.k.weight'
            v_weight_path = top_path + '.attn.v.weight'
            if self.model.is_exist(weight_path):
                qkv_weight = self.model.read(weight_path)
                qkv_weight = np.ascontiguousarray(np.transpose(qkv_weight, (1, 0)))
                q_w = np.ascontiguousarray(qkv_weight[:, :self.vit_embed_dim])
                k_w = np.ascontiguousarray(qkv_weight[:, self.vit_embed_dim:2 * self.vit_embed_dim])
                v_w = np.ascontiguousarray(qkv_weight[:, 2 * self.vit_embed_dim:])
                weight_dict[q_weight_path] = q_w
                weight_dict[k_weight_path] = k_w
                weight_dict[v_weight_path] = v_w
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        else:
            qweight_path = top_path + '.attn.qkv.qweight'
            scale_path = top_path + '.attn.qkv.scales'
            zp_path = top_path + '.attn.qkv.qzeros'
            if self.model.is_exist(qweight_path):
                q_qweight_path = top_path + '.attn.q.qweight'
                k_qweight_path = top_path + '.attn.k.qweight'
                v_qweight_path = top_path + '.attn.v.qweight'
                q_scale_path = top_path + '.attn.q.scales'
                k_scale_path = top_path + '.attn.k.scales'
                v_scale_path = top_path + '.attn.v.scales'
                q_zp_path = top_path + '.attn.q.qzeros'
                k_zp_path = top_path + '.attn.k.qzeros'
                v_zp_path = top_path + '.attn.v.qzeros'
                qweight_data = self.model.read(qweight_path)
                scale_data = self.model.read(scale_path)
                zp_data = self.model.read(zp_path)
                _, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                    qweight_data, zp_data, self.quant_bits, self.quant_mode)

                last_dim_size = pack_int8_weights.shape[-1]
                pack_int8_weights = np.ascontiguousarray(np.transpose(pack_int8_weights, (1, 0)))
                unpacked_zeros = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
                scale_data = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))

                weight_dict[q_qweight_path] = np.ascontiguousarray(
                    pack_int8_weights[:last_dim_size // 3, :])
                weight_dict[q_scale_path] = np.ascontiguousarray(scale_data[:last_dim_size // 3, :])
                weight_dict[q_zp_path] = np.ascontiguousarray(unpacked_zeros[:last_dim_size //
                                                                             3, :])
                weight_dict[k_qweight_path] = np.ascontiguousarray(
                    pack_int8_weights[last_dim_size // 3:last_dim_size // 3 * 2, :])
                weight_dict[k_scale_path] = np.ascontiguousarray(
                    scale_data[last_dim_size // 3:last_dim_size // 3 * 2, :])
                weight_dict[k_zp_path] = np.ascontiguousarray(
                    unpacked_zeros[last_dim_size // 3:last_dim_size // 3 * 2, :])
                weight_dict[v_qweight_path] = np.ascontiguousarray(
                    pack_int8_weights[last_dim_size // 3 * 2:, :])
                weight_dict[v_scale_path] = np.ascontiguousarray(scale_data[last_dim_size // 3 *
                                                                            2:, :])
                weight_dict[v_zp_path] = np.ascontiguousarray(unpacked_zeros[last_dim_size // 3 *
                                                                             2:, :])
            else:
                raise RuntimeError("Can't find key: {}".format(qweight_path))

        bias_path = top_path + '.attn.qkv.bias'
        if self.model.is_exist(bias_path):
            q_bias_path = top_path + '.attn.q.bias'
            k_bias_path = top_path + '.attn.k.bias'
            v_bias_path = top_path + '.attn.v.bias'
            bias = self.model.read(bias_path)
            last_dim_size = bias.shape[-1]
            q_b = bias[:last_dim_size // 3]
            k_b = bias[last_dim_size // 3:2 * (last_dim_size // 3)]
            v_b = bias[2 * (last_dim_size // 3):]
            weight_dict[q_bias_path] = q_b
            weight_dict[k_bias_path] = k_b
            weight_dict[v_bias_path] = v_b

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        # some name
        downsample = "model.visual.downsample"
        position_embed = "model.visual.embeddings.position_embedding"
        patch_embed = "model.visual.patch_embed.proj"
        post_conv_ln = "model.visual.post_conv_layernorm"
        post_ln = "model.visual.post_layernorm"
        layers = "model.visual.blocks"
        merger = "model.visual.merger"
        rotary_cos = "model.visual.rotary.cos"
        rotary_sin = "model.visual.rotary.sin"

        def save_weights():
            # rotary_pos_emb
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + '.weight': cos,
                rotary_sin + '.weight': sin,
            }
            # patch_embed
            data = self.model.read(patch_embed + '.weight')
            data = data.reshape(self.vit_embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            weights_dict[patch_embed + '.weight'] = data
            data = self.model.read(patch_embed + '.bias')
            weights_dict[patch_embed + '.bias'] = data
            # position_embed
            pos_embed_weight = self.model.read(position_embed + '.weight')
            # vit_num_token = np.prod(self.grid_thw[0]).item()
            # grid_thw = torch.tensor(self.grid_thw)
            # pos_ids = self.rot_pos(grid_thw)
            # pos_embed_weight = torch.tensor(pos_embed_weight)
            # pos_embeddings = self.embeddings([vit_num_token], grid_thw, pos_ids[:, 0],
            #                                  pos_ids[:, 1], pos_embed_weight)
            # weights_dict[position_embed + '.weight'] = pos_embeddings.detach().numpy()
            shape_size = self.image_size // self.patch_size
            pos_embed_weight = np.transpose(pos_embed_weight,
                                            (1, 0)).reshape(1, self.vit_embed_dim, shape_size,
                                                            shape_size)
            pos_embed_weight = np.ascontiguousarray(pos_embed_weight)
            weights_dict[position_embed + '.weight'] = pos_embed_weight
            np.save("pos_embed_weight.npy", pos_embed_weight)
            # merger
            self.set_linear_weight(merger + ".proj", weights_dict)
            self.set_common_weight(merger + ".post_projection_norm", weights_dict)
            self.set_linear_weight(merger + ".gate_proj", weights_dict)
            self.set_linear_weight(merger + ".up_proj", weights_dict)
            self.set_linear_weight(merger + ".down_proj", weights_dict)
            # post_conv_layernorm
            self.set_common_weight(post_conv_ln, weights_dict)
            # downsample
            self.set_common_weight(downsample, weights_dict)
            # post_layernorm
            self.set_common_weight(post_ln, weights_dict)
            # blocks
            for i in range(self.vit_depth):
                self.set_common_weight(layers + f".{i}.norm1", weights_dict)
                self.set_common_weight(layers + f".{i}.norm2", weights_dict)
                self.set_linear_weight(layers + f".{i}.attn.proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.down_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.gate_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.up_proj", weights_dict)
                # split qkv
                self.set_qkv_weight(f"{layers}.{i}", weights_dict)
            # save weights
            self.weights.extend(weights_dict.keys())
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.patch_dim]
        position_shape = [self.num_patches, 2]
        mask_shape = [1, 1, self.num_patches, self.num_patches]
        # embed_grid_shape = [1, self.num_patches, 1, 2]
        pos_embed_shape = [self.num_patches, self.vit_embed_dim]
        out_dim = self.num_patches // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.hidden_size]
        input_shapes = [in_shape, position_shape, mask_shape, pos_embed_shape]
        input_types = ['F32', 'INT32', 'F32', 'F32']

        # vit_mlir = MLIRImporter(input_shapes, [[2944, 1536]],
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

        def vision_merger(in_op):
            mlp_fc0 = "model.visual.merger.proj"
            mlp_fc1 = "model.visual.merger.gate_proj"
            mlp_fc2 = "model.visual.merger.up_proj"
            mlp_fc3 = "model.visual.merger.down_proj"
            post_norm = "model.visual.merger.post_projection_norm"

            in_dim = self.num_patches // (self.spatial_merge_size**2)
            in_shape = [in_dim, self.vit_hidden_size]
            proj_op = self.linear(vit_mlir, mlp_fc0, in_op,
                                  [self.vit_hidden_size, self.vit_hidden_size], in_shape)

            up_shape = [in_dim, self.vit_intermediate_size]
            ln_op = self.layer_norm(vit_mlir, proj_op, post_norm, eps=1e-6)
            act0_op = self.activate(vit_mlir, ln_op, ActType.GELU, post_norm)
            gate_op = self.linear(vit_mlir, mlp_fc1, act0_op,
                                  [self.vit_hidden_size, self.vit_intermediate_size], up_shape)
            act1_op = self.activate(vit_mlir, gate_op, self.config.hidden_act, mlp_fc1)
            up_op = self.linear(vit_mlir, mlp_fc2, act0_op,
                                [self.vit_hidden_size, self.vit_intermediate_size], up_shape)
            mul_op = top.MulOp(T(up_shape), [act1_op, up_op], loc=L(mlp_fc2 + '.mul'), ip=ip).output

            down_op = self.linear(vit_mlir, mlp_fc3, mul_op,
                                  [self.vit_intermediate_size, self.vit_hidden_size], in_shape)
            return down_op

        save_weights()

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        in1_op = vit_mlir.create_input_op(L('position_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('full_attn_mask'), 2)
        # in3_op = vit_mlir.create_input_op(L('embed_grid'), 3)
        in3_op = vit_mlir.create_input_op(L('pos_embed'), 3)

        new_weight = vit_mlir.create_weight_op(patch_embed + ".weight",
                                               [self.patch_dim, self.vit_embed_dim])
        new_bias = vit_mlir.create_weight_op(patch_embed + ".bias", [self.vit_embed_dim])
        new_op = top.MatMulOp(T([self.num_patches, self.vit_embed_dim]),
                              in0_op,
                              new_weight,
                              new_bias,
                              loc=L(patch_embed),
                              ip=ip).output
        new_op = self.rms_norm(vit_mlir, new_op, post_conv_ln, eps=self.rms_eps)

        output_shape = [self.num_patches, self.vit_embed_dim]
        new_op = top.AddOp(T(output_shape), [new_op, in3_op], loc=L(position_embed + '.add'),
                           ip=ip).output
        # Add embed_position
        # shape_size = self.image_size // self.patch_size
        # weight_shape = [1, self.vit_embed_dim, shape_size, shape_size]
        # pos_embed_weight = vit_mlir.create_weight_op(position_embed + ".weight", weight_shape)
        # output_shape = [1, self.vit_embed_dim, self.num_patches, 1]
        # pos_emb_op = top.GridSamplerOp(
        #     T(output_shape),
        #     pos_embed_weight,
        #     in3_op,
        #     mode=0,  # "bilinear"
        #     padding_mode=1,  # "border"
        #     align_corners=False,
        #     loc=L(position_embed),
        #     ip=ip).output
        # output_shape = [1, 1, self.num_patches, self.vit_embed_dim]
        # pos_emb_op = top.PermuteOp(T(output_shape),
        #                            pos_emb_op,
        #                            order=[0, 3, 2, 1],
        #                            loc=L(position_embed + '.permute'),
        #                            ip=ip).output
        # output_shape = [self.num_patches, self.vit_embed_dim]
        # pos_emb_op = top.ReshapeOp(T(output_shape),
        #                            pos_emb_op,
        #                            shape=output_shape,
        #                            loc=L(position_embed + '.permute.reshape'),
        #                            ip=ip).output
        # # input_shape = new_op.type.shape
        # # new_weight = vit_mlir.create_weight_op(position_embed + ".weight", input_shape)
        # new_op = top.AddOp(T(output_shape), [new_op, pos_emb_op],
        #                    loc=L(position_embed + '.add'),
        #                    ip=ip).output

        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight",
                                               [self.num_patches, self.vit_head_dim // 4])
        cos_op = top.GatherOp(T([self.num_patches, 2, self.vit_head_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_cos),
                              ip=ip).output
        cos_op = top.ReshapeOp(T([1, self.num_patches, 1, self.vit_head_dim // 2]),
                               cos_op,
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        cos_op = top.TileOp(T([1, self.num_patches, 1, self.vit_head_dim]),
                            cos_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_cos + ".tile"),
                            ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight",
                                               [self.num_patches, self.vit_head_dim // 4])
        sin_op = top.GatherOp(T([self.num_patches, 2, self.vit_head_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_sin),
                              ip=ip).output
        sin_op = top.ReshapeOp(T([1, self.num_patches, 1, self.vit_head_dim // 2]),
                               sin_op,
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        sin_op = top.TileOp(T([1, self.num_patches, 1, self.vit_head_dim]),
                            sin_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_sin + ".tile"),
                            ip=ip).output
        # block
        for idx in range(self.vit_depth):
            new_op = self.vision_block(vit_mlir, idx, new_op, cos_op, sin_op, in2_op)

        # before merger
        new_op = self.rms_norm(vit_mlir, new_op, post_ln, eps=self.rms_eps)
        in_dim = self.num_patches // (self.spatial_merge_size**2)
        out_shape = [in_dim, self.spatial_merge_size, self.spatial_merge_size, self.vit_embed_dim]
        new_op = top.ReshapeOp(T(out_shape), new_op, loc=L(post_ln + '.reshape'), ip=ip).output
        out_shape = [in_dim, self.vit_embed_dim, self.spatial_merge_size, self.spatial_merge_size]
        new_op = top.PermuteOp(T(out_shape),
                               new_op,
                               order=[0, 3, 1, 2],
                               loc=L(post_ln + '.reshape.permute'),
                               ip=ip).output
        out_shape = [in_dim, self.vit_hidden_size, 1, 1]
        new_weight = vit_mlir.create_weight_op(downsample + '.weight', [
            self.vit_hidden_size, self.vit_embed_dim, self.spatial_merge_size,
            self.spatial_merge_size
        ])
        new_bias = vit_mlir.create_weight_op(downsample + '.bias', [self.vit_hidden_size])
        new_op = top.ConvOp(T(out_shape),
                            new_op,
                            new_weight,
                            new_bias,
                            kernel_shape=[self.spatial_merge_size, self.spatial_merge_size],
                            strides=[self.spatial_merge_size, self.spatial_merge_size],
                            pads=[0, 0, 0, 0],
                            dilations=[1, 1],
                            loc=L(downsample),
                            ip=ip).output
        out_shape = [in_dim, self.vit_hidden_size]
        new_op = top.ReshapeOp(T(out_shape), new_op, loc=L(downsample + '.reshape'), ip=ip).output
        # merger
        new_op = vision_merger(new_op)

        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)

    def set_gate_up_weight(self, top_path: str, weight_dict: dict):
        mlp_gate_up = top_path + self.model_info.weights[LlmList.MLP_GATE_UP]
        mlp_gate = top_path + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = top_path + self.model_info.weights[LlmList.MLP_UP]
        is_quant = self.quant_mode is not None and self.model.is_exist(mlp_gate_up + '.qweight')
        bias_path = mlp_gate_up + '.bias'
        if not is_quant:
            weight_path = mlp_gate_up + '.weight'
            if self.model.is_exist(weight_path):
                weight = self.model.read(weight_path)
                weight = np.ascontiguousarray(np.transpose(weight, (1, 0)))
                gate_w = np.ascontiguousarray(weight[:, :self.intermediate_size])
                up_w = np.ascontiguousarray(weight[:, self.intermediate_size:])
                weight_dict[mlp_gate + '.weight'] = gate_w
                weight_dict[mlp_up + '.weight'] = up_w
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        else:
            qweight_path = mlp_gate_up + '.qweight'
            scale_path = mlp_gate_up + '.scales'
            zp_path = mlp_gate_up + '.qzeros'
            if self.model.is_exist(qweight_path):
                gate_qweight_path = mlp_gate + '.qweight'
                gate_scale_path = mlp_gate + '.scales'
                gate_zp_path = mlp_gate + '.qzeros'
                up_qweight_path = mlp_up + '.qweight'
                up_scale_path = mlp_up + '.scales'
                up_zp_path = mlp_up + '.qzeros'
                qweight_data = self.model.read(qweight_path)
                scale_data = self.model.read(scale_path)
                zp_data = self.model.read(zp_path)
                _, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                    qweight_data, zp_data, self.quant_bits, self.quant_mode)

                last_dim_size = pack_int8_weights.shape[-1]
                pack_int8_weights = np.ascontiguousarray(np.transpose(pack_int8_weights, (1, 0)))
                unpacked_zeros = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
                scale_data = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))

                # gate_int8_weights = pack_int8_weights[:last_dim_size//2]
                # up_int8_weights = pack_int8_weights[last_dim_size//2:]
                weight_dict[gate_qweight_path] = np.ascontiguousarray(
                    pack_int8_weights[:last_dim_size // 2])
                weight_dict[gate_scale_path] = np.ascontiguousarray(scale_data[:last_dim_size // 2])
                weight_dict[gate_zp_path] = np.ascontiguousarray(unpacked_zeros[:last_dim_size //
                                                                                2])
                weight_dict[up_qweight_path] = np.ascontiguousarray(
                    pack_int8_weights[last_dim_size // 2:])
                weight_dict[up_scale_path] = np.ascontiguousarray(scale_data[last_dim_size // 2:])
                weight_dict[up_zp_path] = np.ascontiguousarray(unpacked_zeros[last_dim_size // 2:])
            else:
                raise RuntimeError("Can't find key: {}".format(qweight_path))

        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)

    @override
    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
        # q_norm = TOP_PATH + self.model_info.weights[LlmList.Q_NORM]
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
        # k_norm = TOP_PATH + self.model_info.weights[LlmList.K_NORM]
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        post_self_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_SELF_ATTN_LN]
        # mlp_gate_up = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE_UP]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        post_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.POST_MLP_LN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.do_lmhead_merge and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"
        weight_file = f"block_{idx}_top_weights.npz"

        # save weight
        def save_weights():
            weight_dict = {
                rotary_cos + ".weight": self.cos,
                rotary_sin + ".weight": self.sin,
            }
            self.set_common_weight(input_ln, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(post_mlp_ln, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(post_attn_ln, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(post_self_attn_ln, weight_dict, WeightType.RMS_NORM)
            self.set_linear_weight(q_proj, weight_dict)
            self.set_linear_weight(k_proj, weight_dict)
            self.set_linear_weight(v_proj, weight_dict)
            self.set_linear_weight(o_proj, weight_dict)
            self.set_linear_weight(mlp_down, weight_dict)
            if do_norm:
                self.set_common_weight(norm, weight_dict, WeightType.RMS_NORM)

            # split mlp gate_up_proj to gate_proj and up_proj
            self.set_gate_up_weight(TOP_PATH, weight_dict)

            # save weights
            self.weights.extend(list(weight_dict.keys()))
            np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            len = input_shape[1]
            new_op = in_op
            new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)
            gate_op = self.linear(mlir_gen, mlp_gate, new_op,
                                  [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size])
            act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
            up_op = self.linear(mlir_gen, mlp_up, new_op,
                                [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size])
            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen, mlp_down, new_op,
                                  [self.intermediate_size, self.hidden_size], input_shape)
            down_op = self.rms_norm(mlir_gen, down_op, post_mlp_ln)
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
            input_len = self.max_input_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            mask_shape = [1, 1, input_len, input_len]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]
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

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim],
                               [1, input_len, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])

            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output

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
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_self_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
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
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim], [1, 1, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in3_op, k_op],
                                axis=1,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in4_op, v_op],
                                axis=1,
                                loc=L(v_proj + ".concat"),
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
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_self_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_block_with_kv():
            # Generate block with kv cache related operations
            name = f"block_{idx}"
            input_len = self.max_input_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            max_kv_len = self.max_prefill_kv_length + self.max_input_length
            mask_shape = [1, 1, self.max_input_length, max_kv_len]
            history_shape = [1, self.max_prefill_kv_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]

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
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, q_dim],
                               [1, input_len, q_dim])
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, input_len, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in3_op, k_op],
                                axis=1,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in4_op, v_op],
                                axis=1,
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
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, o_proj, fa_op, [q_dim, self.hidden_size], input_shape)
            o_op = self.rms_norm(block_mlir, o_op, post_self_attn_ln)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        save_weights()
        if self.use_block_with_kv:
            gen_block_with_kv()
        else:
            gen_block()
        gen_block_cache()
