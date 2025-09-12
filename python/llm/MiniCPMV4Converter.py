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
import math


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class MiniCPMV4Converter(LlmConverter):

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
        # vision config, e.g. MiniCPMV4
        self.vision_config = config.vision_config
        self.vit_ln_eps = self.vision_config.layer_norm_eps  # 1e-6
        self.image_size = self.vision_config.image_size  # 980
        self.patch_size = self.vision_config.patch_size  # 14
        self.vit_depth = self.vision_config.num_hidden_layers  # 27
        self.vit_embed_dim = self.vision_config.hidden_size  # 1152
        self.vit_num_heads = self.vision_config.num_attention_heads  # 16
        self.vit_head_dim = self.vit_embed_dim // self.vit_num_heads  # 128
        self.vit_intermediate_size = self.vision_config.intermediate_size  # 4304
        self.query_num = self.config.query_num

        # self.position_shape = [3, self.max_input_length]  # mrope position shape

        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.in_channels = self.vision_config.num_channels
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = MINICPMV_INFO
        self.llm_type = LlmType.LLAMA

    # modeling_navit_siglip.py:SiglipEncoderLayer
    def vision_block(self, vit_mlir, idx: int, in_op, mask_op):
        norm1 = f"vpm.encoder.layers.{idx}.layer_norm1"
        norm2 = f"vpm.encoder.layers.{idx}.layer_norm2"
        attn_q = f"vpm.encoder.layers.{idx}.self_attn.q_proj"
        attn_k = f"vpm.encoder.layers.{idx}.self_attn.k_proj"
        attn_v = f"vpm.encoder.layers.{idx}.self_attn.v_proj"
        attn_out = f"vpm.encoder.layers.{idx}.self_attn.out_proj"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        # modeling_glm4v.py:Glm4vVisionAttention
        def vision_attention(in_op):
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, eps=self.vit_ln_eps)
            hidden_shape = [self.num_patches, self.vit_embed_dim]
            weight_shape = [self.vit_embed_dim, self.vit_embed_dim]
            q_op = self.linear(vit_mlir,
                               attn_q,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=True)
            k_op = self.linear(vit_mlir,
                               attn_k,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               attn_v,
                               norm1_op,
                               weight_shape,
                               hidden_shape,
                               force_bias=True)
            qkv_shape = [1, self.num_patches, self.vit_num_heads, self.vit_head_dim]
            q_op = top.ReshapeOp(T(qkv_shape), q_op, loc=L(attn_q + '.reshape'), ip=ip).output
            k_op = top.ReshapeOp(T(qkv_shape), k_op, loc=L(attn_k + '.reshape'), ip=ip).output
            v_op = top.ReshapeOp(T(qkv_shape), v_op, loc=L(attn_v + '.reshape'), ip=ip).output

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
                                     loc=L(f"vpm.encoder.layers.{idx}.fattention"),
                                     ip=ip).output
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  loc=L(f"vpm.encoder.layers.{idx}.fattention.reshape"),
                                  ip=ip).output
            out_op = self.linear(vit_mlir,
                                 attn_out,
                                 fa_op,
                                 weight_shape,
                                 hidden_shape,
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_out + '.add'),
                               ip=ip).output
            return out_op

        # modeling_glm4v.py:Glm4VisionMlp
        def vision_mlp(in_op):
            in_shape = [self.num_patches, self.vit_embed_dim]
            fc1_shape = [self.num_patches, self.vit_intermediate_size]
            fc2_shape = [self.num_patches, self.vit_embed_dim]
            mlp_fc1 = f"vpm.encoder.layers.{idx}.mlp.fc1"
            mlp_fc2 = f"vpm.encoder.layers.{idx}.mlp.fc2"
            ln_op = self.layer_norm(vit_mlir, in_op, norm2, eps=self.vit_ln_eps)
            fc1_op = self.linear(vit_mlir,
                                 mlp_fc1,
                                 ln_op, [self.vit_embed_dim, self.vit_intermediate_size],
                                 fc1_shape,
                                 force_bias=True)

            act_op = self.activate(vit_mlir, fc1_op, self.vision_config.hidden_act, mlp_fc1)
            fc2_op = self.linear(vit_mlir,
                                 mlp_fc2,
                                 act_op, [self.vit_intermediate_size, self.vit_embed_dim],
                                 fc2_shape,
                                 force_bias=True)
            add_op = top.AddOp(T(in_shape), [in_op, fc2_op], loc=L(mlp_fc2 + '.add'), ip=ip).output
            return add_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    def set_patch_embed_weight(self, path: str, weights_dict: dict):
        weight_path = path + ".weight"
        bias_path = path + ".bias"
        has_weight = self.model.is_exist(weight_path)
        has_bias = self.model.is_exist(bias_path)
        if has_weight:
            data = self.model.read(weight_path)
            data = data.reshape(self.vit_embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            weights_dict[weight_path] = data
        if has_bias:
            data = self.model.read(bias_path)
            weights_dict[bias_path] = data

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        # some name
        patch_embed = "vpm.embeddings.patch_embedding"
        position_embed = "vpm.embeddings.position_embedding"
        layers = "vpm.encoder.layers"
        post_ln = "vpm.post_layernorm"

        kv_proj = "resampler.kv_proj"
        attn_out = "resampler.attn.out_proj"
        ln_q = "resampler.ln_q"
        ln_kv = "resampler.ln_kv"
        ln_post = "resampler.ln_post"

        def save_weights():
            weights_dict = dict()
            # patch_embed
            self.set_patch_embed_weight(patch_embed, weights_dict)
            # position_embed
            pos_embed_weight = self.model.read(position_embed + '.weight')
            weights_dict[position_embed + '.weight'] = pos_embed_weight
            # post_layernorm
            self.set_common_weight(post_ln, weights_dict)
            # resample
            self.set_common_weight(ln_q, weights_dict)
            self.set_common_weight(ln_kv, weights_dict)
            self.set_common_weight(ln_post, weights_dict)
            self.set_linear_weight(kv_proj, weights_dict)
            self.set_linear_weight(attn_out, weights_dict)
            query = self.model.read("resampler.query")
            weights_dict["resampler.query"] = query
            proj = self.model.read("resampler.proj")
            weights_dict["resampler.proj.weight"] = proj

            max_size = (70, 70)
            resampler_pos_embed = get_2d_sincos_pos_embed(self.hidden_size, max_size)
            resampler_pos_embed = resampler_pos_embed.reshape(-1, self.hidden_size)
            weights_dict["resampler.pos_embed.weight"] = resampler_pos_embed

            attn_qkv = "resampler.attn.in_proj"
            qkv_weight = self.model.read(attn_qkv + "_weight")
            qkv_weight = np.ascontiguousarray(np.transpose(qkv_weight, (1, 0)))
            dim_size = qkv_weight.shape[-1] // 3
            q_w = np.ascontiguousarray(qkv_weight[:, :dim_size])
            k_w = np.ascontiguousarray(qkv_weight[:, dim_size:2 * dim_size])
            v_w = np.ascontiguousarray(qkv_weight[:, 2 * dim_size:])
            weights_dict["resampler.attn.q_proj.weight"] = q_w
            weights_dict["resampler.attn.k_proj.weight"] = k_w
            weights_dict["resampler.attn.v_proj.weight"] = v_w

            qkv_bias = self.model.read(attn_qkv + "_bias")
            dim_size = qkv_bias.shape[-1] // 3
            q_b = qkv_bias[:dim_size]
            k_b = qkv_bias[dim_size:2 * dim_size]
            v_b = qkv_bias[2 * dim_size:]
            weights_dict["resampler.attn.q_proj.bias"] = q_b
            weights_dict["resampler.attn.k_proj.bias"] = k_b
            weights_dict["resampler.attn.v_proj.bias"] = v_b

            # layers
            for i in range(self.vit_depth):
                self.set_common_weight(layers + f".{i}.layer_norm1", weights_dict)
                self.set_common_weight(layers + f".{i}.layer_norm2", weights_dict)
                self.set_linear_weight(layers + f".{i}.self_attn.q_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.self_attn.k_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.self_attn.v_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.self_attn.out_proj", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(layers + f".{i}.mlp.fc2", weights_dict)
            # save weights
            self.weights.extend(weights_dict.keys())
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.patch_dim]
        pos_ids_shape = [self.num_patches]
        mask_shape = [1, 1, self.num_patches, self.num_patches]
        pos_embed_ids_shape = [self.num_patches]
        resampler_mask_shape = [1, 1, self.query_num, self.num_patches]
        out_shape = [self.query_num, self.hidden_size]
        input_shapes = [
            in_shape, pos_ids_shape, mask_shape, pos_embed_ids_shape, resampler_mask_shape
        ]
        input_types = ['F32', 'INT32', 'F32', 'INT32', 'F32']

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                "vit",
                                Platform.LLM,
                                input_types,
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        # in_shape = [self.num_patches, self.patch_dim]
        # pos_ids_shape = [self.num_patches]
        # input_shapes = [in_shape, pos_ids_shape]
        # input_types = ['F32', 'INT32']
        # out_shape = [self.num_patches, self.vit_embed_dim]

        # vit_mlir = MLIRImporter(input_shapes, [out_shape],
        #                         "vit",
        #                         Platform.LLM,
        #                         input_types,
        #                         weight_file=vit_npz)
        # ip = vit_mlir.insert_point

        # in_shape = [self.num_patches, self.vit_embed_dim]
        # pos_ids_shape = [self.num_patches]
        # mask_shape = [1, 1, self.query_num, self.num_patches]
        # input_shapes = [in_shape, pos_ids_shape, mask_shape]
        # input_types = ['F32', 'INT32', 'F32']
        # out_shape = [self.query_num, self.hidden_size]

        # vit_mlir = MLIRImporter(input_shapes, [out_shape],
        #                         "vit",
        #                         Platform.LLM,
        #                         input_types,
        #                         weight_file=vit_npz)
        # ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_resampler(hidden_state_op, pos_embed_ids_op, mask_op):
            kv_proj = "resampler.kv_proj"
            out_proj = "resampler.attn.out_proj"
            ln_q = "resampler.ln_q"
            ln_kv = "resampler.ln_kv"
            ln_post = "resampler.ln_post"
            proj = "resampler.proj"
            query = "resampler.query"
            pos_embed = "resampler.pos_embed"

            # query
            query_op = vit_mlir.create_weight_op(query, [self.query_num, self.hidden_size])
            q_op = self.layer_norm(vit_mlir, query_op, ln_q, eps=self.vit_ln_eps)

            new_weight = vit_mlir.create_weight_op(pos_embed + '.weight', [4900, self.hidden_size])
            pos_embed_op = top.GatherOp(T([self.num_patches, self.hidden_size]),
                                        new_weight,
                                        pos_embed_ids_op,
                                        axis=0,
                                        loc=L(pos_embed),
                                        ip=ip).output
            # kv proj
            new_op = self.linear(vit_mlir,
                                 kv_proj,
                                 hidden_state_op,
                                 weight_shape=[self.vit_embed_dim, self.hidden_size],
                                 out_shape=[self.num_patches, self.hidden_size],
                                 force_bias=False)
            kv_op = self.layer_norm(vit_mlir, new_op, ln_kv, eps=self.vit_ln_eps)
            output_shape = [self.num_patches, self.hidden_size]
            k_op = top.AddOp(T(output_shape), [kv_op, pos_embed_op], loc=L(ln_kv + ".add"),
                             ip=ip).output
            v_op = kv_op
            # qkv proj
            q_proj = "resampler.attn.q_proj"
            k_proj = "resampler.attn.k_proj"
            v_proj = "resampler.attn.v_proj"

            weight_shape = [self.hidden_size, self.hidden_size]
            output_shape = [self.num_patches, self.hidden_size]
            q_op = self.linear(vit_mlir,
                               q_proj,
                               q_op,
                               weight_shape=weight_shape,
                               out_shape=[self.query_num, self.hidden_size],
                               force_bias=True)
            # q_op = top.MulConstOp(T(q_op.type.shape),
            #                       q_op,
            #                       const_val=128**(-0.5),
            #                       loc=L(q_proj + ".mulconst"),
            #                       ip=ip).output
            k_op = self.linear(vit_mlir,
                               k_proj,
                               k_op,
                               weight_shape=weight_shape,
                               out_shape=output_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               v_proj,
                               v_op,
                               weight_shape=weight_shape,
                               out_shape=output_shape,
                               force_bias=True)

            # attention
            qk_shape = [self.query_num, self.hidden_size]
            fa_op = top.FAttentionOp(T(qk_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     mask_op,
                                     vit_mlir.none_op,
                                     scale=128**-0.5,
                                     batch=1,
                                     q_head=self.hidden_size // 128,
                                     kv_head=self.hidden_size // 128,
                                     dim=128,
                                     mq=self.query_num,
                                     mk=self.num_patches,
                                     loc=L("resampler.fattention"),
                                     ip=ip).output
            # out proj
            new_op = self.linear(vit_mlir,
                                 out_proj,
                                 fa_op,
                                 weight_shape=[self.hidden_size, self.hidden_size],
                                 out_shape=[self.query_num, self.hidden_size],
                                 force_bias=True)

            # proj
            new_op = self.layer_norm(vit_mlir, new_op, ln_post, eps=self.vit_ln_eps)
            new_op = self.linear(vit_mlir,
                                 proj,
                                 new_op,
                                 weight_shape=[self.hidden_size, self.hidden_size],
                                 out_shape=[self.query_num, self.hidden_size],
                                 force_bias=False)

            return new_op

        def vision_embedding(pixel_values_op, position_ids_op):
            output_shape = [self.num_patches, self.vit_embed_dim]
            patch_embed_op = self.linear(vit_mlir,
                                         patch_embed,
                                         pixel_values_op,
                                         weight_shape=[self.patch_dim, self.vit_embed_dim],
                                         out_shape=output_shape,
                                         force_bias=True)
            new_weight = vit_mlir.create_weight_op(position_embed + '.weight',
                                                   [self.num_patches, self.vit_embed_dim])
            pos_embedding = top.GatherOp(T([self.num_patches, self.vit_embed_dim]),
                                         new_weight,
                                         position_ids_op,
                                         axis=0,
                                         loc=L(position_embed),
                                         ip=ip).output

            new_op = top.AddOp(T(output_shape), [patch_embed_op, pos_embedding],
                               loc=L(patch_embed + '.add'),
                               ip=ip).output
            return new_op

        save_weights()

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        in1_op = vit_mlir.create_input_op(L('pos_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('attn_mask'), 2)
        in3_op = vit_mlir.create_input_op(L('pos_embed_ids'), 3)
        in4_op = vit_mlir.create_input_op(L('resampler_attn_mask'), 4)

        new_op = vision_embedding(in0_op, in1_op)

        # layers
        for idx in range(self.vit_depth):
            new_op = self.vision_block(vit_mlir, idx, new_op, in2_op)

        # resampler
        new_op = vision_resampler(new_op, in3_op, in4_op)
        # new_op = vision_resampler(in0_op, in1_op, in2_op)

        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
