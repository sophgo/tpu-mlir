# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
import math
import numpy as np
import torch


class SigLIPRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (self.theta**(torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PaddleOCRVLConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_pixels = args.max_pixels  #784*784
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        self.max_shape = args.max_shape
        self.do_vit = True
        # vision config
        self.init_vconfig()
        self.vit_path = "visual"
        self.image_grid_thw = []

    def init_vconfig(self):
        self.out_hidden_size = self.config.hidden_size
        self.vconfig = self.config.vision_config
        self.vit_ln_eps = self.vconfig.layer_norm_eps  # 1e-6
        self.patch_size = self.vconfig.patch_size  # 14
        self.spatial_merge_size = self.vconfig.spatial_merge_size
        self.in_channels = self.vconfig.num_channels
        self.depth = self.vconfig.num_hidden_layers
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.embed_dim = self.vconfig.hidden_size
        self.vnum_heads = self.vconfig.num_attention_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [3, self.max_input_length]
        self.image_size = self.config.vision_config.image_size

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_type = LlmType.QWEN2

    def vision_rotary(self):
        head_dim = self.embed_dim // self.vnum_heads
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)
        split_hids = list()
        split_wids = list()
        for t, h, w in self.image_grid_thw:
            image_pids = torch.arange(t * h * w) % (h * w)
            sample_hids = image_pids // w
            sample_wids = image_pids % w
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)
        width_position_ids = torch.concat(split_wids, dim=0)
        height_position_ids = torch.concat(split_hids, dim=0)
        pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
        max_grid_size = pids.max() + 1
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        rope_emb = rope_emb_max_grid[pids].flatten(1)
        rope_emb = rope_emb.repeat(1, 2)
        return rope_emb.cos().numpy(), rope_emb.sin().numpy()

    def vision_block(self, vit_mlir, id: int, in_op, cos_op, sin_op):
        norm1 = f"{self.vit_path}.vision_model.encoder.layers.{id}.layer_norm1"
        eager_attn = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.eager_attn"
        attn_q = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.q_proj"
        attn_k = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.k_proj"
        attn_v = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.v_proj"
        attn_proj = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.out_proj"
        norm2 = f"{self.vit_path}.vision_model.encoder.layers.{id}.layer_norm2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_attention(in_op):
            hidden_shape = [1, self.num_patches, self.embed_dim]
            in_op = top.ReshapeOp(T(hidden_shape),
                                  in_op,
                                  shape=hidden_shape,
                                  loc=L(norm1 + ".reshape"),
                                  ip=ip).output
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, self.vit_ln_eps)
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
            q_op = self.rotary_pos(vit_mlir, q_op, cos_op, sin_op, attn_q + ".rotary")
            k_op = self.rotary_pos(vit_mlir, k_op, cos_op, sin_op, attn_k + ".rotary")
            q_op = top.PermuteOp(T([1, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 q_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_q + ".permute"),
                                 ip=ip).output
            k_op = top.PermuteOp(T([1, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 k_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_k + ".permute"),
                                 ip=ip).output
            v_op = top.PermuteOp(T([1, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 v_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_v + ".permute"),
                                 ip=ip).output
            # eager_attention
            k_op = top.PermuteOp(T([1, self.vnum_heads, self.vhead_dim, self.num_patches]),
                                 k_op,
                                 order=[0, 1, 3, 2],
                                 loc=L(attn_k + ".permute"),
                                 ip=ip).output
            attn_weights = top.MatMulOp(T([1, self.vnum_heads, self.num_patches, self.num_patches]),
                                        q_op,
                                        k_op,
                                        vit_mlir.none_op,
                                        do_relu=False,
                                        is_lora=False,
                                        loc=L(eager_attn + ".matmul"),
                                        ip=ip).output
            scaling_value = float(self.vhead_dim**-0.5)
            attn_weights = top.MulConstOp(T(
                [1, self.vnum_heads, self.num_patches, self.num_patches]),
                                          attn_weights,
                                          const_val=scaling_value,
                                          loc=L(eager_attn + ".scale"),
                                          ip=ip).output
            attn_weights = top.SoftmaxOp(T([1, self.vnum_heads, self.num_patches,
                                            self.num_patches]),
                                         attn_weights,
                                         axis=3,
                                         loc=L(eager_attn + ".softmax"),
                                         ip=ip).output
            attn_output = top.MatMulOp(T([1, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                       attn_weights,
                                       v_op,
                                       vit_mlir.none_op,
                                       do_relu=False,
                                       is_lora=False,
                                       loc=L(eager_attn + ".matmul2"),
                                       ip=ip).output
            attn_output = top.PermuteOp(T([1, self.num_patches, self.vnum_heads, self.vhead_dim]),
                                        attn_output,
                                        order=[0, 2, 1, 3],
                                        loc=L(eager_attn + ".permute"),
                                        ip=ip).output
            fa_op = top.ReshapeOp(
                T(hidden_shape),
                attn_output,
                shape=[-1, self.embed_dim],
                loc=L(f"{self.vit_path}.vision_model.encoder.layers.{id}.fattention.reshape"),
                ip=ip).output

            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 fa_op, [1, self.embed_dim, self.embed_dim],
                                 [1, self.num_patches, self.embed_dim],
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        mlp = f"{self.vit_path}.vision_model.encoder.layers.{id}.mlp"

        def vision_mlp(in_op):
            fc1_shape = [1, self.num_patches, self.vintermediate_size]
            fc2_shape = [1, self.num_patches, self.embed_dim]
            mlp_fc1 = f"{self.vit_path}.vision_model.encoder.layers.{id}.mlp.fc1"
            mlp_fc2 = f"{self.vit_path}.vision_model.encoder.layers.{id}.mlp.fc2"
            ln_op = self.layer_norm(vit_mlir, in_op, norm2, self.vit_ln_eps)
            #mlp
            fc1_op = self.linear(vit_mlir,
                                 mlp_fc1,
                                 ln_op, [1, self.embed_dim, self.vintermediate_size],
                                 fc1_shape,
                                 force_bias=True)
            act_op = self.activate(vit_mlir, fc1_op, self.config.vision_config.hidden_act, mlp_fc1)
            fc2_op = self.linear(vit_mlir,
                                 mlp_fc2,
                                 act_op, [1, self.vintermediate_size, self.embed_dim],
                                 fc2_shape,
                                 force_bias=True)
            return fc2_op

        in_op = vision_attention(in_op)
        fc2_op = vision_mlp(in_op)
        add_op = top.AddOp(T([1, self.num_patches, self.embed_dim]), [in_op, fc2_op],
                           loc=L(mlp + '.add'),
                           ip=ip).output
        return add_op

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        name = "vit"
        # create weights file
        vit_npz = "vit_top_weights.npz"
        patch_embed = f"{self.vit_path}.vision_model.embeddings.patch_embedding"
        position_embed = f"{self.vit_path}.vision_model.embeddings.position_embedding"
        # pack_embed = f"{self.vit_path}.vision_model.embeddings.packing_position_embedding"
        post_layernorm = f"{self.vit_path}.vision_model.post_layernorm"
        rotary_cos = f"{self.vit_path}.rotary_pos_emb.cos"
        rotary_sin = f"{self.vit_path}.rotary_pos_emb.sin"
        mlp_ar = f"mlp_AR"
        mlp_ar_prenorm = f"mlp_AR.pre_norm"
        mlp_ar_linear_2 = f"mlp_AR.linear_2"
        mlp_ar_linear_1 = f"mlp_AR.linear_1"

        def save_weights():
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + ".weight": cos,
                rotary_sin + ".weight": sin,
            }
            self.set_common_weight(patch_embed, weights_dict)
            self.set_common_weight(position_embed, weights_dict)
            self.set_common_weight(post_layernorm, weights_dict)
            for i in range(self.depth):
                self.set_common_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.layer_norm1", weights_dict)
                self.set_linear_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.self_attn.k_proj",
                    weights_dict)
                self.set_linear_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.self_attn.v_proj",
                    weights_dict)
                self.set_linear_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.self_attn.q_proj",
                    weights_dict)
                self.set_linear_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.self_attn.out_proj",
                    weights_dict)
                self.set_common_weight(
                    f"{self.vit_path}.vision_model.encoder.layers.{i}.layer_norm2", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.vision_model.encoder.layers.{i}.mlp.fc1",
                                       weights_dict)
                self.set_linear_weight(f"{self.vit_path}.vision_model.encoder.layers.{i}.mlp.fc2",
                                       weights_dict)
            self.set_linear_weight(mlp_ar_linear_1, weights_dict)
            self.set_linear_weight(mlp_ar_linear_2, weights_dict)
            self.set_common_weight(mlp_ar_prenorm, weights_dict)
            # save weights
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.in_channels, self.patch_size, self.patch_size]
        pos_ids_shape = [self.num_patches]
        mask_shape = [1, 1, self.num_patches, self.num_patches]
        resize_h = self.max_shape[0]
        resize_w = self.max_shape[1]
        grid_h = resize_h // self.patch_size
        grid_w = resize_w // self.patch_size
        self.image_grid_thw = [(1, grid_h, grid_w)]
        out_shape = [1, grid_h * grid_w // 4, self.out_hidden_size]
        input_shapes = [in_shape, pos_ids_shape, mask_shape]
        input_types = ['F32', 'INT32', 'F32']

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                name,
                                Platform.LLM,
                                input_types,
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_embedding(pixel_values_op):  #, position_ids_op):
            output_shape = [self.num_patches, self.embed_dim]
            weight_op0 = vit_mlir.create_weight_op(
                patch_embed + ".weight",
                [self.embed_dim, self.in_channels, self.patch_size, self.patch_size])
            weight_op1 = vit_mlir.create_weight_op(patch_embed + ".bias", [self.embed_dim])
            patch_embed_op = top.ConvOp(T([self.num_patches, self.embed_dim, 1, 1]),
                                        pixel_values_op,
                                        weight_op0,
                                        weight_op1,
                                        kernel_shape=[self.patch_size, self.patch_size],
                                        strides=[self.patch_size, self.patch_size],
                                        pads=[0, 0, 0, 0],
                                        dilations=[1, 1],
                                        loc=L(patch_embed),
                                        ip=ip).output
            patch_embed_op = top.ReshapeOp(T(output_shape),
                                           patch_embed_op,
                                           shape=output_shape,
                                           loc=L(patch_embed + ".reshape"),
                                           ip=ip).output
            # interpolate_pos_encoding
            num_positions = (self.image_size // self.patch_size)**2
            patch_pos_embed = vit_mlir.create_weight_op(position_embed + ".weight",
                                                        [1, num_positions, self.embed_dim])
            sqrt_num_positions = int(math.sqrt(num_positions))
            patch_pos_embed = top.ReshapeOp(
                T([1, sqrt_num_positions, sqrt_num_positions, self.embed_dim]),
                patch_pos_embed,
                shape=[1, sqrt_num_positions, sqrt_num_positions, self.embed_dim],
                loc=L(position_embed + ".reshape"),
                ip=ip).output
            patch_pos_embed = top.PermuteOp(T(
                [1, self.embed_dim, sqrt_num_positions, sqrt_num_positions]),
                                            patch_pos_embed,
                                            order=[0, 3, 1, 2],
                                            loc=L(position_embed + ".permute"),
                                            ip=ip).output
            patch_pos_embed = top.InterpOp(T([1, self.embed_dim, grid_h, grid_w]),
                                           patch_pos_embed,
                                           target_shape=vit_mlir.none_op,
                                           scale_h=float((grid_h - 1) / (sqrt_num_positions - 1)),
                                           scale_w=float((grid_w - 1) / (sqrt_num_positions - 1)),
                                           mode=StringAttr.get("linear"),
                                           coord_mode=StringAttr.get("align_corners"),
                                           loc=L(position_embed + ".interp"),
                                           ip=ip).output
            patch_pos_embed = top.PermuteOp(T([1, grid_h, grid_w, self.embed_dim]),
                                            patch_pos_embed,
                                            order=[0, 2, 3, 1],
                                            loc=L(position_embed + ".permute2"),
                                            ip=ip).output
            patch_pos_embed = top.ReshapeOp(T(output_shape),
                                            patch_pos_embed,
                                            shape=output_shape,
                                            loc=L(position_embed + ".reshape2"),
                                            ip=ip).output
            new_op = top.AddOp(T(output_shape), [patch_embed_op, patch_pos_embed],
                               loc=L(position_embed + ".add"),
                               ip=ip).output
            return new_op

        save_weights()
        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        new_op = vision_embedding(in0_op)

        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight", [self.num_patches, 72])
        cos_op = top.ReshapeOp(T([1, self.num_patches, 1, 72]),
                               new_weight,
                               shape=[1, -1, 1, 72],
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight", [self.num_patches, 72])
        sin_op = top.ReshapeOp(T([1, self.num_patches, 1, 72]),
                               new_weight,
                               shape=[1, -1, 1, 72],
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        # layers
        for idx in range(self.depth):
            new_op = self.vision_block(vit_mlir, idx, new_op, cos_op, sin_op)
        grid_h = grid_h // 2
        grid_w = grid_w // 2
        new_op = self.layer_norm(vit_mlir, new_op, post_layernorm, self.vit_ln_eps)
        new_op = self.layer_norm(vit_mlir, new_op, mlp_ar_prenorm, 1e-05)
        new_op = top.ReshapeOp(T([1, grid_h, 2, grid_w, 2, self.embed_dim]),
                               new_op,
                               shape=[1, grid_h, 2, grid_w, 2, self.embed_dim],
                               loc=L(mlp_ar + ".reshape"),
                               ip=ip).output
        new_op = top.PermuteOp(T([1, grid_h, 2, grid_w, 2, self.embed_dim]),
                               new_op,
                               order=[0, 1, 3, 2, 4, 5],
                               loc=L(mlp_ar + ".permute"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, grid_h * grid_w, 4 * self.embed_dim]),
                               new_op,
                               shape=[1, grid_h * grid_w, 4 * self.embed_dim],
                               loc=L(mlp_ar + ".reshape2"),
                               ip=ip).output
        new_op = self.linear(vit_mlir,
                             mlp_ar_linear_1,
                             new_op,
                             weight_shape=[1, 4 * self.embed_dim, 4 * self.embed_dim],
                             out_shape=[1, grid_h * grid_w, 4 * self.embed_dim],
                             force_bias=True)
        new_op = self.activate(vit_mlir, new_op, self.config.vision_config.hidden_act,
                               mlp_ar_linear_1)
        new_op = self.linear(vit_mlir,
                             mlp_ar_linear_2,
                             new_op,
                             weight_shape=[1, 4 * self.embed_dim, self.out_hidden_size],
                             out_shape=[1, grid_h * grid_w, self.out_hidden_size],
                             force_bias=True)
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)

    @override
    def compile_vit(self):
        if not self.do_vit:
            return
        name = "vit"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--disable_layer_group', f'--model {name}.bmodel'
        ]
        deploy_args.append('--quantize F32')
        deploy_args.append('--quant_output_bf16')
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.debug:
            deploy_args.append('--debug')
        if self.dynamic_vit:
            deploy_args.append('--dynamic')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")
