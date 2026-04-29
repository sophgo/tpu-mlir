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
import torch.nn.functional as F


def reorder(shape, order):
    return [shape[i] for i in order]


class LFM2VLConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_shape = args.max_shape
        self.do_vit = True
        # vision config
        self.init_vconfig()
        self.vit_path = "model.vision_tower"
        self.image_grid_thw = []

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = LFM2_INFO
        self.llm_config = config.text_config
        self.llm_type = LlmType.LFM2
        self.layer_types = self.llm_config.layer_types
        self.conv_L_cache = self.llm_config.conv_L_cache

    def init_vconfig(self):
        self.out_hidden_size = self.config.text_config.hidden_size
        self.downsample_factor = self.config.downsample_factor
        self.projector_hidden_size = self.config.projector_hidden_size
        self.vconfig = self.config.vision_config
        self.vit_ln_eps = self.vconfig.layer_norm_eps  # 1e-6
        self.in_channels = self.vconfig.num_channels
        self.depth = self.vconfig.num_hidden_layers
        self.batch_size = 1
        self.patch_size = self.vconfig.patch_size  # 16
        self.tile_size = 512
        self.num_patches = self.tile_size // self.patch_size * self.tile_size // self.patch_size
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.embed_dim = self.vconfig.hidden_size  #1152
        self.vnum_heads = self.vconfig.num_attention_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [3, self.max_input_length]

    def eager_attention_siglip(self,
                               mlir,
                               q_op,
                               q_shape,
                               k_op,
                               v_op,
                               kv_shape,
                               mask_op=None,
                               loc=""):
        ip = mlir.insert_point

        T = mlir.get_tensor_type
        L = lambda name: self.get_loc(name, mlir)

        k_shape1 = reorder(kv_shape, [0, 1, 3, 2])
        k_op = top.PermuteOp(T(k_shape1), k_op, order=[0, 1, 3, 2], loc=L(loc + ".permute"),
                             ip=ip).output
        attn_weights_shape = [q_shape[0], q_shape[1], q_shape[2], k_shape1[3]]
        attn_weights = top.MatMulOp(T(attn_weights_shape),
                                    q_op,
                                    k_op,
                                    mlir.none_op,
                                    do_relu=False,
                                    is_lora=False,
                                    loc=L(loc + ".matmul"),
                                    ip=ip).output
        if mask_op is not None:
            attn_weights = top.AddOp(T(attn_weights_shape), [attn_weights, mask_op],
                                     loc=L(loc + ".add"),
                                     ip=ip).output
        scaling_value = float(self.vhead_dim**-0.5)
        attn_weights = top.MulConstOp(T(attn_weights_shape),
                                      attn_weights,
                                      const_val=scaling_value,
                                      loc=L(loc + ".scale"),
                                      ip=ip).output
        attn_weights = top.SoftmaxOp(T(attn_weights_shape),
                                     attn_weights,
                                     axis=3,
                                     loc=L(loc + ".softmax"),
                                     ip=ip).output
        attn_output_shape = [q_shape[0], q_shape[1], q_shape[2], kv_shape[3]]
        attn_output = top.MatMulOp(T(attn_output_shape),
                                   attn_weights,
                                   v_op,
                                   mlir.none_op,
                                   do_relu=False,
                                   is_lora=False,
                                   loc=L(loc + ".matmul1"),
                                   ip=ip).output
        attn_output_shape = reorder(attn_output_shape, [0, 2, 1, 3])
        attn_output = top.PermuteOp(T(attn_output_shape),
                                    attn_output,
                                    order=[0, 2, 1, 3],
                                    loc=L(loc + ".permute1"),
                                    ip=ip).output
        return attn_output

    def vision_block(self, vit_mlir, id: int, in_op, mask_op=None):
        norm1 = f"{self.vit_path}.vision_model.encoder.layers.{id}.layer_norm1"
        eager_attn = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.eager_attn"
        attn_q = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.q_proj"
        attn_k = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.k_proj"
        attn_v = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.v_proj"
        attn_proj = f"{self.vit_path}.vision_model.encoder.layers.{id}.self_attn.out_proj"
        norm2 = f"{self.vit_path}.vision_model.encoder.layers.{id}.layer_norm2"
        ip = vit_mlir.insert_point

        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        def vision_attention(in_op, mask_op):
            hidden_shape = [1, self.num_patches, self.embed_dim]
            in_op = top.ReshapeOp(T(hidden_shape),
                                  in_op,
                                  shape=hidden_shape,
                                  loc=L(norm1 + ".reshape"),
                                  ip=ip).output
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, self.vit_ln_eps)
            q_op = self.linear(vit_mlir,
                               attn_q,
                               norm1_op, [1, self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            k_op = self.linear(vit_mlir,
                               attn_k,
                               norm1_op, [1, self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               attn_v,
                               norm1_op, [1, self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            qk_shape = [self.batch_size, self.num_patches, self.vnum_heads, self.vhead_dim]
            q_op = top.ReshapeOp(T(qk_shape),
                                 q_op,
                                 shape=qk_shape,
                                 loc=L(attn_q + ".reshape"),
                                 ip=ip).output

            k_op = top.ReshapeOp(T(qk_shape),
                                 k_op,
                                 shape=qk_shape,
                                 loc=L(attn_k + ".reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(qk_shape),
                                 v_op,
                                 shape=qk_shape,
                                 loc=L(attn_v + ".reshape"),
                                 ip=ip).output
            q_op = top.PermuteOp(T(
                [self.batch_size, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 q_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_q + ".permute"),
                                 ip=ip).output
            k_op = top.PermuteOp(T(
                [self.batch_size, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 k_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_k + ".permute"),
                                 ip=ip).output
            v_op = top.PermuteOp(T(
                [self.batch_size, self.vnum_heads, self.num_patches, self.vhead_dim]),
                                 v_op,
                                 order=[0, 2, 1, 3],
                                 loc=L(attn_v + ".permute"),
                                 ip=ip).output
            qk_shape = [self.batch_size, self.vnum_heads, self.num_patches, self.vhead_dim]
            # eager_attention
            attn_output = self.eager_attention_siglip(vit_mlir, q_op, qk_shape, k_op, v_op,
                                                      qk_shape, mask_op, eager_attn)
            attn_output = top.ReshapeOp(
                T(hidden_shape),
                attn_output,
                shape=[-1, self.embed_dim],
                loc=L(f"{self.vit_path}.vision_model.encoder.layers.{id}.fattention.reshape"),
                ip=ip).output
            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 attn_output, [1, self.embed_dim, self.embed_dim],
                                 [self.batch_size, self.num_patches, self.embed_dim],
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        mlp = f"{self.vit_path}.vision_model.encoder.layers.{id}.mlp"

        def vision_mlp(in_op):
            fc1_shape = [self.batch_size, self.num_patches, self.vintermediate_size]
            fc2_shape = [self.batch_size, self.num_patches, self.embed_dim]
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

        in_op = vision_attention(in_op, mask_op)
        fc2_op = vision_mlp(in_op)
        add_op = top.AddOp(T([self.batch_size, self.num_patches, self.embed_dim]), [in_op, fc2_op],
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
        post_layernorm = f"{self.vit_path}.vision_model.post_layernorm"
        mlp_ar = f"model.multi_modal_projector"
        mlp_ar_prenorm = f"model.multi_modal_projector.layer_norm"
        mlp_ar_linear_1 = f"model.multi_modal_projector.linear_1"
        mlp_ar_linear_2 = f"model.multi_modal_projector.linear_2"

        def save_weights():
            weights_dict = {}
            self.set_linear_weight(patch_embed, weights_dict)
            self.set_common_weight(position_embed, weights_dict)
            positional_embeddings = torch.tensor(weights_dict[position_embed + ".weight"]).reshape(
                [self.patch_size, self.patch_size, self.embed_dim])
            positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
            self.spatial_size = self.tile_size // self.patch_size
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized_embeddings = resized_embeddings.reshape(self.embed_dim, self.spatial_size *
                                                            self.spatial_size).transpose(0, 1)
            weights_dict[position_embed + ".weight"] = resized_embeddings.numpy()

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
        in_shape = [
            self.batch_size, self.num_patches, self.in_channels * self.patch_size * self.patch_size
        ]  # [1,1024,768]
        # out_shape = [self.batch_size, self.num_patches, self.embed_dim]
        out_shape = [self.batch_size, self.patch_size**2, self.out_hidden_size]  # [1,256,2048]
        input_shapes = [in_shape]
        input_types = ['F32', 'INT32']

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                name,
                                Platform.LLM,
                                input_types,
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point

        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        def vision_embedding(pixel_values_op):  #, position_ids_op):
            #patch_embedding
            patch_embed_op = self.linear(
                vit_mlir,
                patch_embed,
                pixel_values_op,
                weight_shape=[
                    1, self.in_channels * self.patch_size * self.patch_size, self.embed_dim
                ],
                out_shape=[self.batch_size, self.num_patches, self.embed_dim],
                force_bias=True)
            patch_pos_embed = vit_mlir.create_weight_op(
                position_embed + ".weight", [self.batch_size, self.spatial_size**2, self.embed_dim])
            new_op = top.AddOp(T([self.batch_size, self.num_patches, self.embed_dim]),
                               [patch_embed_op, patch_pos_embed],
                               loc=L(position_embed + ".add"),
                               ip=ip).output
            return new_op

        save_weights()
        in0_op = vit_mlir.create_input_op(L('pixel_values'), 0)
        new_op = vision_embedding(in0_op)
        # layers
        for idx in range(self.depth):
            new_op = self.vision_block(vit_mlir, idx, new_op, None)
        new_op = self.layer_norm(vit_mlir, new_op, post_layernorm, self.vit_ln_eps)

        new_op = top.ReshapeOp(
            T([self.batch_size, self.spatial_size, self.spatial_size, self.embed_dim]),
            new_op,
            shape=[self.batch_size, self.spatial_size, self.spatial_size, self.embed_dim],
            loc=L(post_layernorm + ".reshape"),
            ip=ip).output
        # Lfm2VlMultiModalProjector
        grid_size = self.spatial_size // self.downsample_factor
        new_op = top.ReshapeOp(T([
            self.batch_size, self.spatial_size, grid_size, self.downsample_factor * self.embed_dim
        ]),
                               new_op,
                               shape=[
                                   self.batch_size, self.spatial_size, grid_size,
                                   self.downsample_factor * self.embed_dim
                               ],
                               loc=L(mlp_ar + ".reshape1"),
                               ip=ip).output
        new_op = top.PermuteOp(T([
            self.batch_size, grid_size, self.spatial_size, self.downsample_factor * self.embed_dim
        ]),
                               new_op,
                               order=[0, 2, 1, 3],
                               loc=L(mlp_ar + ".permute1"),
                               ip=ip).output
        new_op = top.ReshapeOp(
            T([self.batch_size, grid_size, grid_size, self.embed_dim * self.downsample_factor**2]),
            new_op,
            shape=[
                self.batch_size, grid_size, grid_size, self.embed_dim * self.downsample_factor**2
            ],
            loc=L(mlp_ar + ".reshape2"),
            ip=ip).output
        new_op = top.PermuteOp(T(
            [self.batch_size, grid_size, grid_size, self.embed_dim * self.downsample_factor**2]),
                               new_op,
                               order=[0, 2, 1, 3],
                               loc=L(mlp_ar + ".permute2"),
                               ip=ip).output
        new_op = self.layer_norm(vit_mlir, new_op, mlp_ar_prenorm, 1e-06)
        new_op = self.linear(
            vit_mlir,
            mlp_ar_linear_1,
            new_op,
            weight_shape=[1, 4 * self.embed_dim, self.projector_hidden_size],
            out_shape=[self.batch_size, grid_size, grid_size, self.projector_hidden_size],
            force_bias=True)
        new_op = self.activate(vit_mlir, new_op, self.config.projector_hidden_act, mlp_ar_linear_1)
        new_op = self.linear(
            vit_mlir,
            mlp_ar_linear_2,
            new_op,
            weight_shape=[1, self.projector_hidden_size, self.out_hidden_size],
            out_shape=[self.batch_size, grid_size, grid_size, self.out_hidden_size],
            force_bias=True)
        new_op = top.ReshapeOp(T([self.batch_size, grid_size**2, self.out_hidden_size]),
                               new_op,
                               shape=[self.batch_size, grid_size**2, self.out_hidden_size],
                               loc=L(mlp_ar + ".reshape3"),
                               ip=ip).output
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)

    @override
    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
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

        # Lfm2ShortConv:
        short_conv_path = TOP_PATH + "conv"
        short_conv = short_conv_path + ".conv"
        short_conv_in_proj = short_conv_path + ".in_proj"
        short_conv_out_proj = short_conv_path + ".out_proj"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict, WeightType.RMSNORM)
        if self.layer_types[idx] == "full_attention":
            self.set_linear_weight(q_proj, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(k_proj, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(v_proj, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(o_proj, weight_dict, do_lora=self.do_lora)
            self.set_common_weight(q_norm, weight_dict, WeightType.RMSNORM)
            self.set_common_weight(k_norm, weight_dict, WeightType.RMSNORM)
        else:
            self.set_common_weight(short_conv, weight_dict)
            self.set_linear_weight(short_conv_in_proj, weight_dict, do_lora=self.do_lora)
            self.set_linear_weight(short_conv_out_proj, weight_dict, do_lora=self.do_lora)
        self.set_common_weight(post_attn_ln, weight_dict, WeightType.RMSNORM)
        self.set_linear_weight(mlp_gate, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_up, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_down, weight_dict, do_lora=self.do_lora)
        if do_norm:
            self.set_common_weight(norm, weight_dict, WeightType.RMSNORM)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weight_keys.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            self.intermediate_size = int(2 * self.llm_config.intermediate_size / 3)
            self.intermediate_size = self.llm_config.block_multiple_of * (
                (self.intermediate_size + self.llm_config.block_multiple_of - 1) //
                self.llm_config.block_multiple_of)
            ip = mlir_gen.insert_point
            len = input_shape[1]
            new_op = in_op
            new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            gate_op = self.linear(mlir_gen,
                                  mlp_gate,
                                  new_op, [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size],
                                  do_lora=self.do_lora)
            act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
            up_op = self.linear(mlir_gen,
                                mlp_up,
                                new_op, [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size],
                                do_lora=self.do_lora)
            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen,
                                  mlp_down,
                                  new_op, [self.intermediate_size, self.hidden_size],
                                  input_shape,
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
            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]
            mask_shape = [1, 1, input_len, input_len]
            if self.layer_types[idx] == "full_attention":
                input_shapes = [input_shape, id_shape, mask_shape]
                input_dtypes = ["F32", "INT32", "F32"]
                output_shapes = [input_shape, kv_shape, kv_shape]
            else:
                conv_state_dim_shape = [1]
                conv_state_shape = [1, self.hidden_size, 3]
                input_shapes = [input_shape, conv_state_dim_shape, conv_state_dim_shape]
                input_dtypes = ["F32", "INT32", "INT32"]
                output_shapes = [input_shape, conv_state_shape]
            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      Platform.LLM,
                                      input_dtypes,
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)
            if self.layer_types[idx] == "full_attention":
                in1_op = block_mlir.create_input_op(L("position_ids"), 1)
                in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
                # q_proj
                q_dim = self.num_attention_heads * self.head_dim
                q_op = self.linear(block_mlir,
                                   q_proj,
                                   ln_op, [self.hidden_size, q_dim], [1, input_len, q_dim],
                                   do_lora=self.do_lora)
                # k_proj
                k_op = self.linear(block_mlir,
                                   k_proj,
                                   ln_op, [self.hidden_size, self.kv_dim],
                                   [1, input_len, self.kv_dim],
                                   do_lora=self.do_lora)
                # v_proj
                v_op = self.linear(block_mlir,
                                   v_proj,
                                   ln_op, [self.hidden_size, self.kv_dim],
                                   [1, input_len, self.kv_dim],
                                   do_lora=self.do_lora)
                # reshape q,k,v
                q_op = top.ReshapeOp(T(q_shape),
                                     q_op,
                                     shape=[1, input_len, self.num_attention_heads, self.head_dim],
                                     loc=L(q_proj + ".reshape"),
                                     ip=ip).output
                k_op = top.ReshapeOp(T(kv_shape),
                                     k_op,
                                     shape=[1, input_len, self.num_key_value_heads, self.head_dim],
                                     loc=L(k_proj + ".reshape"),
                                     ip=ip).output
                v_op = top.ReshapeOp(T(kv_shape),
                                     v_op,
                                     shape=[1, input_len, self.num_key_value_heads, self.head_dim],
                                     loc=L("v_cache"),
                                     ip=ip).output
                # apply q_norm after reshape (per head normalization)
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                # apply k_norm after reshape (per head normalization)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)
                # rotary cos/sin
                q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                                   rotary_sin)
                return_ops.append(k_op)
                return_ops.append(v_op)
                # Apply attention with proper mask handling
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
                # attn_output = self.eager_attention(block_mlir, q_op, q_shape, k_op, v_op, kv_shape, None, TOP_PATH+"eager_attn")
                o_op = self.linear(block_mlir,
                                   o_proj,
                                   fa_op, [q_dim, self.hidden_size],
                                   input_shape,
                                   do_lora=self.do_lora)
            else:
                x_op = ln_op
                conv_state_offsetT_op = block_mlir.create_input_op(L("conv_state_offsetT"), 1)
                conv_state_endsT_op = block_mlir.create_input_op(L("conv_state_endsT"), 2)
                BCx_op = self.linear(block_mlir,
                                     short_conv_in_proj,
                                     x_op, [self.hidden_size, 3 * self.hidden_size],
                                     [1, input_len, 3 * self.hidden_size],
                                     do_lora=self.do_lora)
                BCx_op = top.PermuteOp(T([1, 3 * self.hidden_size, input_len]),
                                       BCx_op,
                                       order=[0, 2, 1],
                                       loc=L(short_conv_in_proj + ".permute"),
                                       ip=ip).output
                B_op = top.SliceOp(T([1, self.hidden_size, input_len]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, 0, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, input_len],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_b"),
                                   ip=ip).output
                C_op = top.SliceOp(T([1, self.hidden_size, input_len]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, self.hidden_size, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, input_len],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_c"),
                                   ip=ip).output
                x_op = top.SliceOp(T([1, self.hidden_size, input_len]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, 2 * self.hidden_size, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, input_len],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_x"),
                                   ip=ip).output
                Bx_op = top.MulOp(T([1, self.hidden_size, input_len]), [B_op, x_op],
                                  loc=L(short_conv_in_proj + ".mul"),
                                  ip=ip).output
                conv_state_op = top.SliceOp(T([1, self.hidden_size, 3]),
                                            Bx_op,
                                            conv_state_offsetT_op,
                                            conv_state_endsT_op,
                                            block_mlir.none_op,
                                            offset=[0, 0, 0],
                                            steps=[1, 1, 1],
                                            ends=[3, 3, 3],
                                            axes=[2],
                                            loc=L(short_conv_in_proj + ".slice_conv_state"),
                                            ip=ip).output
                return_ops.append(conv_state_op)
                conv_weight_op = block_mlir.create_weight_op(
                    short_conv + ".weight", [self.hidden_size, 1, 1, self.conv_L_cache])
                Bx_op = top.ReshapeOp(T([1, self.hidden_size, 1, input_len]),
                                      Bx_op,
                                      shape=[1, self.hidden_size, 1, input_len],
                                      loc=L(short_conv_in_proj + ".reshape_bx"),
                                      ip=ip).output
                conv_out_op = top.ConvOp(T(
                    [1, self.hidden_size, 1, input_len + self.conv_L_cache - 1]),
                                         Bx_op,
                                         conv_weight_op,
                                         block_mlir.none_op,
                                         group=self.hidden_size,
                                         kernel_shape=[1, self.conv_L_cache],
                                         strides=[1, 1],
                                         pads=[0, self.conv_L_cache - 1, 0, self.conv_L_cache - 1],
                                         dilations=[1, 1],
                                         loc=L(short_conv + ".conv"),
                                         ip=ip).output
                conv_out_op = top.ReshapeOp(
                    T([1, self.hidden_size, input_len + self.conv_L_cache - 1]),
                    conv_out_op,
                    shape=[1, self.hidden_size, input_len + self.conv_L_cache - 1],
                    loc=L(short_conv + ".reshape_conv_out"),
                    ip=ip).output
                conv_out_op = top.SliceOp(T([1, self.hidden_size, input_len]),
                                          conv_out_op,
                                          block_mlir.none_op,
                                          block_mlir.none_op,
                                          block_mlir.none_op,
                                          offset=[0, 0, 0],
                                          steps=[1, 1, 1],
                                          ends=[1, self.hidden_size, input_len],
                                          axes=[],
                                          loc=L(short_conv + ".slice_conv_out"),
                                          ip=ip).output

                y_op = top.MulOp(T([1, self.hidden_size, input_len]), [C_op, conv_out_op],
                                 loc=L(short_conv + ".mul"),
                                 ip=ip).output
                y_op = top.PermuteOp(T([1, input_len, self.hidden_size]),
                                     y_op,
                                     order=[0, 2, 1],
                                     loc=L(short_conv + ".permute_out"),
                                     ip=ip).output
                o_op = self.linear(block_mlir,
                                   short_conv_out_proj,
                                   y_op, [self.hidden_size, self.hidden_size],
                                   input_shape,
                                   do_lora=self.do_lora)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
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
            input_shape = [1, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            mask_shape = [1, 1, 1, self.seq_length + 1]

            q_shape = [1, 1, self.num_attention_heads, self.head_dim]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]

            if self.layer_types[idx] == "full_attention":
                input_shapes = [input_shape, id_shape, mask_shape, history_shape, history_shape]
                input_dtypes = ["F32", "INT32", "F32", "F32", "F32"]
                output_shapes = [input_shape, kv_shape, kv_shape]
            else:
                conv_state_shape = [1, self.hidden_size, self.conv_L_cache]
                input_shapes = [input_shape, conv_state_shape]
                input_dtypes = ["F32", "F32"]
                output_shapes = [input_shape, conv_state_shape]
            block_mlir = MLIRImporter(input_shapes,
                                      output_shapes,
                                      name,
                                      Platform.LLM,
                                      input_dtypes,
                                      lora_rank=self.lora_rank,
                                      weight_file=f"../{weight_file}")

            T = block_mlir.get_tensor_type
            L = lambda name: self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)
            if self.layer_types[idx] == "full_attention":
                in1_op = block_mlir.create_input_op(L("position_ids"), 1)
                in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
                in3_op = block_mlir.create_input_op(L("history_k"), 3)
                in4_op = block_mlir.create_input_op(L("history_v"), 4)
                # q_proj
                q_dim = self.num_attention_heads * self.head_dim
                q_op = self.linear(block_mlir,
                                   q_proj,
                                   ln_op, [self.hidden_size, q_dim], [1, 1, q_dim],
                                   do_lora=self.do_lora)
                # k_proj
                k_op = self.linear(block_mlir,
                                   k_proj,
                                   ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                                   do_lora=self.do_lora)
                # v_proj
                v_op = self.linear(block_mlir,
                                   v_proj,
                                   ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                                   do_lora=self.do_lora)
                # reshape q,k,v
                q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
                k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
                v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
                # apply q_norm after reshape (per head normalization)
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                # apply k_norm after reshape (per head normalization)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)
                # rotary cos/sin
                q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                                   rotary_sin)
                return_ops.append(k_op)
                return_ops.append(v_op)
                k_op = in3_op
                v_op = in4_op
                fa_op = top.FAttentionOp(
                    T([1, 1, q_dim]),
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
                    mk=self.seq_length,  # + 1,
                    keep_dims=False,
                    loc=L(TOP_PATH + "fattention"),
                    ip=ip).output
                o_op = self.linear(block_mlir,
                                   o_proj,
                                   fa_op, [q_dim, self.hidden_size],
                                   input_shape,
                                   do_lora=self.do_lora)
            else:
                conv_state_op = block_mlir.create_input_op(L("conv_state"), 1)
                x_op = ln_op
                BCx_op = self.linear(block_mlir,
                                     short_conv_in_proj,
                                     x_op, [self.hidden_size, 3 * self.hidden_size],
                                     [1, 1, 3 * self.hidden_size],
                                     do_lora=self.do_lora)
                BCx_op = top.PermuteOp(T([1, 3 * self.hidden_size, 1]),
                                       BCx_op,
                                       order=[0, 2, 1],
                                       loc=L(short_conv_in_proj + ".permute"),
                                       ip=ip).output
                B_op = top.SliceOp(T([1, self.hidden_size, 1]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, 0, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, 1],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_b"),
                                   ip=ip).output
                C_op = top.SliceOp(T([1, self.hidden_size, 1]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, self.hidden_size, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, 1],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_c"),
                                   ip=ip).output
                x_op = top.SliceOp(T([1, self.hidden_size, 1]),
                                   BCx_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   block_mlir.none_op,
                                   offset=[0, 2 * self.hidden_size, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, self.hidden_size, 1],
                                   axes=[],
                                   loc=L(short_conv_in_proj + ".slice_x"),
                                   ip=ip).output
                Bx_op = top.MulOp(T([1, self.hidden_size, 1]), [B_op, x_op],
                                  loc=L(short_conv_in_proj + ".mul"),
                                  ip=ip).output
                # Properly implement the rolling mechanism to match PyTorch's conv_state.roll(shifts=-1, dims=-1)
                # This moves all elements one position to the left and inserts new element at the end
                # Original conv_state: [x0, x1, x2, ..., x_(L-1)]
                # After roll:          [x1, x2, ..., x_(L-1), new_element]

                # Extract the "body" (elements from index 1 to L-1)
                body_op = top.SliceOp(
                    T([1, self.hidden_size, self.conv_L_cache - 1]),
                    conv_state_op,
                    block_mlir.none_op,
                    block_mlir.none_op,
                    block_mlir.none_op,
                    offset=[0, 0, 1],  # Start from index 1
                    steps=[1, 1, 1],
                    ends=[1, self.hidden_size, self.conv_L_cache],  # End at index L
                    axes=[],
                    loc=L(short_conv_path + ".conv_state_body"),
                    ip=ip).output

                # Concatenate body [x1, x2, ..., x_(L-1)] with new element [head_op] at the end
                conv_state_op = top.ConcatOp(
                    T([1, self.hidden_size, self.conv_L_cache]),
                    [body_op, Bx_op
                     ],  # Note: Using Bx_op instead of head_op since it's the new value
                    axis=2,
                    only_merge=True,
                    loc=L(short_conv_path + ".conv_state_rolled"),
                    ip=ip).output
                return_ops.append(conv_state_op)
                conv_weight_op = block_mlir.create_weight_op(short_conv + ".weight",
                                                             [self.hidden_size, self.conv_L_cache])
                conv_out_op = top.MulOp(T([1, self.hidden_size, self.conv_L_cache]),
                                        [conv_state_op, conv_weight_op],
                                        loc=L(short_conv_path + ".conv_out_muled"),
                                        ip=ip).output
                conv_out_op = top.ReduceOp(T([1, self.hidden_size, 1]),
                                           conv_out_op,
                                           axes=[2],
                                           keepdims=True,
                                           mode=StringAttr.get("ReduceSum"),
                                           loc=L(short_conv_path + ".reducesum"),
                                           ip=ip).output
                y_op = top.MulOp(T([1, self.hidden_size, 1]), [C_op, conv_out_op],
                                 loc=L(short_conv + ".mul"),
                                 ip=ip).output
                y_op = top.PermuteOp(T([1, 1, self.hidden_size]),
                                     y_op,
                                     order=[0, 2, 1],
                                     loc=L(short_conv + ".permute_out"),
                                     ip=ip).output
                o_op = self.linear(block_mlir,
                                   short_conv_out_proj,
                                   y_op, [self.hidden_size, self.hidden_size],
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

        gen_block()
        if self.share_prompt:
            gen_block()
        gen_block_cache()
