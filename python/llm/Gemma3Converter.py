# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# TODO: in Gemma3, the rms_norm weights should be f32

from .LlmConverter import *
from typing_extensions import override


class Gemma3Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.do_vit = True
        self.vit_f16_out_bf16 = True  # Gemma3 vit is f16, but we force output to bf16

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GEMMA3_INFO
        self.llm_config = config.text_config
        self.llm_type = LlmType.GEMMA3

    @override
    def init_config(self):
        super().init_config()
        self.tie_word_embeddings = True
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        vconfig = self.config.vision_config
        image_size = vconfig.image_size
        patch_size = vconfig.patch_size
        patches_per_image = image_size // patch_size
        num_patches = patches_per_image**2
        embed_dim = vconfig.hidden_size
        mm_tokens_per_image = self.config.mm_tokens_per_image
        hidden_act = vconfig.hidden_act

        # create weights file
        vit_npz = "vit_top_weights.npz"
        top_path = "vision_tower.vision_model"
        post_layernorm = f"{top_path}.post_layernorm"
        patch_embedding = f"{top_path}.embeddings.patch_embedding"
        positional_embedding = f"{top_path}.embeddings.position_embedding"
        mm_projector_norm = f"multi_modal_projector.mm_soft_emb_norm"
        mm_projector_mm = f"multi_modal_projector.mm_input_projection_weight"

        # Placeholder for the actual implementation of generating MLIR for Siglip Vision
        def save_weights():
            weights_dict = {}
            self.set_common_weight(post_layernorm, weights_dict)
            for idx in range(vconfig.num_hidden_layers):
                layer_path = f"{top_path}.encoder.layers.{idx}"
                self.set_common_weight(f"{layer_path}.layer_norm1", weights_dict)
                self.set_common_weight(f"{layer_path}.layer_norm2", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.fc1", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.fc2", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.q_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.k_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.v_proj", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.out_proj", weights_dict)

            # refine position embedding
            pos_embed = self.model.read(positional_embedding + ".weight")
            pos_ids = np.arange(num_patches, dtype=np.int32).reshape((1, num_patches))
            pos_embed_data = pos_embed[pos_ids]
            weights_dict[positional_embedding + ".weight"] = pos_embed_data
            # refine patch embedding
            patch_embed = self.model.read(patch_embedding + ".weight")
            patch_embed_data = patch_embed.reshape(
                (embed_dim, -1)).transpose(1, 0)  #[3*14*14, embed_dim]
            weights_dict[patch_embedding + ".weight"] = patch_embed_data
            patch_embed_bias = self.model.read(patch_embedding + ".bias")
            weights_dict[patch_embedding + ".bias"] = patch_embed_bias
            # mm projector
            weights_dict[mm_projector_mm] = self.model.read(mm_projector_mm)
            self.set_common_weight(mm_projector_norm, weights_dict, WeightType.RMS_NORM)
            # save to npz
            np.savez(vit_npz, **weights_dict)

        save_weights()
        # generate mlir
        in_shape = [1, 3, image_size, image_size]
        out_shape = [1, mm_tokens_per_image, self.hidden_size]
        hidden_shape = [1, num_patches, embed_dim]
        vit_mlir = MLIRImporter([in_shape], [out_shape],
                                "vit",
                                Platform.LLM, ["F32"],
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        in_op = vit_mlir.create_input_op(L('pixel_values'), 0)
        new_op = top.ReshapeOp(T(
            [1, 3, patches_per_image, patch_size, patches_per_image, patch_size]),
                               in_op,
                               loc=L("pixel_reshape"),
                               ip=ip).output
        new_op = top.PermuteOp(T(
            [1, patches_per_image, patches_per_image, 3, patch_size, patch_size]),
                               new_op,
                               order=[0, 2, 4, 1, 3, 5],
                               loc=L("pixel_transpose"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, num_patches, 3 * patch_size * patch_size]),
                               new_op,
                               loc=L("pixel_reshape2"),
                               ip=ip).output
        new_weight = vit_mlir.create_weight_op(patch_embedding + ".weight",
                                               [3 * patch_size * patch_size, embed_dim])
        new_bias = vit_mlir.create_weight_op(patch_embedding + ".bias", [1, 1, embed_dim])

        new_op = top.MatMulOp(T(hidden_shape),
                              new_op,
                              new_weight,
                              new_bias,
                              loc=L(patch_embedding),
                              ip=ip).output
        new_weight = vit_mlir.create_weight_op(positional_embedding + ".weight", hidden_shape)
        new_op = top.AddOp(T(hidden_shape), [new_op, new_weight],
                           loc=L(patch_embedding + ".add"),
                           ip=ip).output

        def vision_mlp(in_op, layer_path):
            intermediate_size = vconfig.intermediate_size
            in_shape = [1, num_patches, embed_dim]

            new_op = self.layer_norm(vit_mlir,
                                     in_op,
                                     f"{layer_path}.layer_norm2",
                                     eps=vconfig.layer_norm_eps)

            fc1_op = self.linear(vit_mlir, f"{layer_path}.mlp.fc1", new_op,
                                 [embed_dim, intermediate_size],
                                 [1, num_patches, intermediate_size])
            act_op = self.activate(vit_mlir, fc1_op, hidden_act, layer_path)
            fc2_op = self.linear(vit_mlir, f"{layer_path}.mlp.fc2", act_op,
                                 [intermediate_size, embed_dim], in_shape)
            new_op = top.AddOp(T([1, num_patches, embed_dim]), [in_op, fc2_op],
                               loc=L(layer_path + ".add"),
                               ip=ip).output
            return new_op

        for idx in range(vconfig.num_hidden_layers):
            layer_path = f"{top_path}.encoder.layers.{idx}"
            norm_path = f"{layer_path}.layer_norm1"
            residual_op = new_op
            new_op = self.layer_norm(vit_mlir, new_op, norm_path, eps=vconfig.layer_norm_eps)
            q_op = self.linear(vit_mlir, f"{layer_path}.self_attn.q_proj", new_op,
                               [embed_dim, embed_dim], hidden_shape)
            k_op = self.linear(vit_mlir, f"{layer_path}.self_attn.k_proj", new_op,
                               [embed_dim, embed_dim], [1, num_patches, embed_dim])
            v_op = self.linear(vit_mlir, f"{layer_path}.self_attn.v_proj", new_op,
                               [embed_dim, embed_dim], [1, num_patches, embed_dim])
            head_dim = vconfig.hidden_size // vconfig.num_attention_heads
            new_shape = [1, num_patches, vconfig.num_attention_heads, head_dim]
            q_op = top.ReshapeOp(T(new_shape),
                                 q_op,
                                 loc=L(f"{layer_path}.self_attn.q_reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(new_shape),
                                 k_op,
                                 loc=L(f"{layer_path}.self_attn.k_reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(new_shape),
                                 v_op,
                                 loc=L(f"{layer_path}.self_attn.v_reshape"),
                                 ip=ip).output
            fa_op = top.FAttentionOp(T(hidden_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     scale=head_dim**-0.5,
                                     batch=1,
                                     q_head=vconfig.num_attention_heads,
                                     kv_head=vconfig.num_attention_heads,
                                     dim=head_dim,
                                     mq=num_patches,
                                     mk=num_patches,
                                     loc=L(f"{layer_path}.fattention"),
                                     ip=ip).output
            o_op = self.linear(vit_mlir, f"{layer_path}.self_attn.out_proj", fa_op,
                               [embed_dim, embed_dim], hidden_shape)
            new_op = top.AddOp(T(hidden_shape), [residual_op, o_op],
                               loc=L(f"{layer_path}.residual_add"),
                               ip=ip).output
            new_op = vision_mlp(new_op, layer_path)

        new_op = self.layer_norm(vit_mlir, new_op, post_layernorm, eps=vconfig.layer_norm_eps)
        ## mm_projector
        new_op = top.PermuteOp(T([1, embed_dim, num_patches]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L("mm_projector_transpose"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, embed_dim, patches_per_image, patches_per_image]),
                               new_op,
                               loc=L("mm_projector_reshape"),
                               ip=ip).output
        tokens_per_side = int(mm_tokens_per_image**0.5)
        kernel_size = patches_per_image // tokens_per_side
        new_op = top.AvgPoolOp(T([1, embed_dim, tokens_per_side, tokens_per_side]),
                               new_op,
                               kernel_shape=[kernel_size, kernel_size],
                               strides=[kernel_size, kernel_size],
                               pads=[0, 0, 0, 0, 0, 0, 0, 0],
                               loc=L("mm_projector_avgpool"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, embed_dim, mm_tokens_per_image]),
                               new_op,
                               loc=L("mm_projector_reshape2"),
                               ip=ip).output
        new_op = top.PermuteOp(T([1, mm_tokens_per_image, embed_dim]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L("mm_projector_transpose2"),
                               ip=ip).output
        new_op = self.rms_norm(vit_mlir, new_op, mm_projector_norm, eps=vconfig.layer_norm_eps)
        new_weight = vit_mlir.create_weight_op(mm_projector_mm, [embed_dim, self.hidden_size])
        new_op = top.MatMulOp(T([1, mm_tokens_per_image, self.hidden_size]),
                              new_op,
                              new_weight,
                              vit_mlir.none_op,
                              loc=L("mm_projector_matmul"),
                              ip=ip).output
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open("vit.mlir", "w") as f:
            f.write(mlir_txt)
