# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .Qwen2_5VLConverter import *
from typing_extensions import override
import torch.nn as nn


# from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding
class SinusoidsPositionEmbedding(nn.Module):

    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OConverter(Qwen2_5VLConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.vit_path = "thinker.visual"
        self.extern_gen_mlirs.append(self.gen_audio_tower)
        self.extern_compiles.append(self.compile_audio_tower)
        self.extern_bmodels.append("audio.bmodel")

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = QWEN2_5O_INFO
        self.llm_config = config.thinker_config.text_config
        self.llm_type = LlmType.QWEN2

    @override
    def init_vconfig(self):
        self.vconfig = self.config.thinker_config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.temporal_patch_size = self.vconfig.temporal_patch_size
        self.spatial_merge_size = self.vconfig.spatial_merge_size
        self.in_channels = self.vconfig.in_chans
        self.depth = self.vconfig.depth
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        self.embed_dim = self.vconfig.hidden_size
        self.vnum_heads = self.vconfig.num_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [3, self.max_input_length]
        self.fullatt_block_indexes = self.vconfig.fullatt_block_indexes
        self.mrope_section = getattr(self.llm_config.rope_scaling, 'mrope_section', [16, 24, 24])

    @override
    def init_quantization(self):
        c = self.model_info.config
        self.quantization_config = getattr(self.config, c.quantization_config, None)
        if self.quantization_config:
            self.quant_mode = self.quantization_config["quant_method"]
            self.q_group_size = self.quantization_config["group_size"]
            self.quant_bits = self.quantization_config["bits"]
            if self.quant_mode == "awq":
                assert self.quantization_config["version"] == "gemm", (
                    "AWQ only support gemm version for now")
                assert self.quant_bits == 4, ("AWQ only support quant bits == 4 for now")
        if self.q_group_size < 0:
            self.q_group_size = 0

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        patch_embed = f"{self.vit_path}.patch_embed.proj"
        rotary_cos = f"{self.vit_path}.rotary.cos"
        rotary_sin = f"{self.vit_path}.rotary.sin"
        merger_ln_q = f"{self.vit_path}.merger.ln_q"
        merger_mlp0 = f"{self.vit_path}.merger.mlp.0"
        merger_mlp2 = f"{self.vit_path}.merger.mlp.2"

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
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm2", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.gate_proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.up_proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.down_proj", weights_dict)
                # qkv
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.q", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.k", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.v", weights_dict)

            # save weights
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.patch_dim]
        position_shape = [self.num_patches, 2]
        mask_shape = [1, 1, self.num_patches, self.num_patches]
        out_dim = self.num_patches // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.hidden_size]
        input_shapes = [in_shape, position_shape, mask_shape, mask_shape, [out_dim]]
        input_types = ['F32', 'INT32', 'F32', 'F32', 'INT32']

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
        in3_op = vit_mlir.create_input_op(L('window_attn_mask'), 3)
        in4_op = vit_mlir.create_input_op(L('reverse_index'), 4)
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
            mask_op = in2_op
            if id not in self.fullatt_block_indexes:
                mask_op = in3_op
            new_op = self.vision_block(vit_mlir, id, new_op, cos_op, sin_op, mask_op)

        # merge
        new_op = self.rms_norm(vit_mlir, new_op, merger_ln_q)
        out_dim = self.embed_dim * (self.spatial_merge_size**2)
        in_dim = self.num_patches // (self.spatial_merge_size**2)
        new_op = top.ReshapeOp(T([in_dim, out_dim]), new_op, loc=L(merger_ln_q + ".reshape"),
                               ip=ip).output
        new_op = self.linear(vit_mlir, merger_mlp0, new_op, [out_dim, out_dim], [in_dim, out_dim])
        new_op = self.activate(vit_mlir, new_op, ActType.GELU, merger_mlp0)
        new_op = self.linear(vit_mlir, merger_mlp2, new_op, [out_dim, self.hidden_size],
                             [in_dim, self.hidden_size])
        # reverse
        new_op = top.GatherOp(T([in_dim, self.hidden_size]),
                              new_op,
                              in4_op,
                              axis=0,
                              loc=L(merger_mlp2 + ".reverse"),
                              ip=ip).output
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

    def gen_audio_tower(self):
        audio_config = self.config.thinker_config.audio_config
        d_model = audio_config.d_model
        num_mel_bins = audio_config.num_mel_bins
        n_window = audio_config.n_window
        max_source_positions = audio_config.max_source_positions
        num_heads = audio_config.encoder_attention_heads
        head_dim = d_model // num_heads
        ffn_dim = audio_config.encoder_ffn_dim
        num_layers = audio_config.num_hidden_layers
        tqdm.write(f"generate audio tower mlir ...")
        audio_path = "thinker.audio_tower."
        # create weights file
        audio_npz = "audio_tower_top_weights.npz"
        conv1 = f"{audio_path}conv1"
        conv2 = f"{audio_path}conv2"
        ln_post = f"{audio_path}ln_post"
        proj = f"{audio_path}proj"
        position_embedding = f"{audio_path}position_embedding"
        weights_dict = {}

        def save_weights():
            self.set_common_weight(conv1, weights_dict)
            self.set_common_weight(conv2, weights_dict)
            positional_embedding = SinusoidsPositionEmbedding(max_source_positions, d_model)
            pos_emb = positional_embedding.forward(n_window)
            weights_dict[position_embedding + ".weight"] = pos_emb.reshape(1, n_window,
                                                                           d_model).numpy()
            self.set_common_weight(ln_post, weights_dict)
            self.set_linear_weight(proj, weights_dict)
            for i in range(num_layers):
                path = f"{audio_path}layers.{i}."
                self.set_linear_weight(f"{path}self_attn.k_proj", weights_dict)
                self.set_linear_weight(f"{path}self_attn.v_proj", weights_dict)
                self.set_linear_weight(f"{path}self_attn.q_proj", weights_dict)
                self.set_linear_weight(f"{path}self_attn.out_proj", weights_dict)
                self.set_common_weight(f"{path}self_attn_layer_norm", weights_dict)
                self.set_linear_weight(f"{path}fc1", weights_dict)
                self.set_linear_weight(f"{path}fc2", weights_dict)
                self.set_common_weight(f"{path}final_layer_norm", weights_dict)
            # save weights
            np.savez(audio_npz, **weights_dict)

        # create mlir file
        in_shape = [1, num_mel_bins, n_window * 2]  # [1, 128, 200]
        out_shape = [1, n_window // 2, self.hidden_size]  # [50, hidden_size]
        input_types = ['F32']

        audio_mlir = MLIRImporter([in_shape], [out_shape],
                                  "audio",
                                  Platform.LLM,
                                  input_types,
                                  weight_file=audio_npz)
        ip = audio_mlir.insert_point

        def T(shape: list):
            return audio_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, audio_mlir)

        in0_op = audio_mlir.create_input_op(L('input_states'), 0)
        weight_op = audio_mlir.create_weight_op(conv1 + ".weight", [d_model, num_mel_bins, 3])
        bias_op = audio_mlir.create_weight_op(conv1 + ".bias", [d_model])
        conv1_op = top.ConvOp(T([1, d_model, n_window * 2]),
                              in0_op,
                              weight_op,
                              bias_op,
                              kernel_shape=[3, 1],
                              strides=[1, 1],
                              dilations=[1, 1],
                              pads=[1, 0, 1, 0],
                              loc=L(conv1),
                              ip=ip).output
        new_op = self.activate(audio_mlir, conv1_op, ActType.GELU, conv1)
        weight_op = audio_mlir.create_weight_op(conv2 + ".weight", [d_model, d_model, 3])
        bias_op = audio_mlir.create_weight_op(conv2 + ".bias", [d_model])
        conv2_op = top.ConvOp(T([1, d_model, n_window]),
                              new_op,
                              weight_op,
                              bias_op,
                              kernel_shape=[3, 1],
                              strides=[2, 1],
                              dilations=[1, 1],
                              pads=[1, 0, 1, 0],
                              loc=L(conv2),
                              ip=ip).output
        new_op = self.activate(audio_mlir, conv2_op, ActType.GELU, conv2)
        new_op = top.PermuteOp(T([1, n_window, d_model]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L(conv2 + ".permute"),
                               ip=ip).output
        weight_op = audio_mlir.create_weight_op(position_embedding + ".weight",
                                                [1, n_window, d_model])
        new_op = top.AddOp(T([1, n_window, d_model]), [new_op, weight_op],
                           loc=L(position_embedding),
                           ip=ip).output

        def audio_block(id: int, in_op):
            path = f"{audio_path}layers.{id}."
            norm1 = f"{path}self_attn_layer_norm"
            attn_q = f"{path}self_attn.q_proj"
            attn_k = f"{path}self_attn.k_proj"
            attn_v = f"{path}self_attn.v_proj"
            attn_out = f"{path}self_attn.out_proj"
            fc1 = f"{path}fc1"
            fc2 = f"{path}fc2"
            norm2 = f"{path}final_layer_norm"

            def audio_attention(in_op):
                in_shape = in_op.type.shape  #[1, 100, 1280] (1, n_window, d_model)
                q_op = self.linear(audio_mlir, attn_q, in_op, [d_model, d_model], in_shape)
                k_op = self.linear(audio_mlir, attn_k, in_op, [d_model, d_model], in_shape)
                v_op = self.linear(audio_mlir, attn_v, in_op, [d_model, d_model], in_shape)
                qkv_shape = [1, n_window, num_heads, head_dim]
                q_op = top.ReshapeOp(T(qkv_shape), q_op, loc=L(attn_q + ".reshape"), ip=ip).output
                k_op = top.ReshapeOp(T(qkv_shape), k_op, loc=L(attn_k + ".reshape"), ip=ip).output
                v_op = top.ReshapeOp(T(qkv_shape), v_op, loc=L(attn_v + ".reshape"), ip=ip).output

                fa_op = top.FAttentionOp(T(in_shape),
                                         q_op,
                                         k_op,
                                         v_op,
                                         audio_mlir.none_op,
                                         audio_mlir.none_op,
                                         scale=head_dim**-0.5,
                                         batch=1,
                                         q_head=num_heads,
                                         kv_head=num_heads,
                                         dim=head_dim,
                                         mq=n_window,
                                         mk=n_window,
                                         loc=L(f"{path}fattention"),
                                         ip=ip).output
                out_op = self.linear(audio_mlir, attn_out, fa_op, [self.embed_dim, self.embed_dim],
                                     in_shape)
                return out_op

            def audio_mlp(in_op):
                in_shape = in_op.type.shape  # [1, n_window, d_model]
                act = audio_config.activation_function
                new_op = self.layer_norm(audio_mlir, in_op, norm2, eps=1e-5)

                fc1_op = self.linear(audio_mlir, fc1, new_op, [d_model, ffn_dim],
                                     [1, n_window, ffn_dim])
                act_op = self.activate(audio_mlir, fc1_op, act, fc1)
                fc2_op = self.linear(audio_mlir, fc2, act_op, [ffn_dim, d_model],
                                     [1, n_window, d_model])

                new_op = top.AddOp(T(in_shape), [in_op, fc2_op], loc=L(fc2 + ".add"), ip=ip).output
                return new_op

            new_op = self.layer_norm(audio_mlir, in_op, norm1, eps=1e-5)
            new_op = audio_attention(new_op)
            new_op = top.AddOp(T([1, n_window, d_model]), [in_op, new_op],
                               loc=L(norm1 + ".add"),
                               ip=ip).output
            new_op = audio_mlp(new_op)
            return new_op

        for id in range(num_layers):
            new_op = audio_block(id, new_op)

        new_op = top.PermuteOp(T([1, d_model, n_window]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L(f"{audio_path}output.permute1"),
                               ip=ip).output
        new_op = top.AvgPoolOp(T([1, d_model, n_window // 2]),
                               new_op,
                               kernel_shape=[2],
                               strides=[2],
                               pads=[0, 0, 0, 0],
                               loc=L(f"{audio_path}output.avgpool"),
                               ip=ip).output
        new_op = top.PermuteOp(T([1, n_window // 2, d_model]),
                               new_op,
                               order=[0, 2, 1],
                               loc=L(f"{audio_path}output.permute2"),
                               ip=ip).output
        new_op = self.layer_norm(audio_mlir, new_op, ln_post, eps=1e-5)
        new_op = self.linear(audio_mlir, proj, new_op, [d_model, self.hidden_size],
                             [1, n_window // 2, self.hidden_size])
        audio_mlir.create_return_op([new_op])
        mlir_txt = audio_mlir.print_module()
        with open(f"audio.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

    def compile_audio_tower(self):
        name = "audio"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        deploy_args.append(f'--quantize {self.half_precision_quantize}')
        deploy_args.append('--quant_output')
        if self.high_precision:
            deploy_args.append('--high_precision')
        self.add_task(deploy_args, f"{name}.log")
