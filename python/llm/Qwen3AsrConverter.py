# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
import torch.nn as nn


# from qwen_asr.core.transformers_backend.modeling_qwen3_asr import SinusoidsPositionEmbedding
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


class Qwen3AsrConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.all_gen_mlirs.append(self.gen_audio_tower)
        self.all_compiles.append(self.compile_audio_tower)

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = QWEN2_5O_INFO
        self.llm_config = config.thinker_config.text_config
        self.llm_type = self.llm_config.model_type
        if self.llm_type == "qwen3_asr_text":
            self.llm_type = "qwen3"

    def get_dtype(self):
        if hasattr(self.llm_config, "dtype"):
            dtype = self.llm_config.dtype
        elif hasattr(self.llm_config, "torch_dtype"):
            dtype = self.llm_config.torch_dtype
        else:
            dtype = None
        if dtype == None and hasattr(self.config.thinker_config, "dtype"):
            dtype = self.config.thinker_config.dtype
        return dtype

    def gen_audio_tower(self):
        audio_config = self.config.thinker_config.audio_config
        d_model = audio_config.d_model
        self.embed_dim = d_model
        downsample_hidden_size = audio_config.downsample_hidden_size
        num_mel_bins = audio_config.num_mel_bins
        n_window = audio_config.n_window
        out_window = (((n_window * 2 - 1) // 2) // 2) // 2 + 1
        max_source_positions = audio_config.max_source_positions
        num_heads = audio_config.encoder_attention_heads
        head_dim = d_model // num_heads
        ffn_dim = audio_config.encoder_ffn_dim
        num_layers = audio_config.num_hidden_layers
        tqdm.write(f"generate audio tower mlir ...")
        audio_path = "thinker.audio_tower."
        name = "audio"
        # create weights file
        audio_npz = "audio_tower_top_weights.npz"
        conv2d1 = f"{audio_path}conv2d1"
        conv2d2 = f"{audio_path}conv2d2"
        conv2d3 = f"{audio_path}conv2d3"
        conv_out = f"{audio_path}conv_out"
        ln_post = f"{audio_path}ln_post"
        proj1 = f"{audio_path}proj1"
        proj2 = f"{audio_path}proj2"
        position_embedding = f"{audio_path}position_embedding"
        weights_dict = {}

        def save_weights():
            self.set_common_weight(conv2d1, weights_dict)
            self.set_common_weight(conv2d2, weights_dict)
            self.set_common_weight(conv2d3, weights_dict)
            positional_embedding = SinusoidsPositionEmbedding(max_source_positions, d_model)
            pos_emb = positional_embedding.forward(out_window)
            weights_dict[position_embedding + ".weight"] = pos_emb.reshape(1, out_window,
                                                                           d_model).numpy()
            self.set_common_weight(ln_post, weights_dict)
            self.set_linear_weight(proj1, weights_dict)
            self.set_linear_weight(proj2, weights_dict)
            self.set_linear_weight(conv_out, weights_dict)
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
        in_shape = [1, 1, num_mel_bins, n_window * 2]  # [1, 1, 128, 100]
        out_shape = [1, out_window, self.hidden_size]  # [13, hidden_size]
        input_types = ['F32']

        audio_mlir = MLIRImporter([in_shape], [out_shape],
                                  name,
                                  self.platform,
                                  input_types,
                                  weight_file=f"../{audio_npz}")
        ip = audio_mlir.insert_point

        def T(shape: list):
            return audio_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, audio_mlir)

        in0_op = audio_mlir.create_input_op(L('input_states'), 0)
        weight_op = audio_mlir.create_weight_op(conv2d1 + ".weight",
                                                [downsample_hidden_size, 1, 3, 3])
        bias_op = audio_mlir.create_weight_op(conv2d1 + ".bias", [downsample_hidden_size])
        conv2d_height = (num_mel_bins - 1) // 2 + 1
        conv2d_width = (n_window * 2 - 1) // 2 + 1
        conv2d1_op = top.ConvOp(T([1, downsample_hidden_size, conv2d_height, conv2d_width]),
                                in0_op,
                                weight_op,
                                bias_op,
                                kernel_shape=[3, 3],
                                strides=[2, 2],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                loc=L(conv2d1),
                                ip=ip).output
        new_op = self.activate(audio_mlir, conv2d1_op, ActType.GELU, conv2d1)
        weight_op = audio_mlir.create_weight_op(
            conv2d2 + ".weight", [downsample_hidden_size, downsample_hidden_size, 3, 3])
        bias_op = audio_mlir.create_weight_op(conv2d2 + ".bias", [downsample_hidden_size])
        conv2d_height = (conv2d_height - 1) // 2 + 1
        conv2d_width = (conv2d_width - 1) // 2 + 1
        conv2d2_op = top.ConvOp(T([1, downsample_hidden_size, conv2d_height, conv2d_width]),
                                new_op,
                                weight_op,
                                bias_op,
                                kernel_shape=[3, 3],
                                strides=[2, 2],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                loc=L(conv2d2),
                                ip=ip).output
        new_op = self.activate(audio_mlir, conv2d2_op, ActType.GELU, conv2d2)
        weight_op = audio_mlir.create_weight_op(
            conv2d3 + ".weight", [downsample_hidden_size, downsample_hidden_size, 3, 3])
        bias_op = audio_mlir.create_weight_op(conv2d3 + ".bias", [downsample_hidden_size])
        conv2d_height = (conv2d_height - 1) // 2 + 1
        conv2d_width = (conv2d_width - 1) // 2 + 1
        assert conv2d_width == out_window, f"conv2d_width {conv2d_width} should be equal to out_window {out_window}"
        conv2d3_op = top.ConvOp(T([1, downsample_hidden_size, conv2d_height, conv2d_width]),
                                new_op,
                                weight_op,
                                bias_op,
                                kernel_shape=[3, 3],
                                strides=[2, 2],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                loc=L(conv2d3),
                                ip=ip).output
        new_op = self.activate(audio_mlir, conv2d3_op, ActType.GELU, conv2d3)
        new_op = top.PermuteOp(T([1, out_window, downsample_hidden_size, conv2d_height]),
                               new_op,
                               order=[0, 3, 1, 2],
                               loc=L(conv2d3 + ".permute"),
                               ip=ip).output
        new_op = top.ReshapeOp(T([1, out_window, downsample_hidden_size * conv2d_height]),
                               new_op,
                               loc=L(conv2d3 + ".reshape"),
                               ip=ip).output
        new_op = self.linear(audio_mlir, conv_out, new_op,
                             [downsample_hidden_size * conv2d_height, d_model],
                             [1, out_window, d_model])
        weight_op = audio_mlir.create_weight_op(position_embedding + ".weight",
                                                [1, out_window, d_model])
        new_op = top.AddOp(T([1, out_window, d_model]), [new_op, weight_op],
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
                in_shape = in_op.type.shape  #[1, 13, 1024] (1, out_window, d_model)
                q_op = self.linear(audio_mlir, attn_q, in_op, [d_model, d_model], in_shape)
                k_op = self.linear(audio_mlir, attn_k, in_op, [d_model, d_model], in_shape)
                v_op = self.linear(audio_mlir, attn_v, in_op, [d_model, d_model], in_shape)
                qkv_shape = [1, out_window, num_heads, head_dim]
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
                                         mq=out_window,
                                         mk=out_window,
                                         keep_dims=False,
                                         loc=L(f"{path}fattention"),
                                         ip=ip).output
                out_op = self.linear(audio_mlir, attn_out, fa_op, [self.embed_dim, self.embed_dim],
                                     in_shape)
                return out_op

            def audio_mlp(in_op):
                in_shape = in_op.type.shape  # [1, out_window, d_model]
                act = audio_config.activation_function
                new_op = self.layer_norm(audio_mlir, in_op, norm2, eps=1e-5)

                fc1_op = self.linear(audio_mlir, fc1, new_op, [d_model, ffn_dim],
                                     [1, out_window, ffn_dim])
                act_op = self.activate(audio_mlir, fc1_op, act, fc1)
                fc2_op = self.linear(audio_mlir, fc2, act_op, [ffn_dim, d_model],
                                     [1, out_window, d_model])

                new_op = top.AddOp(T(in_shape), [in_op, fc2_op], loc=L(fc2 + ".add"), ip=ip).output
                return new_op

            new_op = self.layer_norm(audio_mlir, in_op, norm1, eps=1e-5)
            new_op = audio_attention(new_op)
            new_op = top.AddOp(T([1, out_window, d_model]), [in_op, new_op],
                               loc=L(norm1 + ".add"),
                               ip=ip).output
            new_op = audio_mlp(new_op)
            return new_op

        for id in range(num_layers):
            new_op = audio_block(id, new_op)

        new_op = self.layer_norm(audio_mlir, new_op, ln_post, eps=1e-5)
        new_op = self.linear(audio_mlir, proj1, new_op, [d_model, d_model],
                             [1, out_window, d_model])
        new_op = self.activate(audio_mlir, new_op, ActType.GELU, proj1)
        new_op = self.linear(audio_mlir, proj2, new_op, [d_model, self.hidden_size],
                             [1, out_window, self.hidden_size])
        audio_mlir.create_return_op([new_op])
        mlir_txt = audio_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

    def compile_audio_tower(self):
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
