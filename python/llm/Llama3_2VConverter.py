# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class MllamaRotaryEmbedding(torch.nn.Module):

    def __init__(self, config: MllamaTextConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    # @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, position_ids):
        inv_freq_expanded = self.inv_freq[None, :,
                                          None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)


class Llama3_2VConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)

        self.do_vit = True
        # vision config
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.max_num_tiles = self.vconfig.max_num_tiles
        self.vision_output_dim = self.vconfig.vision_output_dim
        self.depth = self.vconfig.num_hidden_layers
        self.global_depth = self.vconfig.num_global_layers
        self.num_patches = (self.vconfig.image_size // self.patch_size)**2 + 1
        self.embed_dim = self.vconfig.hidden_size
        self.vnum_heads = self.vconfig.attention_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [1, self.seq_length]
        self.vit_hidden_size = self.vconfig.hidden_size
        self.max_aspect_ratio_id = len(self.vconfig.supported_aspect_ratios)
        self.vnorm_eps = self.vconfig.norm_eps

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = MLLAMA_INFO
        self.llm_type = LlmType.MLLAMA

    @override
    def rotary_embedding(self):
        rotary_embed = MllamaRotaryEmbedding(config=self.llm_config, device="cpu")
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(1, self.seq_length)
        cos, sin = rotary_embed(position_ids)
        return cos.numpy(), sin.numpy()

    @override
    def gen_embedding_lmhead_mlir(self):
        tqdm.write("generate embedding and lm_head mlir ...")
        embedding_path = self.model_info.weights[LlmList.EMBEDING] + ".weight"
        embedding_data = self.model.read(embedding_path)
        if self.embedding_disk:
            self.gen_embedding_bin(embedding_data)
        else:
            # read embedding weights
            embedding_weights = {embedding_path: embedding_data}
            embedding_npz = "embedding_top_weights.npz"
            np.savez(embedding_npz, **embedding_weights)
        # read lm_head weights
        lmhead = self.model_info.weights[LlmList.LMHEAD]
        lmhead_path = lmhead + ".weight"
        norm = self.model_info.weights[LlmList.NORM]
        norm_path = norm + ".weight"
        if self.tie_word_embeddings:
            lmhead_data = embedding_data
        else:
            lmhead_data = self.model.read(lmhead_path)
        if not self.do_lmhead_merge:
            lmhead_data = np.ascontiguousarray(np.transpose(lmhead_data, (1, 0)))
            norm_data = self.model.read(norm_path)
            lmhead_weights = {lmhead_path: lmhead_data, norm_path: norm_data}
        else:
            lmhead_weights = {lmhead_path: lmhead_data}

        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **lmhead_weights)

        # gen embedding mlir
        def gen_embedding_by_length(name: str, seq_length: int):
            out_shape = [1, seq_length, self.hidden_size]
            embedding_mlir = MLIRImporter([[1, seq_length]], [out_shape],
                                          name,
                                          Platform.LLM,
                                          input_types=["INT32"],
                                          weight_file=embedding_npz)
            input_op = embedding_mlir.create_input_op(self.get_loc("input_ids", embedding_mlir), 0)
            weight_op = embedding_mlir.create_weight_op(embedding_path,
                                                        [self.vocab_size + 8, self.hidden_size])
            new_op = top.GatherOp(embedding_mlir.get_tensor_type(out_shape),
                                  weight_op,
                                  input_op,
                                  axis=0,
                                  loc=self.get_loc(name, embedding_mlir),
                                  ip=embedding_mlir.insert_point).output
            if self.llm_type in [LlmType.GEMMA3]:
                new_op = top.MulConstOp(embedding_mlir.get_tensor_type(out_shape),
                                        new_op,
                                        const_val=self.hidden_size**0.5,
                                        loc=self.get_loc(name + ".scale", embedding_mlir),
                                        ip=embedding_mlir.insert_point).output
            if self.llm_type == LlmType.MINICPM4:
                new_op = top.MulConstOp(embedding_mlir.get_tensor_type(out_shape),
                                        new_op,
                                        const_val=self.scale_emb,
                                        loc=self.get_loc(name + ".scale", embedding_mlir),
                                        ip=embedding_mlir.insert_point).output
            embedding_mlir.create_return_op([new_op])
            mlir_txt = embedding_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # gen lm_head mlir
        def gen_lm_head():
            out_shape = [[1, self.vocab_size]]
            if self.lmhead_with_topk:
                out_shape = [[1, 1]]
            lmhead_mlir = MLIRImporter([[1, self.hidden_size]],
                                       out_shape,
                                       "lm_head",
                                       Platform.LLM,
                                       weight_file=lmhead_npz)
            input_op = lmhead_mlir.create_input_op(self.get_loc("hidden_states", lmhead_mlir), 0)
            if not self.do_lmhead_merge:
                weight_op = lmhead_mlir.create_weight_op(norm_path, [1, self.hidden_size])
                input_op = self.rms_norm(lmhead_mlir, input_op, norm)
                if self.llm_type == LlmType.MINICPM4:
                    input_op = top.MulConstOp(lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                                              input_op,
                                              const_val=self.dim_model_base / self.hidden_size,
                                              loc=self.get_loc(lmhead + ".scale", lmhead_mlir),
                                              ip=lmhead_mlir.insert_point).output
                w_shape = [self.hidden_size, self.vocab_size]
                lmhead_op = self.linear(lmhead_mlir, lmhead, input_op, w_shape,
                                        [1, self.vocab_size])
            else:
                w_shape = [self.vocab_size, self.hidden_size]
                weight_op = lmhead_mlir.create_weight_op(lmhead + ".weight", w_shape)
                lmhead_op = top.MatMulOp(lmhead_mlir.get_tensor_type([self.vocab_size, 1]),
                                         weight_op,
                                         input_op,
                                         lmhead_mlir.none_op,
                                         do_relu=False,
                                         right_transpose=True,
                                         loc=self.get_loc(lmhead, lmhead_mlir),
                                         ip=lmhead_mlir.insert_point).output
                lmhead_op = top.ReshapeOp(lmhead_mlir.get_tensor_type([1, self.vocab_size]),
                                          lmhead_op,
                                          loc=self.get_loc(lmhead + ".reshape", lmhead_mlir),
                                          ip=lmhead_mlir.insert_point).output
            if self.lmhead_with_topk:
                topk_op = top.TopKOp(*lmhead_mlir.get_tensor_type([[1, 1], [1, 1]]),
                                     lmhead_op,
                                     axis=1,
                                     K=1,
                                     loc=self.get_loc(["token_value", "token_id"], lmhead_mlir),
                                     ip=lmhead_mlir.insert_point)
                # topk_op.values, topk_op.indices
                lmhead_mlir.create_return_op([topk_op.indices])
            else:
                lmhead_mlir.create_return_op([lmhead_op])

            mlir_txt = lmhead_mlir.print_module()
            with open("lm_head.mlir", "w") as f:
                f.write(mlir_txt)

        if not self.embedding_disk:
            gen_embedding_by_length("embedding", self.max_input_length)
            gen_embedding_by_length("embedding_cache", 1)
        gen_lm_head()

    @override
    def apply_rotary_pos(self,
                         mlir_gen,
                         pos_op,
                         q_op,
                         k_op,
                         rotary_cos: str,
                         rotary_sin: str,
                         decode: bool = False):
        dim = 1 if decode else self.seq_length

        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight",
                                              [self.seq_length, self.head_dim])
        cos_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, self.head_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_cos, mlir_gen),
                              ip=mlir_gen.insert_point).output
        un_cos_op = top.UnsqueezeOp(mlir_gen.get_tensor_type([1, 1, dim, self.head_dim]),
                                    cos_op,
                                    axes=[0],
                                    loc=self.get_loc(rotary_cos + "_unsqueeze", mlir_gen),
                                    ip=mlir_gen.insert_point).output
        cos_per_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                                   un_cos_op,
                                   order=[0, 2, 1, 3],
                                   loc=self.get_loc(rotary_cos + "_permute", mlir_gen),
                                   ip=mlir_gen.insert_point).output

        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight",
                                              [self.seq_length, self.head_dim])
        sin_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, self.head_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_sin, mlir_gen),
                              ip=mlir_gen.insert_point).output
        un_sin_op = top.UnsqueezeOp(mlir_gen.get_tensor_type([1, 1, dim, self.head_dim]),
                                    sin_op,
                                    axes=[0],
                                    loc=self.get_loc(rotary_sin + "_unsqueeze", mlir_gen),
                                    ip=mlir_gen.insert_point).output
        sin_per_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                                   un_sin_op,
                                   order=[0, 2, 1, 3],
                                   loc=self.get_loc(rotary_sin + "_permute", mlir_gen),
                                   ip=mlir_gen.insert_point).output

        # ===== q_proj rotary ========
        q_op = self.rotary_pos(mlir_gen, q_op, cos_per_op, sin_per_op, "q_proj")

        # ===== k_proj rotary ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_per_op, sin_per_op, "past_k_Add")
        return q_op, k_op

    def vision_block(self, vit_mlir, id: int, in_op, mask_op, is_gated: bool,
                     num_padding_patches: int):
        if not is_gated:
            norm1 = f"vision_model.transformer.layers.{id}.input_layernorm"
            attn_q = f"vision_model.transformer.layers.{id}.self_attn.q_proj"
            attn_k = f"vision_model.transformer.layers.{id}.self_attn.k_proj"
            attn_v = f"vision_model.transformer.layers.{id}.self_attn.v_proj"
            attn_proj = f"vision_model.transformer.layers.{id}.self_attn.o_proj"
            norm2 = f"vision_model.transformer.layers.{id}.post_attention_layernorm"
            mlp_fc1 = f"vision_model.transformer.layers.{id}.mlp.fc1"
            mlp_fc2 = f"vision_model.transformer.layers.{id}.mlp.fc2"
        else:
            norm1 = f"vision_model.global_transformer.layers.{id}.input_layernorm"
            attn_q = f"vision_model.global_transformer.layers.{id}.self_attn.q_proj"
            attn_k = f"vision_model.global_transformer.layers.{id}.self_attn.k_proj"
            attn_v = f"vision_model.global_transformer.layers.{id}.self_attn.v_proj"
            attn_proj = f"vision_model.global_transformer.layers.{id}.self_attn.o_proj"
            norm2 = f"vision_model.global_transformer.layers.{id}.post_attention_layernorm"
            gate_ffn = f"vision_model.global_transformer.layers.{id}.gate_ffn"
            gate_attn = f"vision_model.global_transformer.layers.{id}.gate_attn"
            mlp_fc1 = f"vision_model.global_transformer.layers.{id}.mlp.fc1"
            mlp_fc2 = f"vision_model.global_transformer.layers.{id}.mlp.fc2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_attention(in_op):
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, eps=self.vnorm_eps)
            hidden_shape = [
                1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.embed_dim
            ]
            q_op = self.linear(vit_mlir, attn_q, norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape)
            k_op = self.linear(vit_mlir, attn_k, norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape)
            v_op = self.linear(vit_mlir, attn_v, norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape)
            qk_shape = [
                1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.vnum_heads,
                self.vhead_dim
            ]
            q_op = top.ReshapeOp(T(qk_shape), q_op, loc=L(attn_q + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(qk_shape), k_op, loc=L(attn_k + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(qk_shape), v_op, loc=L(attn_v + ".reshape"), ip=ip).output

            fa_op = top.FAttentionOp(
                T([
                    1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.embed_dim
                ]),
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
                mq=(self.num_patches + num_padding_patches) * self.max_num_tiles,
                mk=(self.num_patches + num_padding_patches) * self.max_num_tiles,
                loc=L(attn_proj + f"attn_proj.{id}.fattention"),
                ip=ip).output

            out_op = self.linear(
                vit_mlir, attn_proj, fa_op, [self.embed_dim, self.embed_dim],
                [1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.embed_dim])
            if is_gated:
                gate_op = vit_mlir.create_weight_op(gate_attn, [1])
                gate_tan_op = top.TanhOp(T([1]), gate_op, loc=L(gate_attn + ".gate_tanh"),
                                         ip=ip).output
                out_op = top.MulOp(T(hidden_shape), [gate_tan_op, out_op],
                                   loc=L(gate_attn + ".mul"),
                                   ip=ip).output
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        def vision_mlp(in_op):
            in_shape = [
                1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.embed_dim
            ]

            new_op = self.layer_norm(vit_mlir, in_op, norm2, eps=self.vnorm_eps)

            gate_op = self.linear(vit_mlir, mlp_fc1, new_op,
                                  [self.embed_dim, self.vintermediate_size], [
                                      1,
                                      (self.num_patches + num_padding_patches) * self.max_num_tiles,
                                      self.vintermediate_size
                                  ])
            act_op = self.activate(vit_mlir, gate_op, self.vconfig.hidden_act, mlp_fc1)
            up_op = self.linear(
                vit_mlir, mlp_fc2, act_op, [self.vintermediate_size, self.embed_dim],
                [1, (self.num_patches + num_padding_patches) * self.max_num_tiles, self.embed_dim])
            if is_gated:
                gate_op = vit_mlir.create_weight_op(gate_ffn, [1])
                gate_tan_op = top.TanhOp(T([1]), gate_op, loc=L(gate_ffn + ".gate_tanh"),
                                         ip=ip).output
                up_op = top.MulOp(T(in_shape), [gate_tan_op, up_op],
                                  loc=L(gate_ffn + ".mul"),
                                  ip=ip).output

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
        patch_embed = "vision_model.patch_embedding"
        class_embedding = "vision_model.class_embedding"
        post_tile_embedding = "vision_model.post_tile_positional_embedding"
        pre_tile_embedding = "vision_model.pre_tile_positional_embedding"
        gated_embedding = "vision_model.gated_positional_embedding"
        layernorm_post = "vision_model.layernorm_post"
        multi_modal_projector = "multi_modal_projector"

        def save_weights():
            weights_dict = {}
            self.set_common_weight(patch_embed, weights_dict)
            data = self.model.read(pre_tile_embedding + ".embedding.weight").reshape(
                self.max_aspect_ratio_id + 1, self.max_num_tiles * self.vit_hidden_size)
            weights_dict[pre_tile_embedding + ".embedding.weight"] = data
            data = self.model.read(pre_tile_embedding + ".gate")
            weights_dict[pre_tile_embedding + ".gate"] = data
            data = self.model.read(post_tile_embedding + ".embedding.weight").reshape(
                self.max_aspect_ratio_id + 1, self.max_num_tiles * self.vit_hidden_size)
            weights_dict[post_tile_embedding + ".embedding.weight"] = data
            data = self.model.read(post_tile_embedding + ".gate")
            weights_dict[post_tile_embedding + ".gate"] = data
            data = self.model.read(gated_embedding + ".embedding")
            weights_dict[gated_embedding + ".embedding"] = data
            data = self.model.read(gated_embedding + ".gate")
            weights_dict[gated_embedding + ".gate"] = data
            data = self.model.read(gated_embedding + ".tile_embedding.weight")
            weights_dict[gated_embedding + ".tile_embedding.weight"] = data
            layernorm_pre = "vision_model.layernorm_pre"
            self.set_linear_weight(multi_modal_projector, weights_dict)
            self.set_common_weight(layernorm_pre, weights_dict)
            self.set_common_weight(layernorm_post, weights_dict)
            data = self.model.read(class_embedding)
            weights_dict[class_embedding] = data

            for i in range(self.depth):
                self.set_linear_weight(f"vision_model.transformer.layers.{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(f"vision_model.transformer.layers.{i}.mlp.fc2", weights_dict)
                self.set_common_weight(
                    f"vision_model.transformer.layers.{i}.post_attention_layernorm", weights_dict)
                self.set_common_weight(f"vision_model.transformer.layers.{i}.input_layernorm",
                                       weights_dict)

                data = self.model.read(
                    f"vision_model.transformer.layers.{i}.self_attn.q_proj.weight")
                weights_dict[
                    f"vision_model.transformer.layers.{i}.self_attn.q_proj.weight"] = np.ascontiguousarray(
                        np.transpose(data, (1, 0)))
                data = self.model.read(
                    f"vision_model.transformer.layers.{i}.self_attn.k_proj.weight")
                weights_dict[
                    f"vision_model.transformer.layers.{i}.self_attn.k_proj.weight"] = np.ascontiguousarray(
                        np.transpose(data, (1, 0)))
                data = self.model.read(
                    f"vision_model.transformer.layers.{i}.self_attn.v_proj.weight")
                weights_dict[
                    f"vision_model.transformer.layers.{i}.self_attn.v_proj.weight"] = np.ascontiguousarray(
                        np.transpose(data, (1, 0)))
                data = self.model.read(
                    f"vision_model.transformer.layers.{i}.self_attn.o_proj.weight")
                weights_dict[
                    f"vision_model.transformer.layers.{i}.self_attn.o_proj.weight"] = np.ascontiguousarray(
                        np.transpose(data, (1, 0)))

                if i < self.global_depth:
                    self.set_linear_weight(f"vision_model.global_transformer.layers.{i}.mlp.fc1",
                                           weights_dict)
                    self.set_linear_weight(f"vision_model.global_transformer.layers.{i}.mlp.fc2",
                                           weights_dict)
                    self.set_common_weight(
                        f"vision_model.global_transformer.layers.{i}.post_attention_layernorm",
                        weights_dict)
                    self.set_common_weight(
                        f"vision_model.global_transformer.layers.{i}.input_layernorm", weights_dict)
                    self.set_common_weight(f"vision_model.global_transformer.layers.{i}.gate_attn",
                                           weights_dict)
                    self.set_common_weight(f"vision_model.global_transformer.layers.{i}.gate_ffn",
                                           weights_dict)
                    self.set_linear_weight(
                        f"vision_model.global_transformer.layers.{i}.self_attn.q_proj",
                        weights_dict)
                    self.set_linear_weight(
                        f"vision_model.global_transformer.layers.{i}.self_attn.k_proj",
                        weights_dict)
                    self.set_linear_weight(
                        f"vision_model.global_transformer.layers.{i}.self_attn.v_proj",
                        weights_dict)
                    self.set_linear_weight(
                        f"vision_model.global_transformer.layers.{i}.self_attn.o_proj",
                        weights_dict)

            np.savez(vit_npz, **weights_dict)

        in_shape = [
            self.max_num_tiles, self.vconfig.num_channels, self.vconfig.image_size,
            self.vconfig.image_size
        ]
        aspect_ratio_ids_shape = [1, 1]
        aspect_ratio_ids = [[6]]
        aspect_ratio_mask_shape = [1, 1, self.max_num_tiles]
        aspect_ratio_mask = [[[1, 1, 1, 1]]]
        out_shape = [self.max_num_tiles, self.num_patches, self.hidden_size]

        input_shapes = [in_shape, aspect_ratio_ids_shape, aspect_ratio_mask_shape]
        input_types = ['F32', 'INT32', 'INT32']

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

        in0_op = vit_mlir.create_input_op(L('pixel_values'), 0)
        in1_op = vit_mlir.create_input_op(L('aspect_ratio_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('aspect_ratio_mask'), 2)

        patch_embeds_weight_op = vit_mlir.create_weight_op(
            patch_embed + ".weight", [self.vit_hidden_size, 3, self.patch_size, self.patch_size])
        conv_op = top.ConvOp(T([
            self.max_num_tiles, self.vit_hidden_size, self.vconfig.image_size // self.patch_size,
            self.vconfig.image_size // self.patch_size
        ]),
                             in0_op,
                             patch_embeds_weight_op,
                             vit_mlir.none_op,
                             kernel_shape=[self.patch_size, self.patch_size],
                             strides=[self.patch_size, self.patch_size],
                             pads=[0, 0, 0, 0],
                             dilations=[1, 1],
                             loc=L(patch_embed),
                             ip=ip).output
        self.num_patches -= 1
        reshape_op = top.ReshapeOp(
            T([self.max_num_tiles, self.vit_hidden_size, self.num_patches]),
            conv_op,
            shape=[self.max_num_tiles, self.vit_hidden_size, self.num_patches],
            loc=L(patch_embed + ".reshape"),
            ip=ip).output
        permute_op = top.PermuteOp(T([self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                                   reshape_op,
                                   order=[0, 2, 1],
                                   loc=L(patch_embed + ".permute"),
                                   ip=ip).output
        weight_op = vit_mlir.create_weight_op(
            pre_tile_embedding + ".embedding.weight",
            [self.max_aspect_ratio_id + 1, self.max_num_tiles * self.vit_hidden_size])
        embedding_op = top.GatherOp(T([1, self.max_num_tiles * self.vit_hidden_size]),
                                    weight_op,
                                    in1_op,
                                    axis=0,
                                    loc=L(pre_tile_embedding + ".embedding"),
                                    ip=ip).output
        embedding_reshape_op = top.ReshapeOp(T([self.max_num_tiles, 1, self.vit_hidden_size]),
                                             embedding_op,
                                             shape=[self.max_num_tiles, 1, self.vit_hidden_size],
                                             loc=L(pre_tile_embedding + ".reshape"),
                                             ip=ip).output
        gate_mulc_op = top.MulConstOp(T([self.max_num_tiles, 1, self.vit_hidden_size]),
                                      embedding_reshape_op,
                                      const_val=0.63514894247055054,
                                      loc=L(pre_tile_embedding + "_mulconst1"),
                                      ip=ip).output
        add_op = top.AddOp(T([self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                           [permute_op, gate_mulc_op],
                           loc=L(pre_tile_embedding + ".add"),
                           ip=ip).output
        weight_op2 = vit_mlir.create_weight_op(class_embedding, [self.vit_hidden_size])
        expand_op = top.UnsqueezeOp(T([1, 1, self.vit_hidden_size]),
                                    weight_op2,
                                    axes=[0, 1],
                                    loc=L(class_embedding + "_squeeze"),
                                    ip=ip).output
        tile_op = top.TileOp(T([self.max_num_tiles, 1, self.vit_hidden_size]),
                             expand_op,
                             tile=[self.max_num_tiles, 1, 1],
                             loc=L(class_embedding + "_tile1"),
                             ip=ip).output
        concat_op = top.ConcatOp(T([self.max_num_tiles, self.num_patches + 1,
                                    self.vit_hidden_size]), [tile_op, add_op],
                                 axis=1,
                                 loc=L(patch_embed + ".concat"),
                                 ip=ip).output
        self.num_patches += 1
        unsqueeze_op = top.UnsqueezeOp(T(
            [1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                                       concat_op,
                                       axes=[0],
                                       loc=L(gated_embedding + "_unsqueeze"),
                                       ip=ip).output
        gate_op = vit_mlir.create_weight_op(gated_embedding + ".gate", [1])
        gate_tan_op = top.TanhOp(T([1]), gate_op, loc=L(gated_embedding + ".gate_tanh"),
                                 ip=ip).output
        sub_op = top.SubConstOp(T([1]),
                                gate_tan_op,
                                const_val=1,
                                is_reverse=True,
                                loc=L(gated_embedding + ".sub_const"),
                                ip=ip).output
        weight_op = vit_mlir.create_weight_op(gated_embedding + ".embedding",
                                              [self.num_patches, self.vit_hidden_size])
        mul_op_1 = top.MulOp(T([self.num_patches, self.vit_hidden_size]), [sub_op, weight_op],
                             loc=L(gated_embedding + ".mul1"),
                             ip=ip).output
        reshape_op = top.ReshapeOp(T([1, 1, self.num_patches, self.vit_hidden_size]),
                                   mul_op_1,
                                   shape=[1, 1, self.num_patches, self.vit_hidden_size],
                                   loc=L(gated_embedding + ".reshape1"),
                                   ip=ip).output
        add_op = top.AddOp(T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                           [unsqueeze_op, reshape_op],
                           loc=L(gated_embedding + ".add1"),
                           ip=ip).output
        weight_op = vit_mlir.create_weight_op(gated_embedding + ".tile_embedding.weight", [
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.vit_hidden_size
        ])
        gather_op = top.GatherOp(T(
            [1, self.max_num_tiles * self.num_patches * self.vit_hidden_size]),
                                 weight_op,
                                 in1_op,
                                 axis=0,
                                 loc=L(gated_embedding + ".gather"),
                                 ip=ip).output
        reshape_op = top.ReshapeOp(
            T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
            gather_op,
            shape=[1, self.max_num_tiles, self.num_patches, self.vit_hidden_size],
            loc=L(gated_embedding + ".reshape2"),
            ip=ip).output
        mul_op = top.MulOp(T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                           [gate_tan_op, reshape_op],
                           loc=L(gated_embedding + ".mul2"),
                           ip=ip).output
        add_op = top.AddOp(T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                           [add_op, mul_op],
                           loc=L(gated_embedding + ".add2"),
                           ip=ip).output
        weight_op = vit_mlir.create_weight_op("vision_model.layernorm_pre.weight",
                                              [self.vit_hidden_size])
        bias_op = vit_mlir.create_weight_op("vision_model.layernorm_pre.bias",
                                            [self.vit_hidden_size])
        layernorm_pre_op = top.LayerNormOp(
            T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
            add_op,
            weight_op,
            bias_op,
            normalized_shape=[self.vit_hidden_size],
            axis=len([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]) - 1,
            eps=1e-5,
            loc=L("layernorm_pre_last"),
            ip=ip).output
        in2_reshape_op = top.ReshapeOp(T([1, self.max_num_tiles, 1, 1]),
                                       in2_op,
                                       shape=[1, self.max_num_tiles, 1, 1],
                                       loc=L("in2_op.reshape1"),
                                       ip=ip).output
        tile_op = top.TileOp(T([1, self.max_num_tiles, self.num_patches, 1]),
                             in2_reshape_op,
                             tile=[1, 1, self.num_patches, 1],
                             loc=L("in2_op.tile"),
                             ip=ip).output
        pad_op = top.PadOp(T([1, self.max_num_tiles, self.num_patches + 7, 1]),
                           tile_op,
                           paddings=[0, 0, 0, 0, 0, 0, 7, 0],
                           val=0,
                           mode=StringAttr.get("constant"),
                           loc=L("in2_op_padding_op"),
                           ip=ip).output
        sub_op = top.SubConstOp(T([1, self.max_num_tiles, self.num_patches + 7, 1]),
                                pad_op,
                                const_val=1,
                                is_reverse=True,
                                loc=L("in2_op.sub_const"),
                                ip=ip).output
        sub_reshape_op = top.ReshapeOp(T([1, self.max_num_tiles * (self.num_patches + 7), 1]),
                                       sub_op,
                                       shape=[1, self.max_num_tiles * (self.num_patches + 7), 1],
                                       loc=L(f"in2_op.reshape2"),
                                       ip=ip).output
        permute_op = top.PermuteOp(T([1, 1, self.max_num_tiles * (self.num_patches + 7)]),
                                   sub_reshape_op,
                                   order=[0, 2, 1],
                                   loc=L(f"in2_op.permute"),
                                   ip=ip).output
        sub_reshape_uns_op = top.UnsqueezeOp(T(
            [1, 1, self.max_num_tiles * (self.num_patches + 7), 1]),
                                             sub_reshape_op,
                                             axes=[0],
                                             loc=L("gated_embedding_in2_op_unsqueeze1"),
                                             ip=ip).output
        permute_uns_op = top.UnsqueezeOp(T([1, 1, 1, self.max_num_tiles * (self.num_patches + 7)]),
                                         permute_op,
                                         axes=[0],
                                         loc=L("gated_embedding_in2_op_unsqueeze2"),
                                         ip=ip).output
        mm_op = top.MatMulOp(T([
            1, 1, self.max_num_tiles * (self.num_patches + 7),
            self.max_num_tiles * (self.num_patches + 7)
        ]),
                             sub_reshape_uns_op,
                             permute_uns_op,
                             vit_mlir.none_op,
                             do_relu=False,
                             loc=L(gated_embedding + "_post_mask_matmul"),
                             ip=ip).output
        mulconst_op = top.MulConstOp(T([
            1, 1, self.max_num_tiles * (self.num_patches + 7),
            self.max_num_tiles * (self.num_patches + 7)
        ]),
                                     mm_op,
                                     const_val=torch.finfo(torch.float32).min,
                                     loc=L(f"in2_op.mulconst"),
                                     ip=ip).output

        num_padding_patches = (8 - (self.num_patches % 8)) % 8
        padding = [0, 0, 0, 0, 0, 0, num_padding_patches, 0]
        pad_op = top.PadOp(T(
            [1, self.max_num_tiles, self.num_patches + num_padding_patches, self.vit_hidden_size]),
                           layernorm_pre_op,
                           paddings=padding,
                           val=0,
                           mode=StringAttr.get("constant"),
                           loc=L("padding_op"),
                           ip=ip).output
        endcoder_rs_op = top.ReshapeOp(T([
            1, self.max_num_tiles * (self.num_patches + num_padding_patches), self.vit_hidden_size
        ]),
                                       pad_op,
                                       shape=[
                                           1, self.max_num_tiles *
                                           (self.num_patches + num_padding_patches),
                                           self.vit_hidden_size
                                       ],
                                       loc=L("endcoder_reshape"),
                                       ip=ip).output

        new_op = endcoder_rs_op
        select_ops = None
        select_ids = 0
        re_ops = []
        mask_op = mulconst_op
        for id in range(self.depth):
            new_op = self.vision_block(vit_mlir,
                                       id,
                                       new_op,
                                       mask_op,
                                       is_gated=False,
                                       num_padding_patches=num_padding_patches)

            if (id + 1) in self.vconfig.intermediate_layers_indices:
                new_uns_op = top.UnsqueezeOp(T([
                    1, self.max_num_tiles * (self.num_patches + num_padding_patches),
                    self.vit_hidden_size, 1
                ]),
                                             new_op,
                                             axes=[-1],
                                             loc=L(f"select_op.new_unsqueeze_{id}"),
                                             ip=ip).output
                if select_ops == None:
                    select_ops = new_uns_op
                else:
                    select_ops = top.ConcatOp(T([
                        1, self.max_num_tiles * (self.num_patches + num_padding_patches),
                        self.vit_hidden_size, (select_ids + 1)
                    ]), [select_ops, new_uns_op],
                                              axis=3,
                                              loc=L(f"select_op.concat_{id}"),
                                              ip=ip).output
                select_ids += 1

        postln_op = self.layer_norm(vit_mlir, new_op, norm_path=layernorm_post, eps=self.vnorm_eps)
        postrs_op = top.ReshapeOp(T(
            [1, self.max_num_tiles, (self.num_patches + num_padding_patches),
             self.vit_hidden_size]),
                                  postln_op,
                                  shape=[
                                      1, self.max_num_tiles,
                                      (self.num_patches + num_padding_patches), self.vit_hidden_size
                                  ],
                                  loc=L("global_endcoder.reshape"),
                                  ip=ip).output
        weight_op = vit_mlir.create_weight_op(
            post_tile_embedding + ".embedding.weight",
            [self.max_aspect_ratio_id + 1, self.max_num_tiles * self.vit_hidden_size])
        post_tile_embedding_op = top.GatherOp(T([1, self.max_num_tiles * self.vit_hidden_size]),
                                              weight_op,
                                              in1_op,
                                              axis=0,
                                              loc=L(post_tile_embedding + ".embedding"),
                                              ip=ip).output
        embedding_reshape_op = top.ReshapeOp(T([1, self.max_num_tiles, 1, self.vit_hidden_size]),
                                             post_tile_embedding_op,
                                             shape=[1, self.max_num_tiles, 1, self.vit_hidden_size],
                                             loc=L(post_tile_embedding + ".reshape1"),
                                             ip=ip).output
        gate_op = vit_mlir.create_weight_op(post_tile_embedding + ".gate", [1])
        gate_tan_op = top.TanhOp(T([1]), gate_op, loc=L(post_tile_embedding + ".gate_tanh"),
                                 ip=ip).output
        mul_gate_op = top.MulOp(T([1, self.max_num_tiles, 1, self.vit_hidden_size]),
                                [embedding_reshape_op, gate_tan_op],
                                loc=L(post_tile_embedding + ".mul"),
                                ip=ip).output
        add_op = top.AddOp(T(
            [1, self.max_num_tiles, self.num_patches + num_padding_patches, self.vit_hidden_size]),
                           [postrs_op, mul_gate_op],
                           loc=L(post_tile_embedding + ".add"),
                           ip=ip).output
        new_op = top.ReshapeOp(T([
            1, self.max_num_tiles * (self.num_patches + num_padding_patches), self.vit_hidden_size
        ]),
                               add_op,
                               shape=[
                                   1, self.max_num_tiles * (self.num_patches + num_padding_patches),
                                   self.vit_hidden_size
                               ],
                               loc=L(post_tile_embedding + ".reshape2"),
                               ip=ip).output

        for id in range(self.global_depth):
            new_op = self.vision_block(vit_mlir,
                                       id,
                                       new_op,
                                       mask_op,
                                       is_gated=True,
                                       num_padding_patches=num_padding_patches)

        rm_pad_op = top.ReshapeOp(T(
            [1, self.max_num_tiles, (self.num_patches + num_padding_patches),
             self.vit_hidden_size]),
                                  new_op,
                                  shape=[
                                      1, self.max_num_tiles,
                                      (self.num_patches + num_padding_patches), self.vit_hidden_size
                                  ],
                                  loc=L("rm_pad.reshape"),
                                  ip=ip).output
        slice_op = top.SliceOp(T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size]),
                               rm_pad_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               vit_mlir.none_op,
                               offset=[0, 0, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=[1, self.max_num_tiles, self.num_patches, self.vit_hidden_size],
                               axes=[],
                               loc=L("rm_pad.slicex"),
                               ip=ip).output
        rm_ind_rs_op = top.ReshapeOp(T([
            1, self.max_num_tiles, self.num_patches + num_padding_patches,
            self.vit_hidden_size * select_ids
        ]),
                                     select_ops,
                                     shape=[
                                         1, self.max_num_tiles,
                                         (self.num_patches + num_padding_patches),
                                         self.vit_hidden_size * select_ids
                                     ],
                                     loc=L("rm_ind.reshape"),
                                     ip=ip).output
        rm_ind_rs_slice_op = top.SliceOp(
            T([1, self.max_num_tiles, self.num_patches, self.vit_hidden_size * select_ids]),
            rm_ind_rs_op,
            vit_mlir.none_op,
            vit_mlir.none_op,
            vit_mlir.none_op,
            offset=[0, 0, 0, 0],
            steps=[1, 1, 1, 1],
            ends=[1, self.max_num_tiles, self.num_patches, self.vit_hidden_size * select_ids],
            axes=[],
            loc=L("rm_ind_rs.slicex"),
            ip=ip).output
        concat_op1 = top.ConcatOp(T(
            [1, self.max_num_tiles, self.num_patches, self.vision_output_dim]),
                                  [slice_op, rm_ind_rs_slice_op],
                                  axis=3,
                                  loc=L("slice_op2.concat"),
                                  ip=ip).output
        mul_mod_weight = vit_mlir.create_weight_op(multi_modal_projector + ".weight",
                                                   [self.vision_output_dim, self.hidden_size])
        mul_mod_bias = vit_mlir.create_weight_op(multi_modal_projector + ".bias",
                                                 [self.hidden_size])
        mul_mod_bias_rs_op = top.ReshapeOp(T([1, 1, 1, self.hidden_size]),
                                           mul_mod_bias,
                                           shape=[1, 1, 1, self.hidden_size],
                                           loc=L(multi_modal_projector + "_bias_reshape"),
                                           ip=ip).output
        conc_mm_op = top.MatMulOp(T([1, self.max_num_tiles, self.num_patches, self.hidden_size]),
                                  concat_op1,
                                  mul_mod_weight,
                                  mul_mod_bias_rs_op,
                                  do_relu=False,
                                  loc=L(multi_modal_projector + "_matmul"),
                                  ip=ip).output
        mulmod_rs_op = top.ReshapeOp(T([self.max_num_tiles, self.num_patches, self.hidden_size]),
                                     conc_mm_op,
                                     shape=[self.max_num_tiles, self.num_patches, self.hidden_size],
                                     loc=L("cross_attention_states_Reshape"),
                                     ip=ip).output

        vit_mlir.create_return_op([mulmod_rs_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

        return mulmod_rs_op

    @override
    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        if idx not in self.llm_config.cross_attention_layers:
            q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
            k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
            v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
            o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        else:
            c_q_proj = TOP_PATH + self.model_info.weights[LlmList.C_Q_PROJ]
            c_q_norm = TOP_PATH + self.model_info.weights[LlmList.C_Q_NORM]
            c_k_proj = TOP_PATH + self.model_info.weights[LlmList.C_K_PROJ]
            c_k_norm = TOP_PATH + self.model_info.weights[LlmList.C_K_NORM]
            c_v_proj = TOP_PATH + self.model_info.weights[LlmList.C_V_PROJ]
            c_o_proj = TOP_PATH + self.model_info.weights[LlmList.C_O_PROJ]
            c_attn_gate = TOP_PATH + self.model_info.weights[LlmList.C_ATTN_GATE]
            c_mlp_gate = TOP_PATH + self.model_info.weights[LlmList.C_MLP_GATE]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.do_lmhead_merge and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict, WeightType.RMS_NORM)
        if idx not in self.llm_config.cross_attention_layers:
            self.set_linear_weight(q_proj, weight_dict)
            self.set_linear_weight(k_proj, weight_dict)
            self.set_linear_weight(v_proj, weight_dict)
            self.set_linear_weight(o_proj, weight_dict)
        else:
            self.set_linear_weight(c_q_proj, weight_dict)
            self.set_linear_weight(c_k_proj, weight_dict)
            self.set_linear_weight(c_v_proj, weight_dict)
            self.set_linear_weight(c_o_proj, weight_dict)
            self.set_common_weight(c_q_norm, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(c_k_norm, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(c_attn_gate, weight_dict)
            self.set_common_weight(c_mlp_gate, weight_dict)

        self.set_common_weight(post_attn_ln, weight_dict, WeightType.RMS_NORM)
        self.set_linear_weight(mlp_gate, weight_dict)
        self.set_linear_weight(mlp_up, weight_dict)
        self.set_linear_weight(mlp_down, weight_dict)
        if do_norm:
            self.set_common_weight(norm, weight_dict, WeightType.RMS_NORM)

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

            last_name = "hidden_states_Add"
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        def gen_mlp_cross(mlir_gen, input_shape, in_op):
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

            return down_op

        # create block mlir
        def gen_block():
            name = f"block_{idx}"
            input_len = self.seq_length
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
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("past_v_Reshape"), ip=ip).output

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

            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # create cross_block mlir
        def gen_cross_block():
            name = f"block_{idx}"
            input_len = self.seq_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = [self.max_num_tiles, self.num_patches, self.hidden_size]
            row_mask = [1, self.seq_length, 1]
            mask_shape = [1, 1, input_len, self.max_num_tiles * self.num_patches]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            past_kv_shape = [
                1, self.num_attention_heads, self.max_num_tiles * self.num_patches, self.head_dim
            ]
            # past_kv_shape = [1, self.max_num_tiles * self.num_patches, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter([input_shape, id_shape, row_mask, mask_shape],
                                      [input_shape, past_kv_shape, past_kv_shape],
                                      name,
                                      Platform.LLM, ["F32", "F32", "F32", "F32"],
                                      weight_file=weight_file)

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("cross_attention_states"), 1)
            in2_op = block_mlir.create_input_op(L("text_row_mask"), 2)
            in3_op = block_mlir.create_input_op(L("attention_mask"), 3)

            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir, c_q_proj, ln_op, [self.hidden_size, q_dim],
                               [1, input_len, q_dim])
            # k_proj
            kv_dim = self.num_key_value_heads * self.head_dim
            k_op = self.linear(block_mlir, c_k_proj, in1_op, [self.hidden_size, kv_dim],
                               [self.max_num_tiles, self.num_patches, kv_dim])
            k_rs_op = top.ReshapeOp(T(
                [1, self.max_num_tiles * self.num_patches, self.num_key_value_heads,
                 self.head_dim]),
                                    k_op,
                                    shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                    loc=L(c_k_proj + "_reshape_op"),
                                    ip=ip).output
            k_rs_uns_op = top.UnsqueezeOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads, 1, self.head_dim
            ]),
                                          k_rs_op,
                                          axes=[3],
                                          loc=L(c_k_proj + "_unsqueeze0"),
                                          ip=ip).output
            k_rs_uns_t_op = top.TileOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads, 4, self.head_dim
            ]),
                                       k_rs_uns_op,
                                       tile=[1, 1, 1, 4, 1],
                                       loc=L(c_k_proj + "_tile0"),
                                       ip=ip).output
            k_rs_uns_t_op2 = top.ReshapeOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads * 4,
                self.head_dim
            ]),
                                           k_rs_uns_t_op,
                                           shape=[
                                               1, self.max_num_tiles * self.num_patches,
                                               self.num_key_value_heads * 4, self.head_dim
                                           ],
                                           loc=L(c_k_proj + "_reshape_op1"),
                                           ip=ip).output
            k_op = top.PermuteOp(T(
                [1, self.num_attention_heads, self.max_num_tiles * self.num_patches,
                 self.head_dim]),
                                 k_rs_uns_t_op2,
                                 order=[0, 2, 1, 3],
                                 loc=L(c_k_proj + "_permute0"),
                                 ip=ip).output

            # v_proj
            v_op = self.linear(block_mlir, c_v_proj, in1_op, [self.hidden_size, kv_dim],
                               [self.max_num_tiles, self.num_patches, kv_dim])
            v_rs_op = top.ReshapeOp(T(
                [1, self.max_num_tiles * self.num_patches, self.num_key_value_heads,
                 self.head_dim]),
                                    v_op,
                                    shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                    loc=L(c_v_proj + "_reshape_op"),
                                    ip=ip).output
            v_rs_uns_op = top.UnsqueezeOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads, 1, self.head_dim
            ]),
                                          v_rs_op,
                                          axes=[3],
                                          loc=L(c_v_proj + "_unsqueeze0"),
                                          ip=ip).output
            v_rs_uns_t_op = top.TileOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads, 4, self.head_dim
            ]),
                                       v_rs_uns_op,
                                       tile=[1, 1, 1, 4, 1],
                                       loc=L(c_v_proj + "_tile0"),
                                       ip=ip).output
            v_rs_uns_t_op2 = top.ReshapeOp(T([
                1, self.max_num_tiles * self.num_patches, self.num_key_value_heads * 4,
                self.head_dim
            ]),
                                           v_rs_uns_t_op,
                                           shape=[
                                               1, self.max_num_tiles * self.num_patches,
                                               self.num_key_value_heads * 4, self.head_dim
                                           ],
                                           loc=L(c_v_proj + "_reshape_op1"),
                                           ip=ip).output
            v_op = top.PermuteOp(T(
                [1, self.num_attention_heads, self.max_num_tiles * self.num_patches,
                 self.head_dim]),
                                 v_rs_uns_t_op2,
                                 order=[0, 2, 1, 3],
                                 loc=L(c_v_proj + "_permute0"),
                                 ip=ip).output

            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(c_q_proj + ".reshape"), ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, c_q_norm)
            k_op = self.rms_norm(block_mlir, k_op, c_k_norm)

            return_ops.append(k_op)
            return_ops.append(v_op)
            # ======= fattention =========
            # fa_op = top.FAttentionOp(T([1, input_len, q_dim]),
            #                          q_op,
            #                          k_rs_op,
            #                          v_rs_op,
            #                          in2_op,
            #                          in3_op,
            #                          scale=self.head_dim**-0.5,
            #                          batch=1,
            #                          q_head=self.num_attention_heads,
            #                          kv_head=self.num_key_value_heads,
            #                          dim=self.head_dim,
            #                          mq=input_len,
            #                          mk=self.max_num_tiles * self.num_patches,
            #                          loc=L(TOP_PATH + "fattention"),
            #                          ip=ip).output
            q_perm_op = top.PermuteOp(T([1, self.num_attention_heads, input_len, self.head_dim]),
                                      q_op,
                                      order=[0, 2, 1, 3],
                                      loc=L(TOP_PATH + "fattention_permute0"),
                                      ip=ip).output
            k_perm_op = top.PermuteOp(T(
                [1, self.num_attention_heads, self.head_dim,
                 self.max_num_tiles * self.num_patches]),
                                      k_op,
                                      order=[0, 1, 3, 2],
                                      loc=L(TOP_PATH + "fattention_permute1"),
                                      ip=ip).output
            qk_mm_op = top.MatMulOp(T(
                [1, self.num_attention_heads, input_len, self.max_num_tiles * self.num_patches]),
                                    q_perm_op,
                                    k_perm_op,
                                    block_mlir.none_op,
                                    do_relu=False,
                                    loc=L(TOP_PATH + "fattention_matmul0"),
                                    ip=ip).output
            qk_mm_const_op = top.MulConstOp(T(
                [1, self.num_attention_heads, input_len, self.max_num_tiles * self.num_patches]),
                                            qk_mm_op,
                                            const_val=self.head_dim**-0.5,
                                            loc=L(TOP_PATH + "fattention_mulconst0"),
                                            ip=ip).output
            qk_mm_add_op = top.AddOp(T(
                [1, self.num_attention_heads, input_len, self.max_num_tiles * self.num_patches]),
                                     [qk_mm_const_op, in3_op],
                                     loc=L(TOP_PATH + "fattention_add0"),
                                     ip=ip).output
            qk_softmax_op = top.SoftmaxOp(T(
                [1, self.num_attention_heads, input_len, self.max_num_tiles * self.num_patches]),
                                          qk_mm_add_op,
                                          axis=3,
                                          loc=L(TOP_PATH + "Softmax0"),
                                          beta=1,
                                          log=False,
                                          ip=ip).output
            qkv_mm_op = top.MatMulOp(T([1, self.num_attention_heads, input_len, self.head_dim]),
                                     qk_softmax_op,
                                     v_op,
                                     block_mlir.none_op,
                                     do_relu=False,
                                     loc=L(TOP_PATH + "fattention_matmul1"),
                                     ip=ip).output
            qkv_perm_op = top.PermuteOp(T([1, input_len, self.num_attention_heads, self.head_dim]),
                                        qkv_mm_op,
                                        order=[0, 2, 1, 3],
                                        loc=L(TOP_PATH + "fattention_permute2"),
                                        ip=ip).output
            qkv_rs_op = top.ReshapeOp(T(input_shape),
                                      qkv_perm_op,
                                      shape=input_shape,
                                      loc=L(TOP_PATH + "fattention_reshape_op1"),
                                      ip=ip).output

            o_op = self.linear(block_mlir, c_o_proj, qkv_rs_op, [q_dim, self.hidden_size],
                               input_shape)
            cross_att_gate_weight = block_mlir.create_weight_op(c_attn_gate, [1])
            tan_c_gate = top.TanhOp(T([1]),
                                    cross_att_gate_weight,
                                    loc=L(c_attn_gate + ".gate_tanh"),
                                    ip=ip).output
            gate_mul = top.MulOp(T(input_shape), [tan_c_gate, o_op],
                                 loc=L(c_attn_gate + ".gate_mul"),
                                 ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, gate_mul], loc=L(c_o_proj + ".add"),
                             ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp_cross(block_mlir, input_shape, o_op)
            new_op = top.MulOp(T(input_shape), [in2_op, new_op],
                               loc=L(c_mlp_gate + ".row_mask_mul"),
                               ip=ip).output
            cross_mlp_gate_weight = block_mlir.create_weight_op(c_mlp_gate, [1])
            tan_c_gate1 = top.TanhOp(T([1]),
                                     cross_mlp_gate_weight,
                                     loc=L(c_mlp_gate + ".gate_tanh"),
                                     ip=ip).output
            gate_mul1 = top.MulOp(T(input_shape), [tan_c_gate1, new_op],
                                  loc=L(c_mlp_gate + ".gate_mul"),
                                  ip=ip).output
            new_op = top.AddOp(T(input_shape), [o_op, gate_mul1], loc=L(c_mlp_gate + ".add"),
                               ip=ip).output

            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_cross_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [1, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            mask_shape = [1, 1, 1, self.max_num_tiles * self.num_patches]
            history_shape = [
                1, self.num_attention_heads, self.max_num_tiles * self.num_patches, self.head_dim
            ]

            q_shape = [1, 1, self.num_attention_heads, self.head_dim]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]

            block_mlir = MLIRImporter([input_shape, mask_shape, history_shape, history_shape],
                                      [input_shape],
                                      name,
                                      Platform.LLM, ["F32", "F32", "F32", "F32"],
                                      weight_file=weight_file)

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("attention_mask"), 1)
            in2_op = block_mlir.create_input_op(L("history_k"), 2)
            in3_op = block_mlir.create_input_op(L("history_v"), 3)

            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir, c_q_proj, ln_op, [self.hidden_size, q_dim],
                               [1, 1, q_dim])

            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(c_q_proj + ".reshape"), ip=ip).output
            q_op = self.rms_norm(block_mlir, q_op, c_q_norm)

            # ======= fattention =========
            # fa_op = top.FAttentionOp(T([1, 1, q_dim]),
            #                          q_op,
            #                          in2_op,
            #                          in3_op,
            #                          in1_op,
            #                          block_mlir.none_op,
            #                          scale=self.head_dim**-0.5,
            #                          batch=1,
            #                          q_head=self.num_attention_heads,
            #                          kv_head=self.num_attention_heads,
            #                          dim=self.head_dim,
            #                          mq=1,
            #                          mk=self.seq_length + 1,
            #                          loc=L(TOP_PATH + "fattention"),
            #                          ip=ip).output
            q_perm_op = top.PermuteOp(T([1, self.num_attention_heads, 1, self.head_dim]),
                                      q_op,
                                      order=[0, 2, 1, 3],
                                      loc=L(TOP_PATH + "fattention_permute0"),
                                      ip=ip).output
            k_perm_op = top.PermuteOp(T(
                [1, self.num_attention_heads, self.head_dim,
                 self.max_num_tiles * self.num_patches]),
                                      in2_op,
                                      order=[0, 1, 3, 2],
                                      loc=L(TOP_PATH + "fattention_permute1"),
                                      ip=ip).output
            qk_mm_op = top.MatMulOp(T(
                [1, self.num_attention_heads, 1, self.max_num_tiles * self.num_patches]),
                                    q_perm_op,
                                    k_perm_op,
                                    block_mlir.none_op,
                                    do_relu=False,
                                    loc=L(TOP_PATH + "fattention_matmul0"),
                                    ip=ip).output
            qk_mm_const_op = top.MulConstOp(T(
                [1, self.num_attention_heads, 1, self.max_num_tiles * self.num_patches]),
                                            qk_mm_op,
                                            const_val=self.head_dim**-0.5,
                                            loc=L(TOP_PATH + "fattention_mulconst0"),
                                            ip=ip).output
            qk_mm_add_op = top.AddOp(T(
                [1, self.num_attention_heads, 1, self.max_num_tiles * self.num_patches]),
                                     [qk_mm_const_op, in1_op],
                                     loc=L(TOP_PATH + "fattention_add0"),
                                     ip=ip).output
            qk_softmax_op = top.SoftmaxOp(T(
                [1, self.num_attention_heads, 1, self.max_num_tiles * self.num_patches]),
                                          qk_mm_add_op,
                                          axis=3,
                                          loc=L(TOP_PATH + "Softmax0"),
                                          beta=1,
                                          log=False,
                                          ip=ip).output
            qkv_mm_op = top.MatMulOp(T([1, self.num_attention_heads, 1, self.head_dim]),
                                     qk_softmax_op,
                                     in3_op,
                                     block_mlir.none_op,
                                     do_relu=False,
                                     loc=L(TOP_PATH + "fattention_matmul1"),
                                     ip=ip).output
            qkv_perm_op = top.PermuteOp(T([1, 1, self.num_attention_heads, self.head_dim]),
                                        qkv_mm_op,
                                        order=[0, 2, 1, 3],
                                        loc=L(TOP_PATH + "fattention_permute2"),
                                        ip=ip).output
            qkv_rs_op = top.ReshapeOp(T(input_shape),
                                      qkv_perm_op,
                                      shape=input_shape,
                                      loc=L(TOP_PATH + "fattention_reshape_op1"),
                                      ip=ip).output

            o_op = self.linear(block_mlir, c_o_proj, qkv_rs_op, [q_dim, self.hidden_size],
                               input_shape)
            cross_att_gate_weight = block_mlir.create_weight_op(c_attn_gate, [1])
            tan_c_gate = top.TanhOp(T([1]),
                                    cross_att_gate_weight,
                                    loc=L(c_attn_gate + ".gate_tanh"),
                                    ip=ip).output
            gate_mul = top.MulOp(T(input_shape), [tan_c_gate, o_op],
                                 loc=L(c_attn_gate + ".gate_mul"),
                                 ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, gate_mul], loc=L(c_o_proj + ".add"),
                             ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp_cross(block_mlir, input_shape, o_op)
            cross_mlp_gate_weight = block_mlir.create_weight_op(c_mlp_gate, [1])
            tan_c_gate1 = top.TanhOp(T([1]),
                                     cross_mlp_gate_weight,
                                     loc=L(c_mlp_gate + ".gate_tanh"),
                                     ip=ip).output
            gate_mul1 = top.MulOp(T(input_shape), [tan_c_gate1, new_op],
                                  loc=L(c_mlp_gate + ".gate_mul"),
                                  ip=ip).output
            new_op = top.AddOp(T(input_shape), [o_op, gate_mul1], loc=L(c_mlp_gate + ".add"),
                               ip=ip).output

            block_mlir.create_return_op([new_op])
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
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("past_v_Reshape"), ip=ip).output

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir,
                                               in1_op,
                                               q_op,
                                               k_op,
                                               rotary_cos,
                                               rotary_sin,
                                               decode=True)
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

            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        if idx in self.llm_config.cross_attention_layers:
            gen_cross_block()
            gen_cross_block_cache()
        else:
            gen_block()
            gen_block_cache()

    @override
    def combine(self):
        bmodel_list = []
        total_bytes = 0
        for i in range(self.num_layers):
            if i not in self.llm_config.cross_attention_layers:
                bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
                total_bytes += os.path.getsize("block_0.bmodel")
        for i in self.llm_config.cross_attention_layers:
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
            total_bytes += os.path.getsize("block_0.bmodel")
        if not self.embedding_disk:
            bmodel_list += ['embedding.bmodel', 'embedding_cache.bmodel']
            total_bytes += os.path.getsize("embedding.bmodel")
        if not self.lmhead_with_topk:
            bmodel_list += ["greedy_head.bmodel", "sample_head.bmodel"]
        if self.do_vit:
            bmodel_list += ["vit.bmodel"]
            total_bytes += os.path.getsize("vit.bmodel")
        bmodel_list += ["lm_head.bmodel"]
        total_bytes += os.path.getsize("lm_head.bmodel")

        combine_args = ['model_tool', '--combine', ' '.join(bmodel_list), '-o', self.out_bmodel]
        self.run_command(['bash', '-c', ' '.join(combine_args)])
        # Get the size of the combined bmodel
        bmodel_size = os.path.getsize(self.out_bmodel)
        print(f"Combined bmodel size: {bmodel_size / (1024.0 ** 3)} GB")
        if bmodel_size > total_bytes * 1.2:
            raise RuntimeError("Combined bmodel size is too large, please check the model.")

        get_info_args = ['model_tool', '--info', self.out_bmodel, '> ../model.log']
        self.run_command(['bash', '-c', ' '.join(get_info_args)])
