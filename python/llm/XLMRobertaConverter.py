# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
# ==============================================================================
"""XLM-RoBERTa encoder converter for bge-reranker-v2-m3.

Converts XLMRobertaForSequenceClassification (encoder-only, post-LN,
bidirectional attention, absolute position embeddings) to bmodel.

Nets produced:
  embedding   — word + position embeddings + LayerNorm
  block_0..23 — encoder layers (post-LN self-attn + FFN)
  classifier  — CLS pool → dense → tanh → dense → logit
"""

import numpy as np
import torch
from transform.MLIRImporter import MLIRImporter, Platform
from .LlmConverter import *
from .LlmInfo import ModelConfig, ModelInfo, LlmList, WeightType, ActType
import mlir.dialects.top as top


class XLMRobertaConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        self.model_info = XLM_ROBERTA_INFO
        super().__init__(args, config, loader)

    # ------------------------------------------------------------------
    # Config overrides
    # ------------------------------------------------------------------
    def init_config(self):
        """Override: set encoder-specific config (no rope, LayerNorm, GELU)."""
        self.num_layers = self.llm_config.num_hidden_layers
        self.num_attention_heads = self.llm_config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.llm_config.hidden_size
        self.vocab_size = self.llm_config.vocab_size
        self.intermediate_size = self.llm_config.intermediate_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_tile = 1
        self.rms_norm_eps = getattr(self.llm_config, "layer_norm_eps", 1e-5)
        self.hidden_act = ActType.GELU
        self.rmsnorm_type = WeightType.LAYERNORM
        self.tie_word_embeddings = False
        self.do_lmhead_merge = False
        self.scale_emb = 1.0
        self.scale_depth = 1.0
        self.dim_model_base = self.hidden_size
        # Encoder-only: no MoE, no LoRA, no experts
        self.num_experts = 1
        self.num_experts_per_tok = 1
        self.moe_intermediate_size = 1
        self.shared_expert_intermediate_size = 1
        self.mlp_only_layers = []
        self.decoder_sparse_step = 1
        self.norm_topk_prob = True
        self.init_quantization()

    def rotary_embedding(self):
        """Override: XLM-RoBERTa uses absolute position embeddings, not RoPE."""
        return np.zeros((1, 1, self.head_dim), dtype=np.float32), \
               np.zeros((1, 1, self.head_dim), dtype=np.float32)

    def get_dtype(self):
        """Override: XLM-RoBERTa stores weights as float32; treat as bf16/f16
        based on the --quantize flag so init_quantization picks the right qtype."""
        if 'bf16' in self.quantize:
            return torch.bfloat16
        return torch.float16

    # ------------------------------------------------------------------
    # MLIR generation overrides
    # ------------------------------------------------------------------
    def gen_all_mlir(self):
        """Only embedding + blocks + classifier (no lm_head, no cache)."""
        self.all_gen_mlirs.append(self.gen_embedding_mlir)
        for i in range(self.num_layers):
            self.all_gen_mlirs.append(lambda i=i: self.gen_block_mlir(i))
        self.all_gen_mlirs.append(self.gen_classifier_mlir)

        if self.debug:
            for func in self.all_gen_mlirs:
                func()
            return

        import concurrent.futures
        import sys
        from tqdm import tqdm
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(
                __import__("os").environ.get("LLM_GEN_MLIR_WORKERS", "4"))) as executor:
            futures = [executor.submit(func) for func in self.all_gen_mlirs]
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="generate mlir"):
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    traceback.print_exc()
                    sys.exit(1)

    def compile_all(self):
        """Only compile embedding + blocks + classifier."""
        # embedding
        if not self.register_bmodel("embedding", with_size=True):
            self.submit_deploy_task("embedding", [
                f'--quantize {self.half_precision_quantize}',
                '--quant_input',
                '--quant_output',
            ])

        # classifier
        if not self.register_bmodel("classifier", with_size=True):
            self.submit_deploy_task("classifier", [
                f'--quantize {self.half_precision_quantize}',
                '--quant_input',
                '--quant_output',
            ])

        # blocks
        for i in range(self.num_layers):
            name = f"block_{i}"
            if not self.register_bmodel(name, with_size=True):
                self._deploy_block_phase(name, i)

        self.execute_tasks()
        self.combine()

        # Clean npz unless debugging
        if not self.debug:
            import os
            for dirpath, _, filenames in os.walk("."):
                if dirpath.startswith("./config"):
                    continue
                for filename in filenames:
                    if filename.endswith(".npz"):
                        os.remove(os.path.join(dirpath, filename))

    # ------------------------------------------------------------------
    # Embedding net
    # ------------------------------------------------------------------
    def gen_embedding_mlir(self):
        """Generate embedding net: word_emb + pos_emb + LayerNorm."""
        name = "embedding"
        from tqdm import tqdm
        tqdm.write("generate embedding mlir ...")

        # Weight paths
        word_emb = "roberta.embeddings.word_embeddings"
        pos_emb = "roberta.embeddings.position_embeddings"
        emb_ln = "roberta.embeddings.LayerNorm"

        # Save weights
        weight_dict = {}
        self.set_common_weight(word_emb, weight_dict)
        self.set_common_weight(pos_emb, weight_dict)
        self.set_common_weight(emb_ln, weight_dict)
        self.weight_keys.extend(list(weight_dict.keys()))
        weight_file = "embedding_top_weights.npz"
        np.savez(weight_file, **weight_dict)

        seq = self.max_input_length
        out_shape = [1, seq, self.hidden_size]

        mlir_gen = MLIRImporter(
            [[1, seq], [1, seq]],  # input_ids, position_ids
            [out_shape],
            name,
            self.platform,
            input_types=["INT32", "INT32"],
            weight_file=f"../{weight_file}")

        T = mlir_gen.get_tensor_type
        L = lambda n: self.get_loc(n, mlir_gen)
        ip = mlir_gen.insert_point

        input_ids = mlir_gen.create_input_op(L("input_ids"), 0)
        position_ids = mlir_gen.create_input_op(L("position_ids"), 1)

        # Word embedding lookup
        w_word = mlir_gen.create_weight_op(word_emb + ".weight",
                                           [self.vocab_size, self.hidden_size])
        word_op = top.GatherOp(T(out_shape), w_word, input_ids, axis=0, loc=L("word_emb"),
                               ip=ip).output

        # Position embedding lookup
        w_pos = mlir_gen.create_weight_op(pos_emb + ".weight", [8194, self.hidden_size])
        pos_op = top.GatherOp(T(out_shape), w_pos, position_ids, axis=0, loc=L("pos_emb"),
                              ip=ip).output

        # word + position (token_type_embedding is zero for type_vocab_size=1)
        emb = top.AddOp(T(out_shape), [word_op, pos_op], loc=L("emb_add"), ip=ip).output

        # LayerNorm
        out = self.layer_norm(mlir_gen, emb, emb_ln, self.rms_norm_eps)

        mlir_gen.create_return_op([out])
        self.save_mlir_module(mlir_gen, name)

    # ------------------------------------------------------------------
    # Block net (encoder layer)
    # ------------------------------------------------------------------
    def gen_block_mlir(self, idx):
        """Generate one encoder layer: post-LN self-attn + post-LN FFN."""
        from tqdm import tqdm
        tqdm.write(f"generate block_{idx} mlir ...")

        layer = f"roberta.encoder.layer.{idx}"
        q_proj = f"{layer}.attention.self.query"
        k_proj = f"{layer}.attention.self.key"
        v_proj = f"{layer}.attention.self.value"
        o_proj = f"{layer}.attention.output.dense"
        attn_ln = f"{layer}.attention.output.LayerNorm"
        ffn_up = f"{layer}.intermediate.dense"
        ffn_down = f"{layer}.output.dense"
        ffn_ln = f"{layer}.output.LayerNorm"

        # Save weights (set_linear_weight transposes (out,in) → (in,out))
        weight_dict = {}
        self.set_linear_weight(q_proj, weight_dict)
        self.set_linear_weight(k_proj, weight_dict)
        self.set_linear_weight(v_proj, weight_dict)
        self.set_linear_weight(o_proj, weight_dict)
        self.set_common_weight(attn_ln, weight_dict)
        self.set_linear_weight(ffn_up, weight_dict)
        self.set_linear_weight(ffn_down, weight_dict)
        self.set_common_weight(ffn_ln, weight_dict)
        self.weight_keys.extend(list(weight_dict.keys()))

        weight_file = f"block_{idx}_top_weights.npz"
        np.savez(weight_file, **weight_dict)

        seq = self.max_input_length
        input_shape = [1, seq, self.hidden_size]
        q_shape = [1, self.num_attention_heads, seq, self.head_dim]
        kv_shape = q_shape  # MHA: num_kv_heads == num_heads
        mask_shape = [1, 1, 1, seq]

        # ---- build MLIR graph ----
        name = f"block_{idx}"
        mlir_gen = MLIRImporter(
            [input_shape, mask_shape],  # hidden_states, additive_mask
            [input_shape],
            name,
            self.platform,
            input_types=["F32", "F32"],
            weight_file=f"../{weight_file}")

        T = mlir_gen.get_tensor_type
        L = lambda n: self.get_loc(n, mlir_gen)
        ip = mlir_gen.insert_point

        in_op = mlir_gen.create_input_op(L("input_states"), 0)
        mask_op = mlir_gen.create_input_op(L("attention_mask"), 1)

        # ---- Self-Attention ----
        residual = in_op

        # Q / K / V projections (with bias, via self.linear)
        q_op = self.linear(mlir_gen, q_proj, in_op, [self.hidden_size, self.hidden_size],
                           input_shape)
        k_op = self.linear(mlir_gen, k_proj, in_op, [self.hidden_size, self.hidden_size],
                           input_shape)
        v_op = self.linear(mlir_gen, v_proj, in_op, [self.hidden_size, self.hidden_size],
                           input_shape)

        # Reshape [1, seq, hidden] → [1, num_heads, seq, head_dim]
        q_op = top.ReshapeOp(T([1, seq, self.num_attention_heads, self.head_dim]),
                             q_op,
                             shape=[1, -1, self.num_attention_heads, self.head_dim],
                             loc=L(q_proj + ".reshape"),
                             ip=ip).output
        k_op = top.ReshapeOp(T([1, seq, self.num_attention_heads, self.head_dim]),
                             k_op,
                             shape=[1, -1, self.num_attention_heads, self.head_dim],
                             loc=L(k_proj + ".reshape"),
                             ip=ip).output
        v_op = top.ReshapeOp(T([1, seq, self.num_attention_heads, self.head_dim]),
                             v_op,
                             shape=[1, -1, self.num_attention_heads, self.head_dim],
                             loc=L("v.reshape"),
                             ip=ip).output

        # Transpose [1, num_heads, seq, head_dim]
        q_op = top.PermuteOp(T(q_shape), q_op, order=[0, 2, 1, 3], loc=L("q.transpose"),
                             ip=ip).output
        k_op = top.PermuteOp(T(kv_shape), k_op, order=[0, 2, 1, 3], loc=L("k.transpose"),
                             ip=ip).output
        v_op = top.PermuteOp(T(kv_shape), v_op, order=[0, 2, 1, 3], loc=L("v.transpose"),
                             ip=ip).output

        # QK^T / sqrt(d)   →  [1, num_heads, seq, seq]
        scale = self.head_dim**-0.5
        k_t = top.PermuteOp(T([1, self.num_attention_heads, self.head_dim, seq]),
                            k_op,
                            order=[0, 1, 3, 2],
                            loc=L("k.transpose2"),
                            ip=ip).output
        scores = top.MatMulOp(T([1, self.num_attention_heads, seq, seq]),
                              q_op,
                              k_t,
                              mlir_gen.none_op,
                              do_relu=False,
                              loc=L("qk_matmul"),
                              ip=ip).output
        scores = top.MulConstOp(T([1, self.num_attention_heads, seq, seq]),
                                scores,
                                const_val=scale,
                                loc=L("scale"),
                                ip=ip).output

        # Add additive mask [1, 1, 1, seq] → broadcast over heads & query dims
        scores = top.AddOp(T([1, self.num_attention_heads, seq, seq]), [scores, mask_op],
                           loc=L("add_mask"),
                           ip=ip).output

        # Softmax
        attn_w = top.SoftmaxOp(T([1, self.num_attention_heads, seq, seq]),
                               scores,
                               axis=3,
                               loc=L("softmax"),
                               ip=ip).output

        # Attention output: attn_w × V → [1, num_heads, seq, head_dim]
        attn_out = top.MatMulOp(T(q_shape),
                                attn_w,
                                v_op,
                                mlir_gen.none_op,
                                do_relu=False,
                                loc=L("attn_matmul"),
                                ip=ip).output

        # Transpose back [1, num_heads, seq, head_dim] → [1, seq, num_heads, head_dim]
        attn_out = top.PermuteOp(T([1, seq, self.num_attention_heads, self.head_dim]),
                                 attn_out,
                                 order=[0, 2, 1, 3],
                                 loc=L("attn.transpose"),
                                 ip=ip).output

        # Reshape [1, seq, hidden]
        attn_out = top.ReshapeOp(T(input_shape),
                                 attn_out,
                                 shape=[1, -1, self.hidden_size],
                                 loc=L("attn.reshape"),
                                 ip=ip).output

        # Output projection (with bias)
        o_op = self.linear(mlir_gen, o_proj, attn_out, [self.hidden_size, self.hidden_size],
                           input_shape)

        # Post-LN: LayerNorm(residual + o_proj_output)
        o_op = top.AddOp(T(input_shape), [residual, o_op], loc=L("attn_residual"), ip=ip).output
        o_op = self.layer_norm(mlir_gen, o_op, attn_ln, self.rms_norm_eps)

        # ---- FFN ----
        ffn_residual = o_op

        # Intermediate: Linear(hidden → 4*hidden) + GELU
        ffn_mid = self.linear(mlir_gen, ffn_up, o_op, [self.hidden_size, self.intermediate_size],
                              [1, seq, self.intermediate_size])
        ffn_mid = self.activate(mlir_gen, ffn_mid, ActType.GELU, ffn_up)

        # Output: Linear(4*hidden → hidden)
        ffn_out = self.linear(mlir_gen, ffn_down, ffn_mid,
                              [self.intermediate_size, self.hidden_size], input_shape)

        # Post-LN: LayerNorm(residual + ffn_output)
        out = top.AddOp(T(input_shape), [ffn_residual, ffn_out], loc=L("ffn_residual"),
                        ip=ip).output
        out = self.layer_norm(mlir_gen, out, ffn_ln, self.rms_norm_eps)

        mlir_gen.create_return_op([out])
        self.save_mlir_module(mlir_gen, name)

    # ------------------------------------------------------------------
    # Classifier net
    # ------------------------------------------------------------------
    def gen_classifier_mlir(self):
        """CLS pool → dense → tanh → dense → logit."""
        name = "classifier"
        from tqdm import tqdm
        tqdm.write("generate classifier mlir ...")

        dense = "classifier.dense"
        out_proj = "classifier.out_proj"

        weight_dict = {}
        self.set_linear_weight(dense, weight_dict)
        self.set_linear_weight(out_proj, weight_dict)
        self.weight_keys.extend(list(weight_dict.keys()))
        np.savez("classifier_top_weights.npz", **weight_dict)

        mlir_gen = MLIRImporter([[1, self.max_input_length, self.hidden_size]], [[1, 1]],
                                name,
                                self.platform,
                                weight_file="../classifier_top_weights.npz")

        T = mlir_gen.get_tensor_type
        L = lambda n: self.get_loc(n, mlir_gen)
        ip = mlir_gen.insert_point

        hidden = mlir_gen.create_input_op(L("hidden_states"), 0)

        # CLS pool: features[:, 0, :] → [1, hidden]
        cls_op = top.SliceOp(T([1, 1, self.hidden_size]),
                             hidden,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             mlir_gen.none_op,
                             offset=[0, 0, 0],
                             steps=[1, 1, 1],
                             ends=[1, 1, self.hidden_size],
                             loc=L("cls_pool"),
                             ip=ip).output
        cls_op = top.ReshapeOp(T([1, self.hidden_size]), cls_op, loc=L("cls_reshape"), ip=ip).output

        # Dense (1024 → 1024) + tanh
        x = self.linear(mlir_gen, dense, cls_op, [self.hidden_size, self.hidden_size],
                        [1, self.hidden_size])
        x = top.TanhOp(T([1, self.hidden_size]), x, loc=L("tanh"), ip=ip).output

        # Out projection (1024 → 1)
        logits = self.linear(mlir_gen, out_proj, x, [self.hidden_size, 1], [1, 1])

        mlir_gen.create_return_op([logits])
        self.save_mlir_module(mlir_gen, name)
