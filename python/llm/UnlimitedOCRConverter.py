# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# Unlimited-OCR converter.
#
# Unlimited-OCR (https://github.com/baidu/Unlimited-OCR) is a DeepseekV2-MoE
# based VLM: a "deeplip" vision tower (CLIP-L + SAM ViT-B) + a 12-layer
# DeepseekV2 MoE LLM (64 routed experts / top-6, 2 shared experts).
#
# This converter reuses LlmConverter's QWEN2_MOE path for the LLM and adds a
# custom vision tower.  Differences from the stock QWEN2_MOE path:
#   * DeepseekV2's shared expert is a *gateless* plain MLP (no
#     `mlp.shared_expert_gate` weight).  We skip loading that weight and drop
#     the sigmoid gate in `moe()`.
#   * Shared expert weight prefix is `mlp.shared_experts.*` (plural).
#   * Shared expert intermediate = n_shared_experts * moe_intermediate_size.

from .LlmConverter import *
from .LlmInfo import ModelInfo, ModelConfig, LlmList, COMMON_INFO, LlmType, ActType
from typing_extensions import override
import math
import numpy as np
import torch


# DeepseekV2 / Unlimited-OCR weight names.  Identical to COMMON_INFO except:
#   * shared experts live under `mlp.shared_experts.*` (plural)
#   * SHARED_GATE is kept as the stock name only so the dispatch builds the
#     path; set_linear_weight() skips actually loading it (it does not exist).
_unlimited_weights = dict(COMMON_INFO.weights)
_unlimited_weights[LlmList.SHARED_EXPERT_GATE] = "mlp.shared_experts.gate_proj"
_unlimited_weights[LlmList.SHARED_EXPERT_UP] = "mlp.shared_experts.up_proj"
_unlimited_weights[LlmList.SHARED_EXPERT_DOWN] = "mlp.shared_experts.down_proj"
UNLIMITED_OCR_INFO = ModelInfo(ModelConfig(), weights=_unlimited_weights)


class UnlimitedOCRConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        super().__init__(args, config, loader=loader)
        # Vision tower (deeplip = CLIP-L + SAM ViT-B + projector).
        # For GGUF input without --mmproj, the vision tower isn't in the LLM
        # gguf -> LLM-only (compile vision separately from safetensors).
        from .ModelHandle import GGUFModelHandle
        is_gguf = isinstance(self.loader, GGUFModelHandle)
        # Env override: UOCR_LLM_ONLY=1 forces LLM-only (skip vit) so the vision
        # tower's address-assign segfault doesn't halt the LLM block queue.
        import os
        if os.environ.get("UOCR_LLM_ONLY"):
            self.do_vit = False
        else:
            self.do_vit = (not is_gguf) or bool(getattr(args, "mmproj", None))
        self.vit_path = "model"
        self.max_pixels = getattr(args, "max_pixels", 1024 * 1024)
        self.max_shape = getattr(args, "max_shape", None) or (1024, 1024)
        if self.do_vit:
            self.init_vconfig()

    def init_vconfig(self):
        """Deeplip vision tower dims (see sample README §3).

        SAM ViT-B: patch16 -> 768, 12 heads/64 dim, 12 blocks, mlp 3072,
        neck 256, net_2 512, net_3 1024 (downsample to 16x16 for 1024 input).
        CLIP-L: 1024, 16 heads/64 dim, 24 blocks, mlp 4096, quick_gelu.
        Projector: Linear 2048 -> 1280.
        """
        vc = self.config.vision_config
        # fixed base-size global view (1024x1024) for v1 (base / multi-page mode)
        self.vit_image_size = 1024
        # SAM
        self.sam_embed = 768
        self.sam_depth = 12
        self.sam_heads = 12
        self.sam_head_dim = 64
        self.sam_mlp = 3072
        self.sam_patch = 16
        self.sam_grid = self.vit_image_size // self.sam_patch   # 64
        self.sam_neck = 256
        self.sam_net2 = 512
        self.sam_net3 = 1024
        # CLIP-L
        self.clip_embed = 1024
        self.clip_depth = 24
        self.clip_heads = 16
        self.clip_head_dim = 64
        self.clip_mlp = 4096
        # after SAM downsample (net_2/net_3 stride 2x2): 64 -> 32 -> 16
        self.clip_grid = 16          # CLIP sees SAM's 16x16 -> 256 tokens
        self.clip_seq = self.clip_grid * self.clip_grid   # 256 (excl. CLS)
        # projector
        self.proj_in = 2048
        self.proj_out = self.hidden_size   # 1280


    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        # No text_config on Unlimited-OCR; the top-level config already exposes
        # all DeepseekV2 LLM fields (hidden_size, n_routed_experts, ...).
        self.llm_type = LlmType.QWEN2_MOE
        self.model_info = UNLIMITED_OCR_INFO
        lc = self.llm_config
        # Map DeepseekV2 -> QWEN2_MOE fields expected by LlmConverter.init_config.
        # Handle both safetensors (n_routed_experts) and GGUF (expert_count) naming.
        lc.n_routed_experts = getattr(lc, "n_routed_experts", None) or getattr(lc, "expert_count", 1)
        lc.num_experts = lc.n_routed_experts
        lc.num_experts_per_tok = getattr(lc, "num_experts_per_tok", None) or getattr(lc, "expert_used_count", 1)
        lc.moe_intermediate_size = getattr(lc, "moe_intermediate_size", None) or getattr(lc, "expert_feed_forward_length", 1)
        lc.n_shared_experts = getattr(lc, "n_shared_experts", None) or getattr(lc, "expert_shared_count", 0)
        lc.shared_expert_intermediate_size = int(lc.moe_intermediate_size) * int(lc.n_shared_experts or 1)
        # layer 0 is dense (first_k_dense_replace / leading_dense_block_count = 1)
        first_k = getattr(lc, "first_k_dense_replace", None)
        if first_k is None:
            first_k = getattr(lc, "leading_dense_block_count", 0)
        lc.first_k_dense_replace = int(first_k)
        lc.mlp_only_layers = list(range(int(first_k)))  # v1.29 LlmConverter uses this
        lc.decoder_sparse_step = 1
        lc.mlp_only_layers = list(range(lc.first_k_dense_replace))
        # norm_topk_prob already False; routed_scaling_factor=1.0 (no-op).
        # intermediate_size (=6848) for the dense layer is already present.
        # This model uses SlidingWindowLlamaAttention (standard MHA + Llama RoPE,
        # rope_theta=10000) — NOT MLA, despite the DeepseekV2 architecture tag.
        # GGUF metadata rope.dimension_count=0 is the MLA decoupled-RoPE field
        # (irrelevant for use_mla=False); do NOT use it to disable RoPE.
        # This model uses Llama RoPE with theta=10000 (DeepseekV2Config default).
        # GGUF has no rope_theta key, so LlmConverter.rotary_embedding() defaults
        # to 1e6 (wrong by 100x).  Force the correct value.
        lc.rope_theta = 10000.0

    @override
    def split_fused_moe(self):
        # Qwen3.5-style: don't split the expert MLP (bypasses the bf16/f16
        # assert + npu_num divisibility in split_fused_moe).  The single
        # MlpOp handles int4 RTN during compile (MlpOp codegen supports W4).
        # Env override UOCR_SPLIT_MOE=N to split into N MlpOps (smaller per-op
        # local mem; workaround if a single 64-expert MlpOp hangs the TPU).
        import os
        n = os.environ.get("UOCR_SPLIT_MOE")
        return int(n) if n else 1

    @override
    def init_quantization(self):
        # GGUF RTN mode (env UOCR_GGUF_RTN=1): mirror the HF w4bf16 path.
        # Dequantize every GGUF tensor to float .weight (via the loader's
        # full-float-fallback read path), set quant_mode=None, and let the
        # block compile-time RTN (convert-top-to-tpu --quantize w4bf16
        # --q_group_size 128) quantize all linears -- including MoE experts --
        # to int4.  This bypasses GGUF's 3D-expert int4 packing (unsupported,
        # 2D-only) and the mixed-quant (Q4_K_M) whole-block float fallback,
        # so MoE experts end up int4 (compact) instead of bf16 (bloated).
        import os
        from .ModelHandle import GGUFModelHandle
        if isinstance(self.loader, GGUFModelHandle) and os.environ.get("UOCR_GGUF_RTN"):
            self.quant_mode = None
            if not self.quantize or self.quantize == "auto":
                self.quantize = "w4bf16"
            self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"
            self.quant_bits = 4 if self.quantize.startswith("w4") else (
                8 if self.quantize.startswith("w8") else 16)
            if not self.q_group_size or self.q_group_size <= 0:
                self.q_group_size = 128
            # Force the loader's per-block float-fallback map ON for every block
            # so GGUFModelHandle.set_linear_weight dequantizes (reads float) for
            # all quantized tensors, including attn / shared-expert linears.
            n = int(getattr(self.llm_config, "num_hidden_layers", 0) or 0)
            self.loader._blocks_full_float_fallback = {i: True for i in range(n)}
            self.loader._block_quant_info = {}
            self.loader._mixed_quant_fallback = False
            # Decouple the block *compile* quantize from the float-fallback:
            # always emit w4bf16 + q_group_size (the fallback map only governs
            # weight *loading*, not the compile quantize param).
            gs = self.q_group_size
            qz = self.quantize

            def _rtn_block_args(conv, layer_id, is_cache=False):
                return qz, [f"--q_group_size {gs}"]

            self.loader.compile_block_args = _rtn_block_args
            return
        super().init_quantization()

    @override
    def set_linear_weight(self, path: str, weight_dict: dict, do_lora: bool = False):
        # DeepseekV2's shared expert has no gate weight; skip the stock
        # `mlp.shared_expert_gate` path so loading does not raise.
        if path.endswith("mlp.shared_expert_gate"):
            return
        import re, os
        # One-hot MoE path (Compare-based): inject expert_range=[0,1,...,E-1]
        # into this block's weight_dict. Referenced by create_weight_op in
        # _moe_unfuse_onehot via CompareOp(Equal). Idempotent across many
        # set_linear_weight calls; weight_dict is per-block.
        if os.environ.get("UOCR_UNFUSE") and os.environ.get("UOCR_DENSE"):
            import numpy as np
            weight_dict["uocr_expert_range"] = np.arange(self.num_experts,
                                                          dtype=np.float32)
        from .ModelHandle import GGUFModelHandle
        is_gguf = isinstance(self.loader, GGUFModelHandle)
        # GGUF RTN mode: routed experts are 3D consolidated -> per-expert RTN
        # int4 (gptq format). The stock gen_block calls set_linear_weight
        # per-expert (indexed 0..N-1) and assembles a 3D `.weight` (float). We
        # (a) store the per-expert `.weight` (transposed float) so the stock
        # assembly doesn't KeyError -- that 3D .weight ends up unreferenced
        # (the MlpOp with quant_mode=gptq uses .qweight), so it stays in the
        # intermediate f32 npz only, NOT in the bmodel; and (b) on the last
        # expert, pack all experts to int4 and assemble 3D .qweight/.scales/
        # .qzeros (Qwen3_5 pattern, npu_num scale transpose) for the fused
        # MlpOp (quantized=True). This makes routed experts int4 (compact)
        # instead of bf16 (bloated); the stock MlpOp with quantized=False +
        # float weight is NOT RTN'd by convert-top-to-tpu (only A16MatMul is).
        if is_gguf and os.environ.get("UOCR_GGUF_RTN"):
            # shared experts (merged): GGUF stores them as blk.N.ffn_{gate,up,down}_shexp.weight
            # (v1.29 ModelHandle doesn't map the .weight via _map_key_to_gguf, so read directly).
            m_sh = re.match(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|up|down)_proj$",
                            path)
            if m_sh:
                layer_sh, gud_sh = int(m_sh.group(1)), m_sh.group(2)
                try:
                    w = self.loader.read(f"blk.{layer_sh}.ffn_{gud_sh}_shexp.weight")  # [out, in] float
                except RuntimeError:
                    # dense layer (no shared experts in GGUF) — skip
                    return
                weight_dict[path + ".weight"] = np.ascontiguousarray(np.ascontiguousarray(w).T)
                return
            m = re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$",
                         path)
            if m:
                layer, eid, gud = int(m.group(1)), int(m.group(2)), m.group(3)
                unfuse = bool(os.environ.get("UOCR_UNFUSE"))
                cache = self.__dict__.setdefault("_uocr_expert_cache", {})
                if (layer, gud) not in cache:
                    cache[(layer, gud)] = self.loader.read(
                        f"blk.{layer}.ffn_{gud}_exps.weight")  # [E, out, in] float
                cons = cache[(layer, gud)]
                # per-expert .weight (transposed to [in, out] for stock 3D assembly)
                w_in_out = np.ascontiguousarray(np.ascontiguousarray(cons[eid]).T)
                weight_dict[path + ".weight"] = w_in_out
                if unfuse:
                    # Un-fuse path: keep a per-expert 2D alias that survives
                    # gen_block's consolidation (it `del`s only "{path}.weight",
                    # not the "__uocr_unfuse.weight" alias). moe() then drives
                    # each expert through self.linear -> MatMulOp -> lowered to
                    # A16MatMul int4 (the SE9-proven block_0 dense-MLP path),
                    # avoiding the fused MlpOp (hangs on SE9 / crashes cmodel).
                    weight_dict[path + "__uocr_unfuse.weight"] = np.array(w_in_out)
                # on the last expert's down call, pack+assemble 3D int4 for the
                # fused MlpOp path (only when NOT un-fusing). In both modes,
                # drop this layer's cached 3D dequantized tensors afterward
                # (the un-fuse path otherwise accumulates ~880MB/layer across
                # all 12 MoE layers -> 10GB+ RSS and swap thrashing).
                if eid == self.num_experts - 1 and gud == "down":
                    if not unfuse:
                        self._assemble_experts_int4(layer, weight_dict)
                    for k in [k for k in self._uocr_expert_cache if k[0] == layer]:
                        del self._uocr_expert_cache[k]
                return
        # Safetensors (non-GGUF) un-fuse path: per-expert weights come straight
        # from HF safetensors (f32, no GGUF Q4_K_M dequant loss).  Load via the
        # stock loader (stores path+".weight" transposed f32) and alias it for
        # _moe_unfuse_onehot.  gen_block's consolidation later dels ".weight" but
        # not the "__uocr_unfuse.weight" alias, so the alias survives into the
        # npz.  quant_mode stays None (HF model has no quantization_config) ->
        # block compile RTN int4 (same as the SE9-proven block_0 dense-MLP path).
        if (not is_gguf) and os.environ.get("UOCR_UNFUSE") and os.environ.get("UOCR_DENSE"):
            m = re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$",
                         path)
            if m:
                super().set_linear_weight(path, weight_dict, do_lora=do_lora)
                wkey = path + ".weight"
                if wkey in weight_dict:
                    weight_dict[path + "__uocr_unfuse.weight"] = np.array(weight_dict[wkey])
                return
        return super().set_linear_weight(path, weight_dict, do_lora=do_lora)

    def _assemble_experts_int4(self, layer: int, weight_dict: dict):
        """Pack all 64 experts (from cached 3D dequantized tensors) to int4
        gptq and assemble 3D .qweight/.scales/.qzeros matching the shapes
        LlmConverter.mlp() creates for the is_expert MlpOp (quant_mode=gptq)."""
        cache = self._uocr_expert_cache
        E = int(self.num_experts)
        npu = int(self.tpu_info.npu_num)        # 32 for BM1688
        gs = int(self.q_group_size)            # 128
        bits = int(self.quant_bits)            # 4
        zp = 1 << (bits - 1)                   # 8, symmetric
        for gud in ("gate", "up", "down"):
            cons = cache[(layer, gud)]          # [E, out, in]
            out_dim, in_dim = cons.shape[1], cons.shape[2]
            n_groups = in_dim // gs
            qw = np.zeros((E, out_dim, in_dim // 2), dtype=np.uint8)
            sc = np.zeros((E, out_dim, n_groups), dtype=np.float32)
            qz = np.zeros((E, out_dim, n_groups // 2), dtype=np.uint8)
            for e in range(E):
                w = cons[e].reshape(out_dim, n_groups, gs).astype(np.float32)
                amax = np.max(np.abs(w), axis=-1)
                scale = np.where(amax == 0, 1.0, amax / 7.0)
                q = np.clip(np.round(w / scale[..., None]), -8, 7).astype(np.int32) + 8
                q = q.reshape(out_dim, in_dim).astype(np.uint8)
                qw[e] = q[:, 0::2] | (q[:, 1::2] << 4)      # 2 int4/byte, low=even
                sc[e] = scale
                zpa = np.full((out_dim, n_groups), zp, dtype=np.uint8)
                qz[e] = zpa[:, 0::2] | (zpa[:, 1::2] << 4)
            if gud in ("gate", "up"):
                # op shape: [E, npu, out//npu * groups] (flat npu-major)
                sc = sc.reshape(E, out_dim // npu, npu, n_groups).transpose(0, 2, 1, 3) \
                        .reshape(E, npu, (out_dim // npu) * n_groups)
                qz = qz.reshape(E, out_dim // npu, npu, n_groups // 2).transpose(0, 2, 1, 3) \
                        .reshape(E, npu, (out_dim // npu) * (n_groups // 2))
            else:  # down: op shape [E, groups, in] -> transpose(0,2,1)
                sc = np.ascontiguousarray(sc.transpose(0, 2, 1))
                qz = np.ascontiguousarray(qz.transpose(0, 2, 1))
            ep = f"model.layers.{layer}.mlp.experts.expert_id.{gud}_proj"
            weight_dict[ep + ".qweight"] = np.ascontiguousarray(qw)
            weight_dict[ep + ".scales"] = np.ascontiguousarray(sc)
            weight_dict[ep + ".qzeros"] = np.ascontiguousarray(qz)
        for k in [k for k in cache if k[0] == layer]:
            del cache[k]

    @override
    def mlp(self, mlir_gen, proj_gate, proj_up, proj_down, input_op, experts_id,
            seq_len, hidden_size, intermediate_size, act_type, is_expert=False,
            num_experts=1, num_experts_per_tok=1, force_bias=False, do_lora=False):
        # GGUF RTN mode: routed experts are pre-packed as int4 .qweight (see
        # set_linear_weight), so force quant_mode="gptq" for the expert MlpOp
        # -> quantized=True, references .qweight/.scales/.qzeros (int4, compact).
        # Non-expert / non-GGUF paths keep the stock quant_mode (A16MatMul RTN).
        import os
        from .ModelHandle import GGUFModelHandle
        if (is_expert and isinstance(self.loader, GGUFModelHandle)
                and os.environ.get("UOCR_GGUF_RTN")):
            orig = self.quant_mode
            self.quant_mode = "gptq"
            try:
                return super().mlp(mlir_gen, proj_gate, proj_up, proj_down, input_op,
                                   experts_id, seq_len, hidden_size, intermediate_size,
                                   act_type, is_expert=True, num_experts=num_experts,
                                   num_experts_per_tok=num_experts_per_tok,
                                   force_bias=force_bias, do_lora=do_lora)
            finally:
                self.quant_mode = orig
        return super().mlp(mlir_gen, proj_gate, proj_up, proj_down, input_op,
                           experts_id, seq_len, hidden_size, intermediate_size,
                           act_type, is_expert=is_expert, num_experts=num_experts,
                           num_experts_per_tok=num_experts_per_tok,
                           force_bias=force_bias, do_lora=do_lora)

    @override
    def moe(self,
            mlir_gen,
            proj_shared_gate,
            proj_shared_expert_gate,
            proj_shared_expert_up,
            proj_shared_expert_down,
            proj_gate,
            proj_experts_gate,
            proj_experts_up,
            proj_experts_down,
            input_op,
            seq_len,
            act_type: ActType,
            num_split_fused_moe=1,
            force_bias: bool = False,
            do_lora: bool = False):
        """DeepseekV2 MoE: softmax gate + greedy top-k (no renormalize since
        norm_topk_prob=False) + per-expert MLP + a *gateless* shared expert MLP
        (intermediate = shared_expert_intermediate_size)."""
        assert (act_type == "silu")
        shared_inter = self.shared_expert_intermediate_size

        # ---- shared expert: plain MLP, NO sigmoid gate ----
        if self.fused_mlp:
            shared_output = self.mlp(mlir_gen,
                                     proj_shared_expert_gate,
                                     proj_shared_expert_up,
                                     proj_shared_expert_down,
                                     input_op,
                                     mlir_gen.none_op,
                                     seq_len,
                                     self.hidden_size,
                                     shared_inter,
                                     act_type,
                                     is_expert=False)
        else:
            shared_mlp_gate = self.linear(mlir_gen, proj_shared_expert_gate, input_op,
                                          [self.hidden_size, shared_inter],
                                          [1, seq_len, shared_inter])
            shared_mlp_up = self.linear(mlir_gen, proj_shared_expert_up, input_op,
                                        [self.hidden_size, shared_inter],
                                        [1, seq_len, shared_inter])
            shared_mlp_silu = self.activate(mlir_gen, shared_mlp_gate, act_type,
                                            proj_shared_expert_gate)
            shared_mlp_mul = top.MulOp(mlir_gen.get_tensor_type([1, seq_len, shared_inter]),
                                       [shared_mlp_silu, shared_mlp_up],
                                       loc=self.get_loc(proj_shared_expert_gate + "_mul", mlir_gen),
                                       ip=mlir_gen.insert_point).output
            shared_output = self.linear(mlir_gen, proj_shared_expert_down, shared_mlp_mul,
                                        [shared_inter, self.hidden_size],
                                        [1, seq_len, self.hidden_size])

        # ---- routed experts gate: softmax + greedy top-k ----
        gate = self.linear(mlir_gen, proj_gate, input_op, [self.hidden_size, self.num_experts],
                           [1, seq_len, self.num_experts])
        softmax = top.SoftmaxOp(mlir_gen.get_tensor_type([1, seq_len, self.num_experts]),
                                gate, axis=2,
                                loc=self.get_loc(proj_gate + "_softmax", mlir_gen),
                                ip=mlir_gen.insert_point).output

        if os.environ.get("UOCR_UNFUSE") and os.environ.get("UOCR_DENSE"):
            # One-hot top-k MoE (SE9-safe, top-k precision). Avoids
            # GatherElements/Permute (which mis-execute on SE9 libsophon ->
            # 5.96e32 garbage / NaN). Path:
            #   64 experts -> reshape[seq,1,H] -> Concat axis1 -> [seq,E,H]
            #   eye[E,E] + GatherOp(axis=0, idx[seq*topk]) -> one_hot[seq,topk,E]
            #   one_hot * routing[seq,topk,1] -> ReduceSum topk -> routing_dense[seq,E]
            #     (== top-k scatter: top-k positions have score, rest 0)
            #   [seq,E,H] * [seq,E,1] -> ReduceSum E -> [seq,H]
            # Math-equivalent to top-k sparse routing (only top-6 experts
            # contribute), unlike full-softmax dense (all 64 -> precision loss).
            # Uses only SE9-proven base ops + GatherOp (Gemma4-proven, simpler
            # than GatherElements). eye weight injected in set_linear_weight.
            topk_op = top.TopKOp(mlir_gen.get_tensor_type([1, seq_len, self.num_experts_per_tok]),
                                 mlir_gen.get_tensor_type([1, seq_len, self.num_experts_per_tok]),
                                 softmax, axis=2, K=self.num_experts_per_tok,
                                 use_hau=seq_len > 1,
                                 loc=self.get_loc([proj_gate + "_values", proj_gate + "_indices"], mlir_gen),
                                 ip=mlir_gen.insert_point)
            experts_out = self._moe_unfuse_onehot(
                mlir_gen, proj_experts_gate, proj_experts_up, proj_experts_down,
                input_op, topk_op.indices, topk_op.values, seq_len, act_type)
            moe_block_res = top.AddOp(
                mlir_gen.get_tensor_type([1, seq_len, self.hidden_size]),
                [shared_output, experts_out],
                loc=self.get_loc(proj_gate + "_moe_block_res", mlir_gen),
                ip=mlir_gen.insert_point).output
            return moe_block_res

        topk = top.TopKOp(mlir_gen.get_tensor_type([1, seq_len, self.num_experts_per_tok]),
                          mlir_gen.get_tensor_type([1, seq_len, self.num_experts_per_tok]),
                          softmax, axis=2, K=self.num_experts_per_tok,
                          use_hau=seq_len > 1,
                          loc=self.get_loc([proj_gate + "_values", proj_gate + "_indices"], mlir_gen),
                          ip=mlir_gen.insert_point)
        routing_scores, expert_ids = topk.values, topk.indices

        # ---- routed experts MLP ----
        if os.environ.get("UOCR_UNFUSE"):
            # Un-fused MoE: 64 per-expert A16MatMul (int4 RTN, SE9-proven) +
            # GatherElements to gather the top-k outputs -> [seq, topk, hidden],
            # the same shape the fused MlpOp would emit. Downstream
            # (routing_scores mul + reduce) is unchanged. See _moe_unfuse_experts.
            experts_mlp = self._moe_unfuse_experts(
                mlir_gen, proj_experts_gate, proj_experts_up, proj_experts_down,
                input_op, expert_ids, seq_len, act_type)
        elif num_split_fused_moe < 2:
            experts_mlp = self.mlp(mlir_gen,
                                   proj_experts_gate, proj_experts_up, proj_experts_down,
                                   input_op, expert_ids, seq_len,
                                   self.hidden_size, self.moe_intermediate_size, act_type,
                                   is_expert=True,
                                   num_experts=self.num_experts,
                                   num_experts_per_tok=self.num_experts_per_tok)
        else:
            split_size = math.ceil(self.moe_intermediate_size / num_split_fused_moe)
            for split_id in range(num_split_fused_moe):
                if split_id == num_split_fused_moe - 1:
                    split_size = self.moe_intermediate_size - split_id * split_size
                experts_mlp_split = self.mlp(mlir_gen,
                                             proj_experts_gate + ".split" + str(split_id),
                                             proj_experts_up + ".split" + str(split_id),
                                             proj_experts_down + ".split" + str(split_id),
                                             input_op, expert_ids, seq_len,
                                             self.hidden_size, split_size, act_type,
                                             is_expert=True,
                                             num_experts=self.num_experts,
                                             num_experts_per_tok=self.num_experts_per_tok)
                if split_id == 0:
                    experts_mlp = experts_mlp_split
                else:
                    experts_mlp = top.AddOp(mlir_gen.get_tensor_type(
                        [seq_len, self.num_experts_per_tok, self.hidden_size]),
                        [experts_mlp, experts_mlp_split],
                        loc=self.get_loc(proj_experts_gate + "_experts_mlp" + str(split_id), mlir_gen),
                        ip=mlir_gen.insert_point).output

        # norm_topk_prob=False for DeepseekV2 -> no renormalize.
        if self.norm_topk_prob:
            routing_scores_sum = top.ReduceOp(mlir_gen.get_tensor_type([1, seq_len, 1]),
                                              routing_scores, axes=[2], keepdims=True,
                                              mode=StringAttr.get("ReduceSum"),
                                              loc=self.get_loc(proj_experts_gate + "_reducesum1", mlir_gen),
                                              ip=mlir_gen.insert_point).output
            routing_scores = top.DivOp(mlir_gen.get_tensor_type([1, seq_len, self.num_experts_per_tok]),
                                       [routing_scores, routing_scores_sum],
                                       loc=self.get_loc(proj_experts_gate + "_div", mlir_gen),
                                       ip=mlir_gen.insert_point).output

        routing_scores_reshape = top.ReshapeOp(
            mlir_gen.get_tensor_type([seq_len, self.num_experts_per_tok, 1]),
            routing_scores, shape=[-1, self.num_experts_per_tok, 1],
            loc=self.get_loc(proj_experts_gate + "_routing_scores", mlir_gen),
            ip=mlir_gen.insert_point).output
        experts_mlp_scores = top.MulOp(
            mlir_gen.get_tensor_type([seq_len, self.num_experts_per_tok, self.hidden_size]),
            [experts_mlp, routing_scores_reshape],
            loc=self.get_loc(proj_experts_gate + "_experts_mlp_scores", mlir_gen),
            ip=mlir_gen.insert_point).output
        experts_mlp_reduce = top.ReduceOp(mlir_gen.get_tensor_type([seq_len, self.hidden_size]),
                                          experts_mlp_scores, axes=[1], keepdims=False,
                                          mode=StringAttr.get("ReduceSum"),
                                          loc=self.get_loc(proj_experts_gate + "_reducesum2", mlir_gen),
                                          ip=mlir_gen.insert_point).output
        experts_mlp_output = top.ReshapeOp(mlir_gen.get_tensor_type([1, seq_len, self.hidden_size]),
                                           experts_mlp_reduce, shape=[1, -1, self.hidden_size],
                                           loc=self.get_loc(proj_experts_gate + "_experts_mlp_output", mlir_gen),
                                           ip=mlir_gen.insert_point).output
        moe_block_res = top.AddOp(mlir_gen.get_tensor_type([1, seq_len, self.hidden_size]),
                                  [shared_output, experts_mlp_output],
                                  loc=self.get_loc(proj_experts_gate + "_moe_block_res", mlir_gen),
                                  ip=mlir_gen.insert_point).output
        return moe_block_res

    def _moe_unfuse_experts(self, mlir_gen, proj_experts_gate, proj_experts_up,
                            proj_experts_down, input_op, expert_ids, seq_len, act_type):
        """Un-fused routed-expert computation (avoids the fused MlpOp, which
        hangs on SE9-16 / crashes cmodel in the dev tpu-mlir build).

        Computes all `num_experts` routed experts as per-expert
        gate/up/down linears (self.linear, quant_mode=None -> MatMulOp ->
        lowered to A16MatMul int4 via RTN, the SE9-proven block_0 path), then
        gathers the top-k expert outputs per token with GatherElements,
        producing [seq, topk, hidden] -- the same shape the fused MlpOp emits
        -- so the stock downstream (routing_scores mul + reduce) is reused.

        Per-expert weights live under the "{experts_path}.{e}.{gud}_proj__uocr_unfuse"
        alias (set in set_linear_weight); gen_block's consolidation only `del`s
        the non-alias ".weight", so the alias survives into the npz.
        """
        E = int(self.num_experts)
        H = int(self.hidden_size)
        inter = int(self.moe_intermediate_size)
        topk = int(self.num_experts_per_tok)
        g_pre, g_suf = proj_experts_gate.split("expert_id")
        u_pre, u_suf = proj_experts_up.split("expert_id")
        d_pre, d_suf = proj_experts_down.split("expert_id")
        T = mlir_gen.get_tensor_type
        L = self.get_loc
        ip = mlir_gen.insert_point
        outs = []
        for e in range(E):
            gp = g_pre + str(e) + g_suf + "__uocr_unfuse"
            up_ = u_pre + str(e) + u_suf + "__uocr_unfuse"
            dp = d_pre + str(e) + d_suf + "__uocr_unfuse"
            g = self.linear(mlir_gen, gp, input_op, [H, inter], [1, seq_len, inter])
            u = self.linear(mlir_gen, up_, input_op, [H, inter], [1, seq_len, inter])
            a = self.activate(mlir_gen, g, act_type, gp)
            m = top.MulOp(T([1, seq_len, inter]), [a, u],
                          loc=L(gp + "_mul", mlir_gen), ip=ip).output
            d = self.linear(mlir_gen, dp, m, [inter, H], [1, seq_len, H])
            outs.append(d)
        # concat 64 x [1,seq,hidden] along axis 0 -> [E, seq, hidden], then
        # permute to [seq, E, hidden] so GatherElements can gather along the
        # E axis (axis=1) per token.
        stacked = top.ConcatOp(T([E, seq_len, H]), outs, axis=0,
                               loc=L(proj_experts_gate + "_unfuse_stack", mlir_gen),
                               ip=ip).output
        permed = top.PermuteOp(T([seq_len, E, H]), stacked, order=[1, 0, 2],
                               loc=L(proj_experts_gate + "_unfuse_perm", mlir_gen),
                               ip=ip).output
        # indices [seq, topk, 1] (broadcast over hidden) for GatherElements
        # axis=1: out[s,k,h] = permed[s, idx[s,k,0], h] = expert(eid[s,k]) of token s
        idx = top.ReshapeOp(T([seq_len, topk, 1]), expert_ids,
                            shape=[seq_len, topk, 1],
                            loc=L(proj_experts_gate + "_unfuse_idx", mlir_gen),
                            ip=ip).output
        experts_mlp = top.GatherElementsOp(T([seq_len, topk, H]),
                                           permed, idx, axis=1,
                                           loc=L(proj_experts_gate + "_unfuse_ge",
                                                 mlir_gen),
                                           ip=ip).output
        return experts_mlp

    def _moe_unfuse_onehot(self, mlir_gen, proj_experts_gate, proj_experts_up,
                           proj_experts_down, input_op, expert_ids, routing_scores,
                           seq_len, act_type):
        """One-hot top-k MoE (SE9-safe, top-k precision). See moe() for the
        rationale. Builds [seq,E,H] from 64 per-expert linears, scatters the
        top-k routing scores to a dense [seq,E] via one-hot (CompareOp(Equal)
        on expert_range=[0..E-1]) + Mul + ReduceSum, then weighted-sums experts.

        Avoids GatherElements / Permute / GatherOp. expert_range weight is
        injected into the block's weight_dict by set_linear_weight.
        """
        E = int(self.num_experts)
        H = int(self.hidden_size)
        inter = int(self.moe_intermediate_size)
        topk = int(self.num_experts_per_tok)
        g_pre, g_suf = proj_experts_gate.split("expert_id")
        u_pre, u_suf = proj_experts_up.split("expert_id")
        d_pre, d_suf = proj_experts_down.split("expert_id")
        T = mlir_gen.get_tensor_type
        L = self.get_loc
        ip = mlir_gen.insert_point
        outs = []
        for e in range(E):
            gp = g_pre + str(e) + g_suf + "__uocr_unfuse"
            up_ = u_pre + str(e) + u_suf + "__uocr_unfuse"
            dp = d_pre + str(e) + d_suf + "__uocr_unfuse"
            g = self.linear(mlir_gen, gp, input_op, [H, inter], [1, seq_len, inter])
            u = self.linear(mlir_gen, up_, input_op, [H, inter], [1, seq_len, inter])
            a = self.activate(mlir_gen, g, act_type, gp)
            m = top.MulOp(T([1, seq_len, inter]), [a, u],
                          loc=L(gp + "_mul", mlir_gen), ip=ip).output
            d = self.linear(mlir_gen, dp, m, [inter, H], [1, seq_len, H])
            d_r = top.ReshapeOp(T([seq_len, 1, H]), d, shape=[seq_len, 1, H],
                                loc=L(gp + "_rsh", mlir_gen), ip=ip).output
            outs.append(d_r)
        stacked = top.ConcatOp(T([seq_len, E, H]), outs, axis=1,
                               loc=L(proj_experts_gate + "_oh_stack", mlir_gen),
                               ip=ip).output  # [seq, E, H]

        # one-hot[seq, topk, E] via CompareOp (avoids Gather):
        # expert_range=[0,1,...,E-1] (i32), indices [seq,topk,1] (i32)
        # Compare Equal -> [seq,topk,E] bool, Cast -> f32 0.0/1.0
        expert_range = mlir_gen.create_weight_op("uocr_expert_range", [E])
        idx_3d = top.ReshapeOp(T([seq_len, topk, 1]), expert_ids,
                               shape=[seq_len, topk, 1],
                               loc=L(proj_experts_gate + "_oh_idx3d", mlir_gen),
                               ip=ip).output
        range_3d = top.ReshapeOp(T([1, 1, E]), expert_range,
                                 shape=[1, 1, E],
                                 loc=L(proj_experts_gate + "_oh_range3d", mlir_gen),
                                 ip=ip).output
        one_hot_f32 = top.CompareOp(T([seq_len, topk, E]), idx_3d, range_3d,
                                    mode=StringAttr.get("Equal"),
                                    loc=L(proj_experts_gate + "_oh_compare", mlir_gen),
                                    ip=ip).output  # [seq, topk, E] f32 (0.0/1.0)
        # routing_scores [1, seq, topk] -> [seq, topk, 1]
        routing_r = top.ReshapeOp(T([seq_len, topk, 1]), routing_scores,
                                  shape=[seq_len, topk, 1],
                                  loc=L(proj_experts_gate + "_oh_rr", mlir_gen),
                                  ip=ip).output
        weighted_oh = top.MulOp(T([seq_len, topk, E]), [one_hot_f32, routing_r],
                                loc=L(proj_experts_gate + "_oh_wmul", mlir_gen),
                                ip=ip).output  # [seq, topk, E] f32
        # reduce topk -> dense routing [seq, E] (top-k positions have score, rest 0)
        routing_dense = top.ReduceOp(T([seq_len, E]), weighted_oh, axes=[1],
                                     keepdims=False, mode=StringAttr.get("ReduceSum"),
                                     loc=L(proj_experts_gate + "_oh_rred", mlir_gen),
                                     ip=ip).output
        # Keep routing_dense in F32 for the weighted sum to avoid Cast-to-BF16
        # (which introduces an int32->bf16 Cast in the lowered TPU dialect for
        # the one-hot indices, potentially causing quantization errors).
        # The stacked expert outputs and the final ReduceSum stay in their
        # natural types (bf16 for expert outputs; f32 accumulation for ReduceSum).
        routing_dense_r = top.ReshapeOp(T([seq_len, E, 1]), routing_dense,
                                        shape=[seq_len, E, 1],
                                        loc=L(proj_experts_gate + "_oh_dr", mlir_gen),
                                        ip=ip).output
        # Cast stacked (bf16 expert outputs) to f32 for the weighted sum
        stacked_f32 = top.CastOp(T([seq_len, E, H]),
                                 stacked,
                                 to="F32",
                                 loc=L(proj_experts_gate + "_oh_stackcast", mlir_gen),
                                 ip=ip).output
        weighted = top.MulOp(T([seq_len, E, H]), [stacked_f32, routing_dense_r],
                             loc=L(proj_experts_gate + "_oh_w", mlir_gen),
                             ip=ip).output  # [seq, E, H]
        out = top.ReduceOp(T([seq_len, H]), weighted, axes=[1], keepdims=False,
                           mode=StringAttr.get("ReduceSum"),
                           loc=L(proj_experts_gate + "_oh_outred", mlir_gen),
                           ip=ip).output  # [seq, H]
        out_4d = top.ReshapeOp(T([1, seq_len, H]), out, shape=[1, -1, H],
                               loc=L(proj_experts_gate + "_oh_out", mlir_gen),
                               ip=ip).output
        return out_4d
    #
    # v1 approximations (see sample README §6):
    #   * SAM attention uses STANDARD scaled-dot-product (no rel_pos_h/w
    #     decomposed bias, no window/global partitioning). rel_pos contributes
    #     to SAM spatial accuracy; dropping it is a v1 trade-off for
    #     tractability. To restore: build the decomposed rel_pos bias
    #     (add_decomposed_rel_pos in deepencoder.py) via reshape+MatMul+gather.
    #   * SAM neck LayerNorm2d (per-channel LN on [B,C,H,W]) is applied as a
    #     standard LayerNorm over the channel axis (transpose CHW->HWC).
    #   * Fixed 1024x1024 input (base / multi-page mode). Gundam 640x640 tiles
    #     need a dynamic-shape variant (TODO).
    # =====================================================================
    @override
    def gen_vit_mlir(self):
        import os
        from mlir.ir import Context
        import mlir.dialects.top as top
        tqdm.write("generate vit mlir (deeplip: SAM+CLIP+projector) ...")
        name = "vit"
        os.makedirs(name, exist_ok=True)
        vit_npz = f"{name}/{name}_top_f32_all_weight.npz"

        in_shape = [1, 3, self.vit_image_size, self.vit_image_size]   # [1,3,1024,1024]
        out_shape = [1, self.clip_seq, self.proj_out]                 # [1, 256, 1280]

        vit_mlir = MLIRImporter([in_shape], [out_shape], name, self.platform,
                                ["F32"], weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, vit_mlir)

        pixel = vit_mlir.create_input_op(L("pixel_values"), 0)

        # ---- SAM ViT-B -> [1, 1024, 16, 16] ----
        sam_out = self._sam_forward(vit_mlir, pixel, T, L, ip)        # [1,1024,16,16]

        # ---- CLIP-L (uses SAM output as patch embeddings) -> [1, 257, 1024] ----
        clip_full = self._clip_forward(vit_mlir, sam_out, T, L, ip)   # [1,257,1024]
        # drop CLS token -> [1,256,1024]
        clip_tok = top.SliceOp(T([1, self.clip_seq, self.clip_embed]), clip_full,
                               vit_mlir.none_op, vit_mlir.none_op, vit_mlir.none_op,
                               offset=[0, 1, 0], steps=[1, 1, 1],
                               ends=[1, 1 + self.clip_seq, self.clip_embed], axes=[],
                               loc=L("clip.nocls"), ip=ip).output

        # ---- concat CLIP[:,1:] + SAM flatten -> [1, 256, 2048] ----
        sam_flat = top.ReshapeOp(T([1, self.clip_seq, self.sam_net3]), sam_out,
                                 shape=[1, self.clip_seq, self.sam_net3],
                                 loc=L("sam.flat"), ip=ip).output      # [1,256,1024]
        concat = top.ConcatOp(T([1, self.clip_seq, self.proj_in]),
                              [clip_tok, sam_flat], axis=2,
                              loc=L("proj.concat"), ip=ip).output       # [1,256,2048]

        # ---- projector Linear 2048 -> 1280 ----
        proj_path = f"{self.vit_path}.projector.layers"
        proj = self.linear(vit_mlir, proj_path, concat,
                           [self.proj_in, self.proj_out], out_shape)   # [1,256,1280]

        vit_mlir.create_return_op([proj])

        # ---- save weights ----
        wd = {}
        self._save_sam_weights(wd)
        self._save_clip_weights(wd)
        self.set_linear_weight(proj_path, wd)
        np.savez(vit_npz, **wd)
        self.save_mlir_module(vit_mlir, name)

    # ----------------------------------------------------------------- SAM
    def _sam_forward(self, m, pixel, T, L, ip):
        import mlir.dialects.top as top
        g = self.sam_grid                                  # 64
        # patch_embed Conv 3->768 k16 s16 -> [1,768,64,64]
        w = m.create_weight_op(f"{self.vit_path}.sam_model.patch_embed.proj.weight",
                               [self.sam_embed, 3, self.sam_patch, self.sam_patch])
        b = m.create_weight_op(f"{self.vit_path}.sam_model.patch_embed.proj.bias", [self.sam_embed])
        x = top.ConvOp(T([1, self.sam_embed, g, g]), pixel, w, b,
                       kernel_shape=[self.sam_patch, self.sam_patch],
                       strides=[self.sam_patch, self.sam_patch], pads=[0, 0, 0, 0],
                       dilations=[1, 1], loc=L("sam.patch"), ip=ip).output  # [1,768,64,64] BCHW
        # + pos_embed [1,64,64,768]: transpose BCHW->BHWC, add, keep BHWC for blocks
        pos = m.create_weight_op(f"{self.vit_path}.sam_model.pos_embed",
                                 [1, g, g, self.sam_embed])
        x_hwc = top.PermuteOp(T([1, g, g, self.sam_embed]), x, order=[0, 2, 3, 1],
                              loc=L("sam.pos.permute"), ip=ip).output
        x_hwc = top.AddOp(T([1, g, g, self.sam_embed]), [x_hwc, pos],
                          loc=L("sam.pos.add"), ip=ip).output
        # 12 blocks (v1: standard attention, no rel_pos / no window)
        for i in range(self.sam_depth):
            x_hwc = self._sam_block(m, x_hwc, i, T, L, ip)
        # neck + net_2 + net_3 on BCHW
        x_chw = top.PermuteOp(T([1, self.sam_embed, g, g]), x_hwc, order=[0, 3, 1, 2],
                              loc=L("sam.neck.permute"), ip=ip).output  # [1,768,64,64]
        x_chw = self._conv_ln(m, x_chw, f"{self.vit_path}.sam_model.neck.0",
                              f"{self.vit_path}.sam_model.neck.1",
                              self.sam_embed, self.sam_neck, 1, 1, 0, T, L, ip)  # [1,256,64,64]
        x_chw = self._conv_ln(m, x_chw, f"{self.vit_path}.sam_model.neck.2",
                              f"{self.vit_path}.sam_model.neck.3",
                              self.sam_neck, self.sam_neck, 3, 1, 1, T, L, ip)
        x_chw = self._conv(m, x_chw, f"{self.vit_path}.sam_model.net_2",
                           self.sam_neck, self.sam_net2, 3, 2, 1, T, L, ip)     # [1,512,32,32]
        x_chw = self._conv(m, x_chw, f"{self.vit_path}.sam_model.net_3",
                           self.sam_net2, self.sam_net3, 3, 2, 1, T, L, ip)     # [1,1024,16,16]
        return x_chw

    def _sam_block(self, m, x_hwc, i, T, L, ip):
        import mlir.dialects.top as top
        g = self.sam_grid
        p = f"{self.vit_path}.sam_model.blocks.{i}"
        shp4 = [1, g, g, self.sam_embed]
        shp3 = [1, g * g, self.sam_embed]
        # pre-norm1 (LayerNorm eps 1e-6) on last dim (C) of [1,g,g,768]
        h = self.layer_norm(m, x_hwc, f"{p}.norm1", 1e-6)
        # BHWC [1,g,g,768] -> seq [1, g*g, 768] for attention
        h_seq = top.ReshapeOp(T(shp3), h, shape=shp3, loc=L(f"sam.b{i}.h2seq"), ip=ip).output
        attn = self._std_attn(m, h_seq, f"{p}.attn", self.sam_embed, self.sam_heads,
                              self.sam_head_dim, shp3, T, L, ip,
                              qkv_name="qkv", out_name="proj")
        attn = top.ReshapeOp(T(shp4), attn, shape=shp4, loc=L(f"sam.b{i}.seq2h"), ip=ip).output
        attn = top.AddOp(T(shp4), [x_hwc, attn], loc=L(f"sam.b{i}.res1"), ip=ip).output
        # mlp
        h2 = self.layer_norm(m, attn, f"{p}.norm2", 1e-6)
        h2_seq = top.ReshapeOp(T(shp3), h2, shape=shp3, loc=L(f"sam.b{i}.h2seq2"), ip=ip).output
        mlp = self._vit_mlp(m, h2_seq, f"{p}.mlp", self.sam_embed, self.sam_mlp,
                            ActType.GELU, shp3, T, L, ip, lin1="lin1", lin2="lin2")
        mlp = top.ReshapeOp(T(shp4), mlp, shape=shp4, loc=L(f"sam.b{i}.seq2h2"), ip=ip).output
        out = top.AddOp(T(shp4), [attn, mlp], loc=L(f"sam.b{i}.res2"), ip=ip).output
        return out

    # ----------------------------------------------------------------- CLIP
    def _clip_forward(self, m, sam_out, T, L, ip):
        import mlir.dialects.top as top
        # SAM [1,1024,16,16] (BCHW) -> permute [0,2,3,1] -> [1,16,16,1024] -> flatten [1,256,1024]
        pe = top.PermuteOp(T([1, self.clip_grid, self.clip_grid, self.clip_embed]), sam_out,
                           order=[0, 2, 3, 1], loc=L("clip.pe.permute"), ip=ip).output
        pe = top.ReshapeOp(T([1, self.clip_seq, self.clip_embed]), pe,
                           shape=[1, self.clip_seq, self.clip_embed],
                           loc=L("clip.pe.reshape"), ip=ip).output       # [1,256,1024]
        # prepend class_embedding [1024] -> [1,257,1024]
        cls = m.create_weight_op(f"{self.vit_path}.vision_model.embeddings.class_embedding",
                                 [1, 1, self.clip_embed])
        x = top.ConcatOp(T([1, self.clip_seq + 1, self.clip_embed]), [cls, pe], axis=1,
                         loc=L("clip.cls.cat"), ip=ip).output             # [1,257,1024]
        # + position_embedding (Embedding 5330 -> interp to 257). v1: slice first 257.
        # TODO: proper get_abs_pos bicubic interp; v1 takes the first 257 learned positions.
        pos = m.create_weight_op(f"{self.vit_path}.vision_model.embeddings.position_embedding.weight",
                                 [1, self.clip_seq + 1, self.clip_embed])
        x = top.AddOp(T([1, self.clip_seq + 1, self.clip_embed]), [x, pos],
                      loc=L("clip.pos.add"), ip=ip).output
        x = self.layer_norm(m, x, f"{self.vit_path}.vision_model.pre_layrnorm", 1e-5)
        seq = self.clip_seq + 1
        for i in range(self.clip_depth):
            x = self._clip_block(m, x, i, seq, T, L, ip)
        return x   # [1,257,1024]

    def _clip_block(self, m, x, i, seq, T, L, ip):
        import mlir.dialects.top as top
        p = f"{self.vit_path}.vision_model.transformer.layers.{i}"
        shp = [1, seq, self.clip_embed]
        h = self.layer_norm(m, x, f"{p}.layer_norm1", 1e-5)
        attn = self._std_attn(m, h, f"{p}.self_attn", self.clip_embed, self.clip_heads,
                              self.clip_head_dim, shp, T, L, ip,
                              qkv_name="qkv_proj", out_name="out_proj")
        x = top.AddOp(T(shp), [x, attn], loc=L(f"clip.b{i}.res1"), ip=ip).output
        h2 = self.layer_norm(m, x, f"{p}.layer_norm2", 1e-5)
        mlp = self._vit_mlp(m, h2, f"{p}.mlp", self.clip_embed, self.clip_mlp,
                            ActType.QUICK_GELU, shp, T, L, ip, lin1="fc1", lin2="fc2")
        return top.AddOp(T(shp), [x, mlp], loc=L(f"clip.b{i}.res2"), ip=ip).output

    # ----------------------------------------------------------- shared ops
    def _std_attn(self, m, in_op, prefix, dim, heads, hd, out_shape, T, L, ip,
                  qkv_name, out_name):
        """Standard MHA: fused qkv Linear -> split -> MatMul -> scale -> softmax
        -> MatMul -> o_proj. in_op [1,N,dim] -> [1,N,dim]."""
        import mlir.dialects.top as top
        qkv = self.linear(m, f"{prefix}.{qkv_name}", in_op, [dim, dim * 3],
                          [1, out_shape[1], dim * 3])                     # [1,N,3*dim]
        q = top.SliceOp(T([1, out_shape[1], dim]), qkv, m.none_op, m.none_op, m.none_op,
                        offset=[0, 0, 0], steps=[1, 1, 1], ends=[1, out_shape[1], dim],
                        axes=[], loc=L(prefix + ".q"), ip=ip).output
        k = top.SliceOp(T([1, out_shape[1], dim]), qkv, m.none_op, m.none_op, m.none_op,
                        offset=[0, 0, dim], steps=[1, 1, 1], ends=[1, out_shape[1], 2 * dim],
                        axes=[], loc=L(prefix + ".k"), ip=ip).output
        v = top.SliceOp(T([1, out_shape[1], dim]), qkv, m.none_op, m.none_op, m.none_op,
                        offset=[0, 0, 2 * dim], steps=[1, 1, 1], ends=[1, out_shape[1], 3 * dim],
                        axes=[], loc=L(prefix + ".v"), ip=ip).output
        N = out_shape[1]
        # reshape [1,N,dim] -> [1,N,heads,hd] -> permute [1,heads,N,hd]
        def split_heads(t, tag):
            t = top.ReshapeOp(T([1, N, heads, hd]), t, shape=[1, N, heads, hd],
                              loc=L(prefix + "." + tag + ".reshape"), ip=ip).output
            return top.PermuteOp(T([1, heads, N, hd]), t, order=[0, 2, 1, 3],
                                 loc=L(prefix + "." + tag + ".permute"), ip=ip).output
        qh, kh, vh = split_heads(q, "q"), split_heads(k, "k"), split_heads(v, "v")
        kt = top.PermuteOp(T([1, heads, hd, N]), kh, order=[0, 1, 3, 2],
                           loc=L(prefix + ".kt"), ip=ip).output
        scores = top.MatMulOp(T([1, heads, N, N]), qh, kt, m.none_op,
                              do_relu=False, is_lora=False, loc=L(prefix + ".qk"), ip=ip).output
        scores = top.MulConstOp(T([1, heads, N, N]), scores,
                                const_val=float(hd ** -0.5),
                                loc=L(prefix + ".scale"), ip=ip).output
        probs = top.SoftmaxOp(T([1, heads, N, N]), scores, axis=3,
                              loc=L(prefix + ".softmax"), ip=ip).output
        av = top.MatMulOp(T([1, heads, N, hd]), probs, vh, m.none_op,
                          do_relu=False, is_lora=False, loc=L(prefix + ".av"), ip=ip).output
        av = top.PermuteOp(T([1, N, heads, hd]), av, order=[0, 2, 1, 3],
                           loc=L(prefix + ".permute_back"), ip=ip).output
        av = top.ReshapeOp(T([1, N, dim]), av, shape=[1, N, dim],
                           loc=L(prefix + ".merge"), ip=ip).output
        return self.linear(m, f"{prefix}.{out_name}", av, [dim, dim], out_shape)

    def _vit_mlp(self, m, in_op, prefix, dim, inter, act, out_shape, T, L, ip,
                 lin1, lin2):
        a = self.linear(m, f"{prefix}.{lin1}", in_op, [dim, inter],
                        [1, out_shape[1], inter])
        a = self.activate(m, a, act, prefix + f".{lin1}")
        return self.linear(m, f"{prefix}.{lin2}", a, [inter, dim], out_shape)

    def _conv(self, m, in_op, prefix, ci, co, k, s, p, T, L, ip):
        import mlir.dialects.top as top
        w = m.create_weight_op(f"{prefix}.weight", [co, ci, k, k])
        b = m.create_weight_op(f"{prefix}.bias", [co]) if self.model.is_exist(prefix + ".bias") \
            else m.none_op
        in_shape = list(in_op.type.shape)
        ho = (in_shape[2] + 2 * p - k) // s + 1
        wo = (in_shape[3] + 2 * p - k) // s + 1
        return top.ConvOp(T([in_shape[0], co, ho, wo]), in_op, w, b,
                          kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p],
                          dilations=[1, 1], loc=L(prefix + ".conv"), ip=ip).output

    def _conv_ln(self, m, in_op, conv_prefix, ln_prefix, ci, co, k, s, p, T, L, ip):
        """Conv2d (bias=False) + LayerNorm2d (per-channel on C). v1: transpose
        CHW->HWC, standard LayerNorm on C (last dim), transpose back."""
        import mlir.dialects.top as top
        y = self._conv(m, in_op, conv_prefix, ci, co, k, s, p, T, L, ip)   # no bias
        sh = list(y.type.shape)                                            # [B,co,H,W]
        y = top.PermuteOp(T([sh[0], sh[2], sh[3], co]), y, order=[0, 2, 3, 1],
                          loc=L(ln_prefix + ".permute"), ip=ip).output      # [B,H,W,co]
        y = self.layer_norm(m, y, ln_prefix, 1e-6)
        return top.PermuteOp(T([sh[0], co, sh[2], sh[3]]), y, order=[0, 3, 1, 2],
                             loc=L(ln_prefix + ".permute_back"), ip=ip).output

    # ----------------------------------------------------------- vit weights
    def _save_sam_weights(self, wd):
        p = f"{self.vit_path}.sam_model"
        self.set_common_weight(f"{p}.patch_embed.proj", wd)
        self.set_common_weight(f"{p}.pos_embed", wd)
        for i in range(self.sam_depth):
            bi = f"{p}.blocks.{i}"
            self.set_common_weight(f"{bi}.norm1", wd)
            self.set_common_weight(f"{bi}.norm2", wd)
            self.set_linear_weight(f"{bi}.attn.qkv", wd)
            self.set_linear_weight(f"{bi}.attn.proj", wd)
            # rel_pos_h/w present but unused in v1 standard attention
            self.set_linear_weight(f"{bi}.mlp.lin1", wd)
            self.set_linear_weight(f"{bi}.mlp.lin2", wd)
        for j in range(4):
            self.set_common_weight(f"{p}.neck.{j}", wd)
        self.set_common_weight(f"{p}.net_2", wd)
        self.set_common_weight(f"{p}.net_3", wd)

    def _save_clip_weights(self, wd):
        p = f"{self.vit_path}.vision_model"
        self.set_common_weight(f"{p}.embeddings.class_embedding", wd)
        self.set_common_weight(f"{p}.embeddings.position_embedding", wd)
        self.set_common_weight(f"{p}.pre_layrnorm", wd)
        for i in range(self.clip_depth):
            li = f"{p}.transformer.layers.{i}"
            self.set_common_weight(f"{li}.layer_norm1", wd)
            self.set_common_weight(f"{li}.layer_norm2", wd)
            self.set_linear_weight(f"{li}.self_attn.qkv_proj", wd)
            self.set_linear_weight(f"{li}.self_attn.out_proj", wd)
            self.set_linear_weight(f"{li}.mlp.fc1", wd)
            self.set_linear_weight(f"{li}.mlp.fc2", wd)
