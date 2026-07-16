# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Falcon-Perception (early-fusion segmentation VLM) converter.

Architecture highlights:
  - single-stack early-fusion transformer (28 layers), no separate ViT
  - fused QKV, QK-norm (unweighted RMSNorm), attention sinks
  - 3D RoPE: head_dim split 64 (1D, theta=10000) + 64 (2D golden, per-head learned freqs)
  - hand-written attention (option D) to obtain logsumexp for the sink scaling
  - squared-ReLU-gate FFN: w2(relu(gate)^2 * up)
  - unweighted RMSNorm before wqkv / w13 / on QK

Compiles the backbone (block + embedding + lm_head), the AnyUp upsampler, the
coord/size/seg/mask heads, and the coord/size Fourier encoders (for re-injection
of predicted coords into the token stream).
Image-patch projection is done host-side (pipeline).
"""

import math
import numpy as np
import torch
from typing_extensions import override

from .LlmConverter import LlmConverter
from .LlmInfo import FALCON_PERCEPTION_INFO, LlmList, WeightType
from mlir.ir import StringAttr
import mlir.dialects.top as top
from transform.MLIRImporter import MLIRImporter, Platform


class FalconPerceptionConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        self.max_pixels = args.max_pixels
        self.max_shape = args.max_shape
        super().__init__(args, config, loader=loader)
        # Falcon-specific dims
        self.rope_half = self.head_dim // 2  # 64  (1D half)
        self.rope_quart = self.rope_half // 2  # 32  (sin/cos dim per half)
        self.n_rep = self.num_attention_heads // self.num_key_value_heads  # 2
        self.attn_scale = self.head_dim**-0.5
        self.q_dim = self.num_attention_heads * self.head_dim  # 2048
        self.kv_dim = self.num_key_value_heads * self.head_dim  # 1024
        # disable fused MLP (we hand-roll the squared-ReLU-gate FFN)
        self.fused_mlp = False
        # segmentation head dims (coord/size/seg decoders + mask einsum)
        c = config
        self.segm_out_dim = getattr(c, "segm_out_dim", 256)
        self.coord_dec_dim = getattr(c, "coord_dec_dim", 8192)
        self.coord_out_dim = getattr(c, "coord_out_dim", 2048)
        # static compile-time max number of seg detections per image (mask_head
        # batch dim; pipeline pads seg tokens up to this and ignores extras)
        self.max_segm_tokens = 16
        # register the heads bmodel nets (coord/size/seg/mask) alongside the
        # backbone; AnyUp is added by gen_anyup_mlir once implemented.
        self.all_gen_mlirs.append(self.gen_heads_mlir)
        self.all_compiles.append(self.compile_heads)
        self.all_gen_mlirs.append(self.gen_anyup_mlir)
        self.all_compiles.append(self.compile_anyup)

    # ------------------------------------------------------------------ config
    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = FALCON_PERCEPTION_INFO
        self.llm_type = "falcon_perception"

    @override
    def get_qtype(self, dtype, bits):
        """Falcon weights are float32 (no quantized release). Map f32 -> 'f32'
        so the deploy backend keeps full float32 compute (the model's
        logsumexp + attention-sink path is numerically sensitive; bf16/f16
        risk overflow / precision loss)."""
        if dtype in (torch.float32, "float32"):
            return "f32"
        return super().get_qtype(dtype, bits)

    @override
    def init_quantization(self):
        """Non-quantized f32 path: the base ModelHandle.init_quantization only
        understands bf16/f16 half-precision and raises on f32. Replicate the
        no-quantconfig branch here with f32 support; defer to the base for the
        actually-quantized (gptq/awq/...) case."""
        c = self.model_info.config
        self.quantization_config = getattr(self.llm_config, c.quantization_config, None)
        if self.quantization_config is None:
            self.quantization_config = getattr(self.config, c.quantization_config, None)
        if self.quantization_config is not None:
            super().init_quantization()
            return
        if self.quantize == "auto":
            raise RuntimeError("No quantization config found, please set quantize type")
        dtype = self.get_dtype()
        real_quantize = self.get_qtype(dtype, 16)
        if real_quantize is None:
            real_quantize = self.quantize
        if "f32" in real_quantize:
            self.half_precision_quantize = "f32"
        elif "bf16" in real_quantize:
            self.half_precision_quantize = "bf16"
        else:
            self.half_precision_quantize = "f16"
        if self.half_precision_quantize not in real_quantize:
            raise RuntimeError(f"Quantize {self.quantize} mismatch with model dtype: {dtype}")

    @override
    def rotary_embedding(self):
        """1D RoPE over the first half of head_dim (64), interleaved pairs.

        Returns cos, sin of shape [seq_length, 1, 32] (half of the 64-dim half).
        Called from LlmConverter.__init__ (before our post-init attrs exist),
        so derive dims from head_dim here and cache them for later use.
        """
        self.rope_half = self.head_dim // 2  # 64  (1D half)
        self.rope_quart = self.rope_half // 2  # 32  (sin/cos dim per half)
        inv_freq = 1.0 / (self.rope_theta
                          **(np.arange(0, self.rope_half, 2, dtype=np.float32) / self.rope_half))
        t = np.arange(self.seq_length, dtype=np.float32)
        freqs = np.outer(t, inv_freq)  # [seq, 32]
        cos = np.cos(freqs).astype(np.float32).reshape(self.seq_length, 1, self.rope_quart)
        sin = np.sin(freqs).astype(np.float32).reshape(self.seq_length, 1, self.rope_quart)
        return cos, sin

    # ------------------------------------------------------------------ helpers
    def _ones_weight(self, mlir_gen, dim, name):
        return mlir_gen.create_weight_op(name, [dim])

    def _rms_norm_unweighted(self, mlir_gen, in_op, gamma_op, name, eps=None):
        """RMSNorm with a constant gamma (F.rms_norm, no learned weight).

        gamma_op must be a weight op created once per block (unique name) with
        a broadcast shape matching in_op's trailing dims; reusing one op for
        multiple norm sites avoids the top-level weight-name uniqueness assert.
        """
        ip = mlir_gen.insert_point
        input_shape = list(in_op.type.shape)
        eps = self.rms_norm_eps if eps is None else eps
        return top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                             in_op,
                             gamma_op,
                             eps=eps,
                             weight_keep_f32=False,
                             loc=self.get_loc(name, mlir_gen),
                             ip=ip).output

    def _repeat_kv(self, mlir_gen, kv_op, name):
        """GQA: repeat kv from n_kv_heads to n_heads along the head axis.

        kv_op: [B, S, n_kv_heads, head_dim] -> [B, S, n_heads, head_dim]
        """
        ip = mlir_gen.insert_point
        shp = list(kv_op.type.shape)
        B, S, Hkv, D = shp
        # [B, S, Hkv, 1, D] -> tile head-rep axis -> [B, S, Hkv, n_rep, D] -> reshape
        rshp = top.ReshapeOp(mlir_gen.get_tensor_type([B, S, Hkv, 1, D]),
                             kv_op,
                             shape=[B, S, Hkv, 1, D],
                             loc=self.get_loc(name + ".rshp", mlir_gen),
                             ip=ip).output
        tiled = top.TileOp(mlir_gen.get_tensor_type([B, S, Hkv, self.n_rep, D]),
                           rshp,
                           tile=[1, 1, 1, self.n_rep, 1],
                           loc=self.get_loc(name + ".tile", mlir_gen),
                           ip=ip).output
        out = top.ReshapeOp(mlir_gen.get_tensor_type([B, S, Hkv * self.n_rep, D]),
                            tiled,
                            shape=[B, S, Hkv * self.n_rep, D],
                            loc=self.get_loc(name + ".rshp2", mlir_gen),
                            ip=ip).output
        return out

    def _to_bhd(self, mlir_gen, q_op, name):
        """[B, S, H, D] -> [H, S, D] (batch=H via hdim_is_batch), B assumed 1."""
        ip = mlir_gen.insert_point
        B, S, H, D = list(q_op.type.shape)
        perm = top.PermuteOp(mlir_gen.get_tensor_type([H, B, S, D]),
                             q_op,
                             order=[2, 0, 1, 3],
                             loc=self.get_loc(name + ".perm", mlir_gen),
                             ip=ip).output
        out = top.ReshapeOp(mlir_gen.get_tensor_type([H, S, D]),
                            perm,
                            shape=[H, -1, D],
                            loc=self.get_loc(name + ".rshp", mlir_gen),
                            ip=ip).output
        return out

    def _from_bhd(self, mlir_gen, out_op, B, S, H, D, name):
        """[H, S, D] -> [B, S, H, D]."""
        ip = mlir_gen.insert_point
        rshp = top.ReshapeOp(mlir_gen.get_tensor_type([1, H, S, D]),
                             out_op,
                             shape=[1, H, -1, D],
                             loc=self.get_loc(name + ".rshp", mlir_gen),
                             ip=ip).output
        perm = top.PermuteOp(mlir_gen.get_tensor_type([B, S, H, D]),
                             rshp,
                             order=[0, 2, 1, 3],
                             loc=self.get_loc(name + ".perm", mlir_gen),
                             ip=ip).output
        return perm

    def _expand_pairs(self, mlir_gen, op, name):
        """Repeat per-pair sin/cos [..., half] -> full-dim [..., 2*half] as
        [s0,s0,s1,s1,...]. RopeOp(interleaved_pairs) requires full-dim repeated
        sin/cos (see Top Rope.cpp reference: out = temp*w0 + in*w1)."""
        ip = mlir_gen.insert_point
        shp = list(op.type.shape)
        half = shp[-1]
        lead = shp[:-1]
        r = top.ReshapeOp(mlir_gen.get_tensor_type(lead + [half, 1]),
                          op,
                          shape=lead + [half, 1],
                          loc=self.get_loc(name + ".r", mlir_gen),
                          ip=ip).output
        t = top.TileOp(mlir_gen.get_tensor_type(lead + [half, 2]),
                       r,
                       tile=[1] * len(lead) + [1, 2],
                       loc=self.get_loc(name + ".tile", mlir_gen),
                       ip=ip).output
        return top.ReshapeOp(mlir_gen.get_tensor_type(lead + [2 * half]),
                             t,
                             shape=lead + [2 * half],
                             loc=self.get_loc(name + ".full", mlir_gen),
                             ip=ip).output

    def _rope_3d(self, mlir_gen, q_op, k_op, pos_op, gcos_op, gsin_op, name):
        """3D RoPE: split head_dim into 1D half + 2D golden half, rotate each, concat.

        q_op/k_op: [B, S, H, head_dim]  (H = n_heads; k already repeated to H heads)
        pos_op:    [B, S] int32 (1D positions)
        gcos_op/gsin_op: [B, S, H, rope_quart] (golden 2D, runtime; text tokens -> cos=1,sin=0)
        """
        ip = mlir_gen.insert_point
        B, S, H, D = list(q_op.type.shape)
        half = self.rope_half  # 64
        quart = self.rope_quart  # 32

        def split_last(op, lo, hi, tag):
            shp = list(op.type.shape)
            half_shp = shp[:-1] + [hi - lo]
            # SliceOp: offset/steps/ends apply to leading slice_dims axes; the
            # non-sliced axes take offset=0 and end=full. Slice last axis [lo:hi].
            return top.SliceOp(mlir_gen.get_tensor_type(half_shp),
                               op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               mlir_gen.none_op,
                               offset=[0] * (len(shp) - 1) + [lo],
                               steps=[1] * len(shp),
                               ends=shp[:-1] + [hi],
                               axes=[],
                               loc=self.get_loc(name + tag, mlir_gen),
                               ip=ip).output

        # 1D cos/sin: gather from baked table by pos -> [B, S, 1, quart]
        cos_w = mlir_gen.create_weight_op("rotary_cos.weight", [self.seq_length, 1, quart])
        sin_w = mlir_gen.create_weight_op("rotary_sin.weight", [self.seq_length, 1, quart])
        cos1d = top.GatherOp(mlir_gen.get_tensor_type([B, S, 1, quart]),
                             cos_w,
                             pos_op,
                             axis=0,
                             loc=self.get_loc(name + ".cos1d", mlir_gen),
                             ip=ip).output
        sin1d = top.GatherOp(mlir_gen.get_tensor_type([B, S, 1, quart]),
                             sin_w,
                             pos_op,
                             axis=0,
                             loc=self.get_loc(name + ".sin1d", mlir_gen),
                             ip=ip).output
        # RopeOp wants full-dim repeated sin/cos [.., half] -> [.., 2*half]
        sin1d = self._expand_pairs(mlir_gen, sin1d, name + ".sin1d")
        cos1d = self._expand_pairs(mlir_gen, cos1d, name + ".cos1d")
        gsin_full = self._expand_pairs(mlir_gen, gsin_op, name + ".gsin")
        gcos_full = self._expand_pairs(mlir_gen, gcos_op, name + ".gcos")

        def rope_both(op, tag):
            oshp = list(op.type.shape)
            t = split_last(op, 0, half, tag + ".t")
            hw = split_last(op, half, D, tag + ".hw")
            t = top.RopeOp(mlir_gen.get_tensor_type(list(t.type.shape)),
                           t,
                           sin1d,
                           cos1d,
                           rope_mode=StringAttr.get("interleaved_pairs"),
                           loc=self.get_loc(name + tag + ".rope_t", mlir_gen),
                           ip=ip).output
            hw = top.RopeOp(mlir_gen.get_tensor_type(list(hw.type.shape)),
                            hw,
                            gsin_full,
                            gcos_full,
                            rope_mode=StringAttr.get("interleaved_pairs"),
                            loc=self.get_loc(name + tag + ".rope_hw", mlir_gen),
                            ip=ip).output
            return top.ConcatOp(mlir_gen.get_tensor_type(oshp), [t, hw],
                                axis=3,
                                loc=self.get_loc(name + tag + ".cat", mlir_gen),
                                ip=ip).output

        # distinct loc-name prefixes per tensor: q and k share the same rope
        # graph, so reusing loc names (.rope_t etc.) collides and trips the
        # all_names uniqueness check during --init.
        q_op = rope_both(q_op, ".q")
        k_op = rope_both(k_op, ".k")
        return q_op, k_op

    def _attention(self, mlir_gen, q_op, k_op, v_op, mask_op, sinks_op, S, MK, name):
        """Hand-written attention with logsumexp + attention sink (option D).

        q: [B, S, H, D]   k,v: [B, MK, H, D] (already repeated to H heads)
        mask: [B, 1, S, MK] (additive, runtime) or None
        sinks: [H]
        returns: [B, S, H*D]
        """
        ip = mlir_gen.insert_point
        B = list(q_op.type.shape)[0]
        H, D = self.num_attention_heads, self.head_dim

        q = self._to_bhd(mlir_gen, q_op, name + ".q")  # [H, S, D]
        k = self._to_bhd(mlir_gen, k_op, name + ".k")  # [H, MK, D]
        v = self._to_bhd(mlir_gen, v_op, name + ".v")  # [H, MK, D]

        # scores = q @ k^T * scale  -> [H, S, MK]
        scores = top.MatMulOp(mlir_gen.get_tensor_type([H, S, MK]),
                              q,
                              k,
                              mlir_gen.none_op,
                              right_transpose=True,
                              hdim_is_batch=True,
                              do_relu=False,
                              loc=self.get_loc(name + ".qk", mlir_gen),
                              ip=ip).output
        scores = top.MulConstOp(mlir_gen.get_tensor_type([H, S, MK]),
                                scores,
                                const_val=self.attn_scale,
                                loc=self.get_loc(name + ".scale", mlir_gen),
                                ip=ip).output
        if mask_op is not None:
            mshp = list(mask_op.type.shape)  # [B,1,S,MK]
            mask2 = top.ReshapeOp(mlir_gen.get_tensor_type([1, S, MK]),
                                  mask_op,
                                  shape=[1, S, MK],
                                  loc=self.get_loc(name + ".mask.rshp", mlir_gen),
                                  ip=ip).output
            scores = top.AddOp(mlir_gen.get_tensor_type([H, S, MK]), [scores, mask2],
                               loc=self.get_loc(name + ".mask.add", mlir_gen),
                               ip=ip).output

        # logsumexp over MK
        m = top.ReduceOp(mlir_gen.get_tensor_type([H, S, 1]),
                         scores,
                         axes=[2],
                         keepdims=True,
                         mode=StringAttr.get("ReduceMax"),
                         loc=self.get_loc(name + ".max", mlir_gen),
                         ip=ip).output
        negm = top.MulConstOp(mlir_gen.get_tensor_type([H, S, 1]),
                              m,
                              const_val=-1.0,
                              loc=self.get_loc(name + ".negm", mlir_gen),
                              ip=ip).output
        smm = top.AddOp(mlir_gen.get_tensor_type([H, S, MK]), [
            scores,
            top.TileOp(mlir_gen.get_tensor_type([H, S, MK]),
                       negm,
                       tile=[1, 1, MK],
                       loc=self.get_loc(name + ".bcast", mlir_gen),
                       ip=ip).output
        ],
                        loc=self.get_loc(name + ".smm", mlir_gen),
                        ip=ip).output
        e = top.ExpOp(mlir_gen.get_tensor_type([H, S, MK]),
                      smm,
                      loc=self.get_loc(name + ".exp", mlir_gen),
                      ip=ip).output
        s = top.ReduceOp(mlir_gen.get_tensor_type([H, S, 1]),
                         e,
                         axes=[2],
                         keepdims=True,
                         mode=StringAttr.get("ReduceSum"),
                         loc=self.get_loc(name + ".sum", mlir_gen),
                         ip=ip).output
        logs = top.LogOp(mlir_gen.get_tensor_type([H, S, 1]),
                         s,
                         loc=self.get_loc(name + ".log", mlir_gen),
                         ip=ip).output
        lse = top.AddOp(mlir_gen.get_tensor_type([H, S, 1]), [logs, m],
                        loc=self.get_loc(name + ".lse", mlir_gen),
                        ip=ip).output

        # w = exp(scores - lse)
        neglse = top.MulConstOp(mlir_gen.get_tensor_type([H, S, 1]),
                                lse,
                                const_val=-1.0,
                                loc=self.get_loc(name + ".neglse", mlir_gen),
                                ip=ip).output
        slse = top.AddOp(mlir_gen.get_tensor_type([H, S, MK]), [
            scores,
            top.TileOp(mlir_gen.get_tensor_type([H, S, MK]),
                       neglse,
                       tile=[1, 1, MK],
                       loc=self.get_loc(name + ".bcast2", mlir_gen),
                       ip=ip).output
        ],
                         loc=self.get_loc(name + ".slse", mlir_gen),
                         ip=ip).output
        w = top.ExpOp(mlir_gen.get_tensor_type([H, S, MK]),
                      slse,
                      loc=self.get_loc(name + ".w", mlir_gen),
                      ip=ip).output

        # out = w @ v -> [H, S, D]
        out = top.MatMulOp(mlir_gen.get_tensor_type([H, S, D]),
                           w,
                           v,
                           mlir_gen.none_op,
                           hdim_is_batch=True,
                           do_relu=False,
                           loc=self.get_loc(name + ".wv", mlir_gen),
                           ip=ip).output

        # attention sink: out *= sigmoid(lse - sinks)
        sinks2 = top.ReshapeOp(mlir_gen.get_tensor_type([H, 1, 1]),
                               sinks_op,
                               shape=[H, 1, 1],
                               loc=self.get_loc(name + ".sinks.rshp", mlir_gen),
                               ip=ip).output
        lse_ms = top.AddOp(mlir_gen.get_tensor_type([H, S, 1]), [
            lse,
            top.MulConstOp(mlir_gen.get_tensor_type([H, 1, 1]),
                           sinks2,
                           const_val=-1.0,
                           loc=self.get_loc(name + ".nsink", mlir_gen),
                           ip=ip).output
        ],
                           loc=self.get_loc(name + ".lse_ms", mlir_gen),
                           ip=ip).output
        sink_scale = top.SigmoidOp(mlir_gen.get_tensor_type([H, S, 1]),
                                   lse_ms,
                                   loc=self.get_loc(name + ".sink_sig", mlir_gen),
                                   ip=ip).output
        out = top.MulOp(mlir_gen.get_tensor_type([H, S, D]), [
            out,
            top.TileOp(mlir_gen.get_tensor_type([H, S, D]),
                       sink_scale,
                       tile=[1, 1, D],
                       loc=self.get_loc(name + ".sink_bcast", mlir_gen),
                       ip=ip).output
        ],
                        loc=self.get_loc(name + ".sink_mul", mlir_gen),
                        ip=ip).output

        # [H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        out = self._from_bhd(mlir_gen, out, B, S, H, D, name + ".out")
        out = top.ReshapeOp(mlir_gen.get_tensor_type([B, S, H * D]),
                            out,
                            shape=[B, S, H * D],
                            loc=self.get_loc(name + ".flat", mlir_gen),
                            ip=ip).output
        return out

    def _ffn(self, mlir_gen, in_op, wgate_path, wup_path, w2_path, gamma_op, S, name):
        """squared-ReLU-gate FFN: w2(relu(gate)^2 * up) with unweighted pre-norm.

        gate/up come from the HF-interleaved w13 weight, deinterleaved at
        weight-storage time into two contiguous weights (w_gate/w_up) — a
        stride-2 SliceOp on the fused matmul output is mangled by the
        MatMulSlice lowering, so we use two separate matmuls instead.
        """
        ip = mlir_gen.insert_point
        B = list(in_op.type.shape)[0]
        hidden = self.hidden_size
        inter = self.intermediate_size
        x = self._rms_norm_unweighted(mlir_gen, in_op, gamma_op, name + ".norm")
        w_gate = mlir_gen.create_weight_op(wgate_path, [hidden, inter])
        w_up = mlir_gen.create_weight_op(wup_path, [hidden, inter])
        gate = top.MatMulOp(mlir_gen.get_tensor_type([B, S, inter]),
                            x,
                            w_gate,
                            mlir_gen.none_op,
                            do_relu=False,
                            loc=self.get_loc(name + ".gate", mlir_gen),
                            ip=ip).output
        up = top.MatMulOp(mlir_gen.get_tensor_type([B, S, inter]),
                          x,
                          w_up,
                          mlir_gen.none_op,
                          do_relu=False,
                          loc=self.get_loc(name + ".up", mlir_gen),
                          ip=ip).output
        g = top.ReluOp(mlir_gen.get_tensor_type([B, S, inter]),
                       gate,
                       loc=self.get_loc(name + ".relu", mlir_gen),
                       ip=ip).output
        g2 = top.MulOp(mlir_gen.get_tensor_type([B, S, inter]), [g, g],
                       loc=self.get_loc(name + ".sq", mlir_gen),
                       ip=ip).output
        h = top.MulOp(mlir_gen.get_tensor_type([B, S, inter]), [g2, up],
                      loc=self.get_loc(name + ".gate_up", mlir_gen),
                      ip=ip).output
        w2 = mlir_gen.create_weight_op(w2_path, [inter, hidden])
        down = top.MatMulOp(mlir_gen.get_tensor_type([B, S, hidden]),
                            h,
                            w2,
                            mlir_gen.none_op,
                            do_relu=False,
                            loc=self.get_loc(name + ".w2", mlir_gen),
                            ip=ip).output
        return down

    # ------------------------------------------------------------------ block
    @override
    def gen_block_mlir(self, idx: int):
        from tqdm import tqdm
        tqdm.write(f"generate falcon block_{idx} mlir ...")
        ip_top = None
        TOP = f"layers.{idx}."
        wqkv = TOP + "attention.wqkv"
        wo = TOP + "attention.wo"
        sinks = TOP + "attention.sinks"
        w13 = TOP + "feed_forward.w13"
        w2 = TOP + "feed_forward.w2"

        # ---- save weights for this block ----
        weight_file = f"block_{idx}_top_weights.npz"
        wd = {
            "rotary_cos.weight": self.cos,
            "rotary_sin.weight": self.sin,
            "ones_1024": np.ones((1024, ), dtype=np.float32),
            "ones_128": np.ones((128, ), dtype=np.float32),
        }
        self.set_linear_weight(wqkv, wd)
        self.set_linear_weight(wo, wd)
        # w13 is interleaved in HF (even cols = gate, odd = up). A stride-2
        # SliceOp on the matmul output is mangled by the MatMulSlice lowering
        # (treated as a contiguous slice → wrong gate/up). Pre-deinterleave the
        # (transposed) weight into separate contiguous gate/up weights and use
        # two matmuls. set_linear_weight loads it transposed to [in, out].
        self.set_linear_weight(w13, wd)
        w13_full = wd[w13 + ".weight"]  # [hidden, 2*inter]
        wd[w13 + "_gate.weight"] = np.ascontiguousarray(w13_full[:, 0::2].copy())
        wd[w13 + "_up.weight"] = np.ascontiguousarray(w13_full[:, 1::2].copy())
        del wd[w13 + ".weight"]
        self.set_linear_weight(w2, wd)
        # sinks is a bare nn.Parameter (no ".weight" suffix in safetensors)
        wd[sinks + ".weight"] = np.ascontiguousarray(self.model.read(sinks).astype(np.float32))
        if idx == self.num_layers - 1:
            self.set_common_weight(self.model_info.weights[LlmList.NORM], wd, self.rmsnorm_type)
        self.weight_keys.extend(list(wd.keys()))
        np.savez(weight_file, **wd)

        H, D = self.num_attention_heads, self.head_dim
        Hkv = self.num_key_value_heads
        hidden = self.hidden_size

        # =================== prefill block ===================
        def gen_block():
            name = f"block_{idx}"
            L = self.max_input_length
            in_shp = [1, L, hidden]
            pos_shp = [1, L]
            gcos_shp = [1, L, H, self.rope_quart]
            mask_shp = [1, 1, L, L]
            # kv cache stores n_heads (16) heads: HF repeats k,v to 16 BEFORE the
            # 2D golden RoPE (16 distinct per-head freqs), so the 16-head post-RoPE
            # k cannot be compressed to n_kv_heads. Cache 16-head k,v.
            kv_shp = [1, L, H, D]
            inter = self.intermediate_size
            out_shps = [in_shp, kv_shp, kv_shp]
            block = MLIRImporter([in_shp, pos_shp, gcos_shp, gcos_shp, mask_shp],
                                 out_shps,
                                 name,
                                 self.platform, ["F32", "INT32", "F32", "F32", "F32"],
                                 weight_file=f"../{weight_file}")
            T = block.get_tensor_type
            Lc = lambda n: self.get_loc(n, block)
            ip = block.insert_point

            in0 = block.create_input_op(Lc("input_states"), 0)
            pos = block.create_input_op(Lc("position_ids"), 1)
            gcos = block.create_input_op(Lc("golden_cos"), 2)
            gsin = block.create_input_op(Lc("golden_sin"), 3)
            mask = block.create_input_op(Lc("attention_mask"), 4)

            # constant-ones gamma for the unweighted RMSNorms (created once,
            # reused; duplicate top-level weight names would trip saveWeight)
            ones_1024_op = block.create_weight_op("ones_1024", [1, 1, hidden])
            ones_128_op = block.create_weight_op("ones_128", [1, 1, 1, D])

            # pre-attn unweighted rmsnorm
            x = self._rms_norm_unweighted(block, in0, ones_1024_op, "pre_attn_norm")
            # fused QKV
            wqkv_op = block.create_weight_op(wqkv + ".weight",
                                             [hidden, self.q_dim + 2 * self.kv_dim])
            qkv = top.MatMulOp(T([1, L, self.q_dim + 2 * self.kv_dim]),
                               x,
                               wqkv_op,
                               block.none_op,
                               do_relu=False,
                               loc=Lc(wqkv),
                               ip=ip).output
            q = top.SliceOp(T([1, L, self.q_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, 0],
                            steps=[1, 1, 1],
                            ends=[1, L, self.q_dim],
                            axes=[],
                            loc=Lc("q"),
                            ip=ip).output
            k = top.SliceOp(T([1, L, self.kv_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, self.q_dim],
                            steps=[1, 1, 1],
                            ends=[1, L, self.q_dim + self.kv_dim],
                            axes=[],
                            loc=Lc("k"),
                            ip=ip).output
            v = top.SliceOp(T([1, L, self.kv_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, self.q_dim + self.kv_dim],
                            steps=[1, 1, 1],
                            ends=[1, L, self.q_dim + 2 * self.kv_dim],
                            axes=[],
                            loc=Lc("v"),
                            ip=ip).output
            q = top.ReshapeOp(T([1, L, H, D]), q, shape=[1, -1, H, D], loc=Lc("q.rshp"),
                              ip=ip).output
            k = top.ReshapeOp(T([1, L, Hkv, D]), k, shape=[1, -1, Hkv, D], loc=Lc("k.rshp"),
                              ip=ip).output
            v = top.ReshapeOp(T([1, L, Hkv, D]), v, shape=[1, -1, Hkv, D], loc=Lc("v.rshp"),
                              ip=ip).output
            # QK-norm (unweighted)
            q = self._rms_norm_unweighted(block, q, ones_128_op, "q_norm")
            k = self._rms_norm_unweighted(block, k, ones_128_op, "k_norm")
            # Repeat k,v to n_heads BEFORE RoPE: HF repeats first, then applies
            # the 2D golden RoPE with 16 distinct per-head freqs. Caching the
            # 16-head post-RoPE k matches HF (KVCache uses n_heads).
            k = self._repeat_kv(block, k, "k16")
            v = self._repeat_kv(block, v, "v16")
            # 3D RoPE (q,k both 16-head)
            q, k = self._rope_3d(block, q, k, pos, gcos, gsin, "rope")
            # attention + sink
            sinks_op = block.create_weight_op(sinks + ".weight", [H])
            attn = self._attention(block, q, k, v, mask, sinks_op, L, L, "attn")
            wo_op = block.create_weight_op(wo + ".weight", [self.q_dim, hidden])
            o = top.MatMulOp(T([1, L, hidden]),
                             attn,
                             wo_op,
                             block.none_op,
                             do_relu=False,
                             loc=Lc(wo),
                             ip=ip).output
            o = top.AddOp(T([1, L, hidden]), [in0, o], loc=Lc("attn_res"), ip=ip).output
            # FFN
            down = self._ffn(block, o, w13 + "_gate.weight", w13 + "_up.weight", w2 + ".weight",
                             ones_1024_op, L, "ffn")
            out = top.AddOp(T([1, L, hidden]), [o, down], loc=Lc("ffn_res"), ip=ip).output
            if idx == self.num_layers - 1:
                out = self.rms_norm(block, out, self.model_info.weights[LlmList.NORM])
            rets = [out, k, v]
            block.create_return_op(rets)
            self.save_mlir_module(block, name)

        # =================== decode block ===================
        def gen_block_cache():
            name = f"block_cache_{idx}"
            L = 1
            MK = self.seq_length + 1
            in_shp = [self.batch, L, hidden]
            pos_shp = [self.batch, L]
            gcos_shp = [self.batch, L, H, self.rope_quart]
            mask_shp = [self.batch, 1, L, MK]
            # history k/v cached at n_heads (16) heads (see gen_block note).
            hist_shp = [self.batch, self.seq_length, H, D]
            kv_shp = [self.batch, L, H, D]
            block = MLIRImporter(
                [in_shp, pos_shp, gcos_shp, gcos_shp, mask_shp, hist_shp, hist_shp],
                [in_shp, kv_shp, kv_shp],
                name,
                self.platform, ["F32", "INT32", "F32", "F32", "F32", "F32", "F32"],
                weight_file=f"../{weight_file}")
            T = block.get_tensor_type
            Lc = lambda n: self.get_loc(n, block)
            ip = block.insert_point

            in0 = block.create_input_op(Lc("input_states"), 0)
            pos = block.create_input_op(Lc("position_ids"), 1)
            gcos = block.create_input_op(Lc("golden_cos"), 2)
            gsin = block.create_input_op(Lc("golden_sin"), 3)
            mask = block.create_input_op(Lc("attention_mask"), 4)
            hk = block.create_input_op(Lc("history_k"), 5)
            hv = block.create_input_op(Lc("history_v"), 6)

            ones_1024_op = block.create_weight_op("ones_1024", [1, 1, hidden])
            ones_128_op = block.create_weight_op("ones_128", [1, 1, 1, D])

            x = self._rms_norm_unweighted(block, in0, ones_1024_op, "pre_attn_norm")
            wqkv_op = block.create_weight_op(wqkv + ".weight",
                                             [hidden, self.q_dim + 2 * self.kv_dim])
            qkv = top.MatMulOp(T([self.batch, L, self.q_dim + 2 * self.kv_dim]),
                               x,
                               wqkv_op,
                               block.none_op,
                               do_relu=False,
                               loc=Lc(wqkv),
                               ip=ip).output
            q = top.SliceOp(T([self.batch, L, self.q_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, 0],
                            steps=[1, 1, 1],
                            ends=[self.batch, L, self.q_dim],
                            axes=[],
                            loc=Lc("q"),
                            ip=ip).output
            k = top.SliceOp(T([self.batch, L, self.kv_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, self.q_dim],
                            steps=[1, 1, 1],
                            ends=[self.batch, L, self.q_dim + self.kv_dim],
                            axes=[],
                            loc=Lc("k"),
                            ip=ip).output
            v = top.SliceOp(T([self.batch, L, self.kv_dim]),
                            qkv,
                            block.none_op,
                            block.none_op,
                            block.none_op,
                            offset=[0, 0, self.q_dim + self.kv_dim],
                            steps=[1, 1, 1],
                            ends=[self.batch, L, self.q_dim + 2 * self.kv_dim],
                            axes=[],
                            loc=Lc("v"),
                            ip=ip).output
            q = top.ReshapeOp(T([self.batch, L, H, D]),
                              q,
                              shape=[self.batch, -1, H, D],
                              loc=Lc("q.rshp"),
                              ip=ip).output
            k = top.ReshapeOp(T([self.batch, L, Hkv, D]),
                              k,
                              shape=[self.batch, -1, Hkv, D],
                              loc=Lc("k.rshp"),
                              ip=ip).output
            v = top.ReshapeOp(T([self.batch, L, Hkv, D]),
                              v,
                              shape=[self.batch, -1, Hkv, D],
                              loc=Lc("v.rshp"),
                              ip=ip).output
            q = self._rms_norm_unweighted(block, q, ones_128_op, "q_norm")
            k = self._rms_norm_unweighted(block, k, ones_128_op, "k_norm")
            # repeat new k,v to n_heads before RoPE (matches HF + prefill block)
            k = self._repeat_kv(block, k, "k16")
            v = self._repeat_kv(block, v, "v16")
            q, k = self._rope_3d(block, q, k, pos, gcos, gsin, "rope")

            # concat history (16-head) + new k/v (16-head); already n_heads
            kall = top.ConcatOp(T([self.batch, MK, H, D]), [hk, k],
                                axis=1,
                                loc=Lc("k.concat"),
                                ip=ip).output
            vall = top.ConcatOp(T([self.batch, MK, H, D]), [hv, v],
                                axis=1,
                                loc=Lc("v.concat"),
                                ip=ip).output
            sinks_op = block.create_weight_op(sinks + ".weight", [H])
            attn = self._attention(block, q, kall, vall, mask, sinks_op, L, MK, "attn")
            wo_op = block.create_weight_op(wo + ".weight", [self.q_dim, hidden])
            o = top.MatMulOp(T([self.batch, L, hidden]),
                             attn,
                             wo_op,
                             block.none_op,
                             do_relu=False,
                             loc=Lc(wo),
                             ip=ip).output
            o = top.AddOp(T([self.batch, L, hidden]), [in0, o], loc=Lc("attn_res"), ip=ip).output
            down = self._ffn(block, o, w13 + "_gate.weight", w13 + "_up.weight", w2 + ".weight",
                             ones_1024_op, L, "ffn")
            out = top.AddOp(T([self.batch, L, hidden]), [o, down], loc=Lc("ffn_res"), ip=ip).output
            if idx == self.num_layers - 1:
                out = self.rms_norm(block, out, self.model_info.weights[LlmList.NORM])
            block.create_return_op([out, k, v])
            self.save_mlir_module(block, name)

        gen_block()
        gen_block_cache()

    # ------------------------------------------------------------------ heads
    # coord/size/seg output heads + mask einsum. These are small, pure
    # feed-forward/conv nets that run interleaved with the decode loop
    # (coord/size/seg) or once after prefill (mask). conv_segm is folded into
    # the AnyUp bmodel (its output is AnyUp's `features` input).
    def gen_heads_mlir(self):
        from tqdm import tqdm
        tqdm.write("generate falcon heads mlir ...")
        weight_file = "heads_weights.npz"
        wd = {}
        # BboxDecoder (coord/size): no bias
        self.set_linear_weight("coord_decoder.w1", wd)
        self.set_linear_weight("coord_decoder.w2", wd)
        self.set_linear_weight("size_decoder.w1", wd)
        self.set_linear_weight("size_decoder.w2", wd)
        # SegmDecoder (proj_segm): layers 0/1 have bias, pixel_layer no bias
        self.set_linear_weight("proj_segm.layers.0", wd)
        self.set_linear_weight("proj_segm.layers.1", wd)
        self.set_linear_weight("proj_segm.pixel_layer", wd)
        # FourierEncoder (coord/size): embed Linear(2->feat/2), transform Linear(feat->hidden), no bias
        self.set_linear_weight("coord_encoder.embed", wd)
        self.set_linear_weight("coord_encoder.transform", wd)
        self.set_linear_weight("size_encoder.embed", wd)
        self.set_linear_weight("size_encoder.transform", wd)
        self.weight_keys.extend(list(wd.keys()))
        np.savez(weight_file, **wd)

        hidden = self.hidden_size  # 1024
        dec_hidden = self.coord_dec_dim  # 8192
        out_dim = self.coord_out_dim  # 2048 (= 2 * 1024)
        segm_out = self.segm_out_dim  # 256
        Kmax = self.max_segm_tokens
        H, W = self.max_shape[0], self.max_shape[1]
        wf = f"../{weight_file}"

        # ---- coord_head / size_head : BboxDecoder  w2(relu(w1(x))^2) ----
        def gen_bbox_head(name, w1p, w2p):
            m = MLIRImporter([[1, hidden]], [[1, 2, out_dim // 2]],
                             name,
                             self.platform, ["F32"],
                             weight_file=wf)
            T = m.get_tensor_type
            Lc = lambda n: self.get_loc(n, m)
            ip = m.insert_point
            x = m.create_input_op(Lc("hidden"), 0)
            w1 = m.create_weight_op(w1p + ".weight", [hidden, dec_hidden])
            h = top.MatMulOp(T([1, dec_hidden]),
                             x,
                             w1,
                             m.none_op,
                             do_relu=False,
                             loc=Lc(w1p),
                             ip=ip).output
            h = top.ReluOp(T([1, dec_hidden]), h, loc=Lc("relu"), ip=ip).output
            h = top.MulOp(T([1, dec_hidden]), [h, h], loc=Lc("sq"), ip=ip).output
            w2 = m.create_weight_op(w2p + ".weight", [dec_hidden, out_dim])
            o = top.MatMulOp(T([1, out_dim]), h, w2, m.none_op, do_relu=False, loc=Lc(w2p),
                             ip=ip).output
            o = top.ReshapeOp(T([1, 2, out_dim // 2]),
                              o,
                              shape=[1, 2, out_dim // 2],
                              loc=Lc("rshp"),
                              ip=ip).output
            m.create_return_op([o])
            self.save_mlir_module(m, name)

        gen_bbox_head("coord_head", "coord_decoder.w1", "coord_decoder.w2")
        gen_bbox_head("size_head", "size_decoder.w1", "size_decoder.w2")

        # ---- seg_head : proj_segm SegmDecoder  relu^2 x2 + pixel_layer ----
        def gen_seg_head():
            name = "seg_head"
            m = MLIRImporter([[1, hidden]], [[1, segm_out]],
                             name,
                             self.platform, ["F32"],
                             weight_file=wf)
            T = m.get_tensor_type
            Lc = lambda n: self.get_loc(n, m)
            ip = m.insert_point
            x = m.create_input_op(Lc("hidden"), 0)
            for i in range(2):
                p = f"proj_segm.layers.{i}"
                w = m.create_weight_op(p + ".weight", [hidden, hidden])
                b = m.create_weight_op(p + ".bias", [1, hidden])
                x = top.MatMulOp(T([1, hidden]), x, w, m.none_op, do_relu=False, loc=Lc(p),
                                 ip=ip).output
                x = top.AddOp(T([1, hidden]), [x, b], loc=Lc(p + ".add"), ip=ip).output
                x = top.ReluOp(T([1, hidden]), x, loc=Lc(p + ".relu"), ip=ip).output
                x = top.MulOp(T([1, hidden]), [x, x], loc=Lc(p + ".sq"), ip=ip).output
            w = m.create_weight_op("proj_segm.pixel_layer.weight", [hidden, segm_out])
            o = top.MatMulOp(T([1, segm_out]),
                             x,
                             w,
                             m.none_op,
                             do_relu=False,
                             loc=Lc("pixel_layer"),
                             ip=ip).output
            m.create_return_op([o])
            self.save_mlir_module(m, name)

        gen_seg_head()

        # ---- mask_head : einsum("kdhw,kd->khw")  hr[1,256,H,W] x segm[K,256] ----
        def gen_mask_head():
            name = "mask_head"
            m = MLIRImporter([[1, segm_out, H, W], [Kmax, segm_out]], [[Kmax, H, W]],
                             name,
                             self.platform, ["F32", "F32"],
                             weight_file=wf)
            T = m.get_tensor_type
            Lc = lambda n: self.get_loc(n, m)
            ip = m.insert_point
            hr = m.create_input_op(Lc("hr_features"), 0)  # [1,256,H,W]
            segm = m.create_input_op(Lc("segm_tokens"), 1)  # [K,256]
            hr2 = top.ReshapeOp(T([segm_out, H * W]),
                                hr,
                                shape=[segm_out, -1],
                                loc=Lc("hr.rshp"),
                                ip=ip).output  # [256, HW]
            out = top.MatMulOp(T([Kmax, H * W]),
                               segm,
                               hr2,
                               m.none_op,
                               do_relu=False,
                               loc=Lc("einsum"),
                               ip=ip).output  # [K, HW]
            out = top.ReshapeOp(T([Kmax, H, W]), out, shape=[Kmax, H, W], loc=Lc("out.rshp"),
                                ip=ip).output
            m.create_return_op([out])
            self.save_mlir_module(m, name)

        gen_mask_head()

        # ---- coord_encoder / size_encoder : FourierEncoder ----
        # Input[1,2] -> MatMul(embed[2,feat/2]) -> MulConst(2pi) -> Cos/Sin
        #            -> Concat(axis=1)[1,feat] -> MatMul(transform[feat,hidden]) -> [1,hidden]
        # Used for re-injection: encode predicted (x,y)/(h,w) into a 1024-d embedding that
        # overwrites the coord/size token's embedding before block_cache_0.
        def gen_fourier_head(name, embed_p, transform_p, feat_dim):
            half = feat_dim // 2
            m = MLIRImporter([[1, 2]], [[1, hidden]], name, self.platform, ["F32"], weight_file=wf)
            T = m.get_tensor_type
            Lc = lambda n: self.get_loc(n, m)
            ip = m.insert_point
            x = m.create_input_op(Lc("coords"), 0)  # [1,2]
            we = m.create_weight_op(embed_p + ".weight", [2, half])
            f = top.MatMulOp(T([1, half]), x, we, m.none_op, do_relu=False, loc=Lc(embed_p),
                             ip=ip).output  # [1,256]
            f = top.MulConstOp(T([1, half]), f, const_val=2 * math.pi, loc=Lc("mul_2pi"),
                               ip=ip).output
            fc = top.CosOp(T([1, half]), f, loc=Lc("cos"), ip=ip).output
            fs = top.SinOp(T([1, half]), f, loc=Lc("sin"), ip=ip).output
            cat = top.ConcatOp(T([1, feat_dim]), [fc, fs], axis=1, loc=Lc("cat"),
                               ip=ip).output  # [1,512]
            wt = m.create_weight_op(transform_p + ".weight", [feat_dim, hidden])
            o = top.MatMulOp(T([1, hidden]),
                             cat,
                             wt,
                             m.none_op,
                             do_relu=False,
                             loc=Lc(transform_p),
                             ip=ip).output  # [1,1024]
            m.create_return_op([o])
            self.save_mlir_module(m, name)

        coord_enc = getattr(self.config, "coord_enc_dim", 512)
        size_enc = getattr(self.config, "size_enc_dim", 512)
        gen_fourier_head("coord_encoder", "coord_encoder.embed", "coord_encoder.transform",
                         coord_enc)
        gen_fourier_head("size_encoder", "size_encoder.embed", "size_encoder.transform", size_enc)

    def compile_heads(self):
        """Compile the head nets (coord/size/seg/mask + coord/size FourierEncoder)
        as f32 bmodels."""
        for name in ("coord_head", "size_head", "seg_head", "mask_head", "coord_encoder",
                     "size_encoder"):
            if self.register_bmodel(name):
                continue
            self.submit_deploy_task(
                name,
                [
                    f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}',
                    '--quant_input', '--quant_output'
                ],
            )

    # AnyUp upsampler bmodel (conv_segm + encoders + LFU + AnyUpRoPE + window
    # cross-attention).
    # ------------------------------------------------------------------ anyup helpers
    def _reflect_pad2d(self, m, in_op, name):
        """Reflect-pad a [B,C,H,W] tensor by 1 on each side via Slice+Concat.

        PyTorch reflect (no border repeat): left col = col 1, right col = col W-2,
        top row = row 1, bottom row = row H-2.
        """
        ip = m.insert_point
        B, C, H, W = list(in_op.type.shape)
        sl = lambda off, end, ax, shp, tag: top.SliceOp(m.get_tensor_type(shp),
                                                        in_op,
                                                        m.none_op,
                                                        m.none_op,
                                                        m.none_op,
                                                        offset=off,
                                                        steps=[1, 1, 1, 1],
                                                        ends=end,
                                                        axes=[],
                                                        loc=self.get_loc(name + tag, m),
                                                        ip=ip).output
        left = sl([0, 0, 0, 1], [B, C, H, 2], 3, [B, C, H, 1], ".l")
        right = sl([0, 0, 0, W - 2], [B, C, H, W - 1], 3, [B, C, H, 1], ".r")
        lr = top.ConcatOp(m.get_tensor_type([B, C, H, W + 2]), [left, in_op, right],
                          axis=3,
                          loc=self.get_loc(name + ".lr", m),
                          ip=ip).output
        top_ = top.SliceOp(m.get_tensor_type([B, C, 1, W + 2]),
                           lr,
                           m.none_op,
                           m.none_op,
                           m.none_op,
                           offset=[0, 0, 1, 0],
                           steps=[1, 1, 1, 1],
                           ends=[B, C, 2, W + 2],
                           axes=[],
                           loc=self.get_loc(name + ".t", m),
                           ip=ip).output
        bot = top.SliceOp(m.get_tensor_type([B, C, 1, W + 2]),
                          lr,
                          m.none_op,
                          m.none_op,
                          m.none_op,
                          offset=[0, 0, H - 2, 0],
                          steps=[1, 1, 1, 1],
                          ends=[B, C, H - 1, W + 2],
                          axes=[],
                          loc=self.get_loc(name + ".b", m),
                          ip=ip).output
        return top.ConcatOp(m.get_tensor_type([B, C, H + 2, W + 2]), [top_, lr, bot],
                            axis=2,
                            loc=self.get_loc(name + ".tb", m),
                            ip=ip).output

    def _conv2d(self, m, in_op, wname, wshape, k, pad, name, reflect=False, group=1):
        """Conv2d, zeros pad by default; reflect pad via _reflect_pad2d when asked."""
        ip = m.insert_point
        if reflect and pad > 0:
            in_op = self._reflect_pad2d(m, in_op, name + ".rpad")
            pad = 0
        B, Cin, H, W = list(in_op.type.shape)
        Cout = wshape[0]
        Ho = (H + 2 * pad - k) // 1 + 1
        Wo = (W + 2 * pad - k) // 1 + 1
        w = m.create_weight_op(wname, wshape)
        return top.ConvOp(m.get_tensor_type([B, Cout, Ho, Wo]),
                          in_op,
                          w,
                          m.none_op,
                          kernel_shape=[k, k],
                          strides=[1, 1],
                          pads=[pad, pad, pad, pad],
                          group=group,
                          loc=self.get_loc(name, m),
                          ip=ip).output

    def _groupnorm(self, m, in_op, prefix, name, num_groups=8, eps=1e-5):
        ip = m.insert_point
        shp = list(in_op.type.shape)
        C = shp[1]
        w = m.create_weight_op(prefix + ".weight", [1, C, 1, 1])
        b = m.create_weight_op(prefix + ".bias", [1, C, 1, 1])
        # NB: the fused GroupNormOp and a manual ReduceMean/Sub/Rsqrt implementation
        # produce byte-identical output here (var << eps on the small-magnitude LFU
        # input, so variance precision is moot — the GN is eps-dominated). The
        # kf resblock's cos 0.945 vs HF is therefore NOT a GroupNorm bug; it is the
        # LFU output's ~1e-3 residual (cos 0.9999) amplified ~316x by the GN's
        # 1/sqrt(eps) normalization on the tiny-DC (mean~1/128) input. The anyup
        # *output* is unaffected (cos 0.995, washed out by the aggregation conv).
        return top.GroupNormOp(m.get_tensor_type(shp),
                               in_op,
                               w,
                               b,
                               num_groups=num_groups,
                               eps=eps,
                               loc=self.get_loc(name, m),
                               ip=ip).output

    def _resblock(self, m, in_op, prefix, name):
        """ResBlock 128->128 (k=1): GN->SiLU->Conv->GN->SiLU->Conv + residual."""
        ip = m.insert_point
        shp = list(in_op.type.shape)
        x = self._groupnorm(m, in_op, prefix + ".block.0", name + ".gn0")
        x = top.SiLUOp(m.get_tensor_type(shp), x, loc=self.get_loc(name + ".silu0", m),
                       ip=ip).output
        x = self._conv2d(m, x, prefix + ".block.2.weight", [128, 128, 1, 1], 1, 0, name + ".c0")
        x = self._groupnorm(m, x, prefix + ".block.3", name + ".gn1")
        x = top.SiLUOp(m.get_tensor_type(shp), x, loc=self.get_loc(name + ".silu1", m),
                       ip=ip).output
        x = self._conv2d(m, x, prefix + ".block.5.weight", [128, 128, 1, 1], 1, 0, name + ".c1")
        return top.AddOp(m.get_tensor_type(shp), [x, in_op],
                         loc=self.get_loc(name + ".res", m),
                         ip=ip).output

    def _encoder(self, m, in_op, prefix, pre_kind, name):
        """pre (conv k=1 or LFU 256->128) + 2 ResBlocks."""
        ip = m.insert_point
        if pre_kind == "lfu":
            x = self._lfu(m, in_op, prefix + ".0", name + ".lfu")
        else:  # conv k=1
            x = self._conv2d(m, in_op, prefix + ".0.weight", pre_kind, 1, 0, name + ".pre")
        x = self._resblock(m, x, prefix + ".1", name + ".rb0")
        x = self._resblock(m, x, prefix + ".2", name + ".rb1")
        return x

    def _lfu(self, m, in_op, prefix, name):
        """LearnedFeatureUnification: depthwise conv (basis.repeat(c)) / denom,
        softmax over out_channels, mean over in_channels -> [1,128,h,w]."""
        ip = m.insert_point
        B, C, h, w = list(in_op.type.shape)  # C=256
        x = self._conv2d(m,
                         in_op,
                         "lfu_basis.weight", [128 * C, 1, 5, 5],
                         5,
                         2,
                         name + ".dw",
                         group=C)  # [1, 128*C, h, w]
        x = top.ReshapeOp(m.get_tensor_type([1, 128, C, h, w]),
                          x,
                          shape=[1, 128, C, h, w],
                          loc=self.get_loc(name + ".rshp", m),
                          ip=ip).output
        denom_inv = m.create_weight_op("lfu_denom_inv", [1, 1, h, w])
        x = top.MulOp(m.get_tensor_type([1, 128, C, h, w]), [x, denom_inv],
                      loc=self.get_loc(name + ".div", m),
                      ip=ip).output
        x = top.SoftmaxOp(m.get_tensor_type([1, 128, C, h, w]),
                          x,
                          axis=1,
                          loc=self.get_loc(name + ".sm", m),
                          ip=ip).output
        s = top.ReduceOp(m.get_tensor_type([1, 128, h, w]),
                         x,
                         axes=[2],
                         keepdims=False,
                         mode=StringAttr.get("ReduceSum"),
                         loc=self.get_loc(name + ".sum", m),
                         ip=ip).output
        return top.MulConstOp(m.get_tensor_type([1, 128, h, w]),
                              s,
                              const_val=1.0 / C,
                              loc=self.get_loc(name + ".mean", m),
                              ip=ip).output

    # ------------------------------------------------------------------ anyup gen
    def gen_anyup_mlir(self):
        from tqdm import tqdm
        tqdm.write("generate falcon anyup mlir ...")
        H, W = self.max_shape[0], self.max_shape[1]
        h, w = H // 16, W // 16
        HW, hw = H * W, h * w
        qk_dim, n_heads = 128, 4
        qk_hd = qk_dim // n_heads  # 32
        v_dim = self.segm_out_dim  # 256
        v_hd = v_dim // n_heads  # 64
        scale = qk_hd**-0.5

        # ---- preprocess + save weights ----
        weight_file = "anyup_weights.npz"
        wd = {}
        read = lambda k: np.ascontiguousarray(self.model.read(k).astype(np.float32))
        # conv weights: keep [Cout,Cin,kH,kW]
        for k in [
                "conv_segm.weight", "itok_upsampler.cross_decode.conv2d.weight",
                "itok_upsampler.aggregation.0.weight", "itok_upsampler.image_encoder.0.weight",
                "itok_upsampler.key_encoder.0.weight", "itok_upsampler.query_encoder.0.weight"
        ]:
            wd[k] = read(k)
        wd["conv_segm.bias"] = read("conv_segm.bias")
        # ResBlock + GroupNorm weights for every encoder/aggregation sub-module
        for enc in [
                "image_encoder.1", "image_encoder.2", "key_encoder.1", "key_encoder.2",
                "query_encoder.1", "query_encoder.2", "key_features_encoder.1",
                "key_features_encoder.2", "aggregation.1", "aggregation.2"
        ]:
            pre = "itok_upsampler." + enc
            for s in [".block.0", ".block.3"]:
                wd[pre + s + ".weight"] = read(pre + s + ".weight")
                wd[pre + s + ".bias"] = read(pre + s + ".bias")
            for s in [".block.2", ".block.5"]:
                wd[pre + s + ".weight"] = read(pre + s + ".weight")
        # LFU basis -> tile(in_ch=256) for depthwise grouped conv (each of the
        # 256 groups must use the FULL basis[0:128]; np.tile repeats the whole
        # tensor, np.repeat would repeat each element — wrong group ordering).
        basis = read("itok_upsampler.key_features_encoder.0.basis")  # [128,1,5,5]
        wd["lfu_basis.weight"] = np.ascontiguousarray(np.tile(basis,
                                                              (256, 1, 1, 1)))  # [128*256,1,5,5]
        denom = torch.nn.functional.conv2d(torch.ones(1, 1, h, w),
                                           torch.ones(1, 1, 5, 5),
                                           padding=2).numpy()
        wd["lfu_denom_inv"] = (1.0 / denom).astype(np.float32)  # [1,1,h,w]
        # cross-attn in_proj: split q/k (v unused), transpose for x@W
        ipw = read("itok_upsampler.cross_decode.cross_attn.attention.in_proj_weight")  # [384,128]
        ipb = read("itok_upsampler.cross_decode.cross_attn.attention.in_proj_bias")  # [384]
        wd["in_proj_q.weight"] = np.ascontiguousarray(ipw[0:128].T.copy())
        wd["in_proj_k.weight"] = np.ascontiguousarray(ipw[128:256].T.copy())
        wd["in_proj_q.bias"] = ipb[0:128].reshape(1, 128)
        wd["in_proj_k.bias"] = ipb[128:256].reshape(1, 128)
        # QK RMSNorm (weighted)
        for s in ["norm_q", "norm_k"]:
            wd["itok_upsampler.cross_decode.cross_attn." + s +
               ".weight"] = read("itok_upsampler.cross_decode.cross_attn." + s + ".weight")
        # AnyUpRoPE freqs + precomputed coords
        wd["anyup_rope.freqs"] = read("itok_upsampler.rope.freqs")  # [2,128]
        xs = np.linspace(0, 1, H, dtype=np.float32)
        ys = np.linspace(0, 1, W, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        wd["anyup_coords"] = np.stack((xx, yy), -1).reshape(1, HW, 2)
        # image normalize: (x*0.5+0.5 - mean)/std  -> x*scale + bias
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        wd["img_norm_scale"] = (0.5 / std).reshape(1, 3, 1, 1)
        wd["img_norm_bias"] = ((0.5 - mean) / std).reshape(1, 3, 1, 1)
        # AnyUp cross-attn window mask. Geometry-only (depends on H,W,h,w,ratio,
        # not the input image), so for static 256x256 it is a compile-time
        # constant — bake it into the bmodel as a weight instead of feeding it
        # as a runtime input. Mirrors HF anyup.compute_attention_mask (0=keep,
        # -1e9=masked); -1e9 (not -inf) avoids NaN in the bmodel.
        ratio = 0.1
        r_pos = (np.arange(H, dtype=np.float32) + 0.5) / H
        c_pos = (np.arange(W, dtype=np.float32) + 0.5) / W
        pos_r, pos_c = np.meshgrid(r_pos, c_pos, indexing="ij")
        r0 = np.floor(np.clip(pos_r - ratio, 0, 1) * h).astype(np.int64)
        r1 = np.ceil(np.clip(pos_r + ratio, 0, 1) * h).astype(np.int64)
        c0 = np.floor(np.clip(pos_c - ratio, 0, 1) * w).astype(np.int64)
        c1 = np.ceil(np.clip(pos_c + ratio, 0, 1) * w).astype(np.int64)
        rows = np.arange(h)
        cols = np.arange(w)
        row_ok = (rows >= r0.reshape(-1, 1)) & (rows < r1.reshape(-1, 1))  # [HW,h]
        col_ok = (cols >= c0.reshape(-1, 1)) & (cols < c1.reshape(-1, 1))  # [HW,w]
        keep = (row_ok[:, :, None] & col_ok[:, None, :]).reshape(HW, hw)
        wd["anyup_window_mask"] = np.where(keep, 0.0, -1e9).astype(np.float32).reshape(1, 1, HW, hw)
        self.weight_keys.extend(list(wd.keys()))
        np.savez(weight_file, **wd)

        wf = f"../{weight_file}"
        name = "anyup"
        m = MLIRImporter([[1, 3, H, W], [1, self.hidden_size, h, w]], [[1, v_dim, H, W]],
                         name,
                         self.platform, ["F32", "F32"],
                         weight_file=wf)
        T = m.get_tensor_type
        Lc = lambda n: self.get_loc(n, m)
        ip = m.insert_point

        images = m.create_input_op(Lc("images"), 0)  # [1,3,H,W]
        lr_tok = m.create_input_op(Lc("lr_tokens"), 1)  # [1,1024,h,w]
        mask = m.create_weight_op("anyup_window_mask", [1, 1, HW, hw])  # baked const

        # image normalize
        ns = m.create_weight_op("img_norm_scale", [1, 3, 1, 1])
        nb = m.create_weight_op("img_norm_bias", [1, 3, 1, 1])
        img = top.MulOp(T([1, 3, H, W]), [images, ns], loc=Lc("img.mul"), ip=ip).output
        img = top.AddOp(T([1, 3, H, W]), [img, nb], loc=Lc("img.add"), ip=ip).output

        # enc = image_encoder(img)  [1,128,H,W]
        enc = self._encoder(m, img, "itok_upsampler.image_encoder", [128, 3, 1, 1], "img_enc")

        # AnyUpRoPE on enc
        enc_flat = top.PermuteOp(T([1, H, W, 128]),
                                 enc,
                                 order=[0, 2, 3, 1],
                                 loc=Lc("enc.perm"),
                                 ip=ip).output
        enc_flat = top.ReshapeOp(T([1, HW, 128]),
                                 enc_flat,
                                 shape=[1, HW, 128],
                                 loc=Lc("enc.flat"),
                                 ip=ip).output
        coords = m.create_weight_op("anyup_coords", [1, HW, 2])
        freqs = m.create_weight_op("anyup_rope.freqs", [2, 128])
        angle = top.MatMulOp(T([1, HW, 128]),
                             coords,
                             freqs,
                             m.none_op,
                             do_relu=False,
                             loc=Lc("rope.angle"),
                             ip=ip).output
        cos = top.CosOp(T([1, HW, 128]), angle, loc=Lc("rope.cos"), ip=ip).output
        sin = top.SinOp(T([1, HW, 128]), angle, loc=Lc("rope.sin"), ip=ip).output
        enc_flat = top.RopeOp(T([1, HW, 128]),
                              enc_flat,
                              sin,
                              cos,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=Lc("rope"),
                              ip=ip).output
        enc = top.ReshapeOp(T([1, H, W, 128]),
                            enc_flat,
                            shape=[1, H, W, 128],
                            loc=Lc("enc.rshp"),
                            ip=ip).output
        enc = top.PermuteOp(T([1, 128, H, W]), enc, order=[0, 3, 1, 2], loc=Lc("enc.perm2"),
                            ip=ip).output

        # features = conv_segm(lr_tok)  [1,256,h,w]
        feats = self._conv2d(m, lr_tok, "conv_segm.weight", [256, 1024, 3, 3], 3, 1, "conv_segm")
        cb = m.create_weight_op("conv_segm.bias", [1, 256, 1, 1])
        feats = top.AddOp(T([1, 256, h, w]), [feats, cb], loc=Lc("conv_segm.add"), ip=ip).output

        # q = query_encoder(enc) (pool to H,W = no-op);  k_enc = pool(key_encoder(enc), (h,w))
        q = self._encoder(m, enc, "itok_upsampler.query_encoder", [128, 128, 1, 1], "q_enc")
        k_enc = self._encoder(m, enc, "itok_upsampler.key_encoder", [128, 128, 1, 1], "k_enc")
        k_enc = top.AvgPoolOp(T([1, 128, h, w]),
                              k_enc,
                              kernel_shape=[H // h, W // w],
                              strides=[H // h, W // w],
                              pads=[0, 0, 0, 0],
                              loc=Lc("k_enc.pool"),
                              ip=ip).output

        # kf = key_features_encoder(l2norm(feats))  [1,128,h,w]
        feats_n = self.l2norm(m, feats, "feats_l2")
        kf_lfu = self._lfu(m, feats_n, "itok_upsampler.key_features_encoder.0", "kf_enc.lfu")
        kf = self._resblock(m, kf_lfu, "itok_upsampler.key_features_encoder.1", "kf_enc.rb0")
        kf = self._resblock(m, kf, "itok_upsampler.key_features_encoder.2", "kf_enc.rb1")
        k = top.ConcatOp(T([1, 256, h, w]), [k_enc, kf], axis=1, loc=Lc("k.cat"), ip=ip).output
        # aggregation: pre conv k=3 reflect + 2 ResBlocks  -> [1,128,h,w]
        k_pre = self._conv2d(m,
                             k,
                             "itok_upsampler.aggregation.0.weight", [128, 256, 3, 3],
                             3,
                             1,
                             "agg.pre",
                             reflect=True)
        k = k_pre
        k = self._resblock(m, k, "itok_upsampler.aggregation.1", "agg.rb0")
        k = self._resblock(m, k, "itok_upsampler.aggregation.2", "agg.rb1")
        v = feats  # [1,256,h,w]

        # cross_decode: q2=conv2d(q); seq reshape; RMSNorm q/k; in_proj q/k
        q2 = self._conv2d(m, q, "itok_upsampler.cross_decode.conv2d.weight", [128, 128, 3, 3], 3, 1,
                          "cross.conv")
        # NCHW -> NHWC -> (HW, C) to match HF rearrange "b c h w -> b (h w) c"
        q_seq = top.PermuteOp(T([1, H, W, 128]), q2, order=[0, 2, 3, 1], loc=Lc("q.perm"),
                              ip=ip).output
        q_seq = top.ReshapeOp(T([1, HW, 128]), q_seq, shape=[1, HW, 128], loc=Lc("q.seq"),
                              ip=ip).output
        k_seq = top.PermuteOp(T([1, h, w, 128]), k, order=[0, 2, 3, 1], loc=Lc("k.perm"),
                              ip=ip).output
        k_seq = top.ReshapeOp(T([1, hw, 128]), k_seq, shape=[1, hw, 128], loc=Lc("k.seq"),
                              ip=ip).output
        v_seq = top.PermuteOp(T([1, h, w, 256]), v, order=[0, 2, 3, 1], loc=Lc("v.perm"),
                              ip=ip).output
        v_seq = top.ReshapeOp(T([1, hw, 256]), v_seq, shape=[1, hw, 256], loc=Lc("v.seq"),
                              ip=ip).output
        # RMSNorm (weighted) on last dim
        nq_w = m.create_weight_op("itok_upsampler.cross_decode.cross_attn.norm_q.weight",
                                  [1, 1, 128])
        nk_w = m.create_weight_op("itok_upsampler.cross_decode.cross_attn.norm_k.weight",
                                  [1, 1, 128])
        xq = top.RMSNormOp(T([1, HW, 128]), q_seq, nq_w, eps=1e-5, loc=Lc("norm_q"), ip=ip).output
        xk = top.RMSNormOp(T([1, hw, 128]), k_seq, nk_w, eps=1e-5, loc=Lc("norm_k"), ip=ip).output
        # in_proj q/k (v unchanged)
        wq = m.create_weight_op("in_proj_q.weight", [128, 128])
        wk = m.create_weight_op("in_proj_k.weight", [128, 128])
        bq = m.create_weight_op("in_proj_q.bias", [1, 128])
        bk = m.create_weight_op("in_proj_k.bias", [1, 128])
        xq = top.MatMulOp(T([1, HW, 128]),
                          xq,
                          wq,
                          m.none_op,
                          do_relu=False,
                          loc=Lc("in_proj.q"),
                          ip=ip).output
        xq = top.AddOp(T([1, HW, 128]), [xq, bq], loc=Lc("in_proj.q.b"), ip=ip).output
        xk = top.MatMulOp(T([1, hw, 128]),
                          xk,
                          wk,
                          m.none_op,
                          do_relu=False,
                          loc=Lc("in_proj.k"),
                          ip=ip).output
        xk = top.AddOp(T([1, hw, 128]), [xk, bk], loc=Lc("in_proj.k.b"), ip=ip).output
        # reshape to heads: q [1,HW,4,32], k [1,hw,4,32], v [1,hw,4,64]
        xq = top.ReshapeOp(T([1, HW, n_heads, qk_hd]),
                           xq,
                           shape=[1, HW, n_heads, qk_hd],
                           loc=Lc("q.heads"),
                           ip=ip).output
        xk = top.ReshapeOp(T([1, hw, n_heads, qk_hd]),
                           xk,
                           shape=[1, hw, n_heads, qk_hd],
                           loc=Lc("k.heads"),
                           ip=ip).output
        xv = top.ReshapeOp(T([1, hw, n_heads, v_hd]),
                           v_seq,
                           shape=[1, hw, n_heads, v_hd],
                           loc=Lc("v.heads"),
                           ip=ip).output
        # Hand-written cross-attention (NOT FAttentionOp): the fused FAttentionOp
        # kernel NaNs for tall non-square attention (mq=HW=65536 >> mk=hw=256,
        # dim=32) — half the query tiles go NaN regardless of mask/chunking. The
        # backbone uses this same manual MatMul+softmax pattern (see _attention).
        # Non-square head dim (qk=32, v=64) handled in one call via hdim_is_batch.
        qb = self._to_bhd(m, xq, "fa.q")  # [n_heads, HW, qk_hd]
        kb = self._to_bhd(m, xk, "fa.k")  # [n_heads, hw, qk_hd]
        vb = self._to_bhd(m, xv, "fa.v")  # [n_heads, hw, v_hd]
        scores = top.MatMulOp(T([n_heads, HW, hw]),
                              qb,
                              kb,
                              m.none_op,
                              right_transpose=True,
                              hdim_is_batch=True,
                              do_relu=False,
                              loc=Lc("fa.qk"),
                              ip=ip).output
        scores = top.MulConstOp(T([n_heads, HW, hw]),
                                scores,
                                const_val=scale,
                                loc=Lc("fa.scale"),
                                ip=ip).output
        mask2 = top.ReshapeOp(T([1, HW, hw]),
                              mask,
                              shape=[1, HW, hw],
                              loc=Lc("fa.mask.rshp"),
                              ip=ip).output
        scores = top.AddOp(T([n_heads, HW, hw]), [scores, mask2], loc=Lc("fa.mask.add"),
                           ip=ip).output
        w = top.SoftmaxOp(T([n_heads, HW, hw]), scores, axis=2, loc=Lc("fa.softmax"), ip=ip).output
        ob = top.MatMulOp(T([n_heads, HW, v_hd]),
                          w,
                          vb,
                          m.none_op,
                          hdim_is_batch=True,
                          do_relu=False,
                          loc=Lc("fa.wv"),
                          ip=ip).output  # [n_heads, HW, v_hd]
        oa = self._from_bhd(m, ob, 1, HW, n_heads, v_hd, "fa.out")  # [1,HW,n_heads,v_hd]
        out = top.ReshapeOp(T([1, HW, v_dim]), oa, shape=[1, HW, v_dim], loc=Lc("fa.flat"),
                            ip=ip).output
        # (HW, C) -> NHWC -> NCHW to match HF rearrange "b (h w) c -> b c h w"
        out = top.ReshapeOp(T([1, H, W, v_dim]),
                            out,
                            shape=[1, H, W, v_dim],
                            loc=Lc("out.nhwc"),
                            ip=ip).output
        out = top.PermuteOp(T([1, v_dim, H, W]), out, order=[0, 3, 1, 2], loc=Lc("out.rshp"),
                            ip=ip).output
        ret = [out]
        m.create_return_op(ret)
        self.save_mlir_module(m, name)

    def compile_anyup(self):
        name = "anyup"
        if self.register_bmodel(name):
            return
        self.submit_deploy_task(
            name,
            [
                f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}',
                '--quant_input', '--quant_output'
            ],
        )

    def gen_vit_mlir(self):
        pass
