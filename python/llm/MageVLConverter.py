# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class MageVLConverter(LlmConverter):
    """Converter for Microsoft Mage-VL (codec-native streaming VLM).

    Phase 1 scope: offline (non-streaming) VLM — ViT + LLM. Gate (StreamMind
    Mamba) and codec (DCVC-RT) are NOT deployed on TPU in phase 1.

    Architecture:
      - LM backbone: plain Qwen3-4B (1D RoPE, QK-norm, GQA, SiLU). Fully inherited
        from LlmConverter with llm_type=LlmType.QWEN3 + QWEN3VL_INFO (weights under
        ``model.language_model.*``). Zero custom text code.
      - Vision tower: Mage-ViT (custom). New gen_vit_mlir:
          Conv patch_embed (k=s=16, bias=False) -> layernorm_pre ->
          24x vision_block (fused qkv split / MHA / 3D RoPE 4:6:6 / GELU erf) ->
          merger (ln_q + 2x2 reshape + fc1 GELU fc2).

    The vision 3D RoPE uses an *interleaved* rotate_half
    ``(x0,x1,x2,x3) -> (-x1, x0, -x3, x2)`` whose adjacent pairs do NOT share a
    frequency, so it cannot use top.RopeOp. rotate_half is realized as a fixed
    64x64 permutation-matrix MatMul (pure reorder+negate, position-independent),
    avoiding the ConcatOp(axis>=2) SupportInplace segfault seen in the earlier
    port. See Mage-VL_plan.md §7/#1 and §10 (historical pitfalls).

    ViT position inputs: the three RoPE axes (t/h/w) are fed as three separate
    [N] INT32 inputs rather than one [N,3] tensor, so each axis gathers its own
    cos/sin table ([num_patches, axis_size]) directly — no SliceOp/SqueezeOp
    (sidesteps historical pitfall #1/#4 too).
    """

    def __init__(self, args, config, loader=None):
        super().__init__(args, config, loader=loader)
        self.max_pixels = args.max_pixels
        self.max_shape = args.max_shape
        if self.max_shape[0] % 32 != 0 or self.max_shape[1] % 32 != 0:
            raise RuntimeError(f"max_shape dims must be multiples of 32: {self.max_shape}")

        # vision config. do_vit=True so the base class auto-registers
        # gen_vit_mlir / compile_vit (once each) into the gen/compile task lists;
        # we override both methods below. (do_vit=False + manual re-append would
        # register vit twice in the combine list.)
        self.do_vit = True
        self.vit_path = "model.visual"
        self.init_vconfig()

        # gate config (StreamMind Gate for streaming)
        self.do_gate = True
        self.gate_T = 4  # frames per segment (static compile)
        self.gate_d_model = 2560
        self.gate_d_state = 16
        self.gate_d_conv = 4
        self.gate_expand = 2
        self.gate_d_inner = self.gate_d_model * self.gate_expand  # 5120
        self.gate_dt_rank = 160  # ceil(2560/16)
        self.gate_leaky_slope = 0.01
        self.gate_num_patches = 98  # after 2x2 merge: 392/4

        # ClsNet config (4-layer micro Qwen3 binary classifier)
        self.cls_num_layers = 4
        self.cls_hidden_size = 2560
        self.cls_intermediate_size = 12288
        self.cls_vocab_size = 2  # silent/speak

    def init_vconfig(self):
        vc = self.config.vision_config
        self.vconfig = vc
        self.patch_size = vc.patch_size  # 16
        self.temporal_patch_size = getattr(vc, "temporal_patch_size", 1)
        self.spatial_merge_size = vc.spatial_merge_size  # 2
        self.in_channels = vc.num_channels  # 3
        self.vit_depth = vc.num_hidden_layers  # 24
        self.embed_dim = vc.hidden_size  # 1024
        self.vnum_heads = vc.num_attention_heads  # 16
        self.vhead_dim = self.embed_dim // self.vnum_heads  # 64
        self.vintermediate_size = vc.intermediate_size  # 4096
        self.vout_hidden_size = vc.out_hidden_size  # 2560
        self.vnorm_eps = vc.layer_norm_eps  # 1e-6
        self.vhidden_act = vc.hidden_act  # "gelu"
        self.vrope_theta = vc.rope_theta  # 10000.0
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * \
            self.temporal_patch_size  # 3*16*16*1 = 768
        # 3D RoPE 4:6:6 split (head_dim=64, half=32, unit=32//16=2)
        half = self.vhead_dim // 2  # 32
        unit = half // 16  # 2
        self.rope_t_size = 4 * unit  # 8
        self.rope_h_size = 6 * unit  # 12
        self.rope_w_size = 6 * unit  # 12

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        # text backbone is plain Qwen3-4B: 1D RoPE (base default) + QK-norm
        # (base gen_block_mlir activates q_norm/k_norm for LlmType.QWEN3).
        self.llm_type = LlmType.QWEN3
        # weights live under model.language_model.* (+lm_head) -> reuse QWEN3VL_INFO
        self.model_info = QWEN3VL_INFO

    # ------------------------------------------------------------------
    # Vision 3D RoPE helpers
    # ------------------------------------------------------------------

    def _build_3d_rope_tables(self):
        """Precompute per-axis cos/sin lookup tables (baked as bmodel weights).

        For each axis a (t/h/w) with size ``sa`` and up to ``max_pos`` positions,
        the table is ``[max_pos, sa]`` where row ``p`` is ``cos(p * inv_freq_a)``.
        At runtime the pipeline gathers row ``pos_<a>[N]`` for every patch,
        concatenates the three axes -> [N, half] -> tile -> [N, 64].
        """
        max_pos = self.num_patches  # upper bound on any axis coordinate
        base = self.vrope_theta
        tables = {}
        for axis, sa in (('t', self.rope_t_size), ('h', self.rope_h_size), ('w', self.rope_w_size)):
            inv = 1.0 / (base**(np.arange(sa, dtype=np.float32) / sa))  # [sa]
            pos = np.arange(max_pos, dtype=np.float32)  # [max_pos]
            freqs = np.outer(pos, inv)  # [max_pos, sa]
            tables[axis] = (np.ascontiguousarray(np.cos(freqs).astype(np.float32)),
                            np.ascontiguousarray(np.sin(freqs).astype(np.float32)))
        return tables

    @staticmethod
    def _rotate_half_perm_matrix(dim: int):
        """Build the (dim x dim) matrix M stored as the right operand of
        ``MatMulOp(x, M)`` so that ``x @ M == rotate_half(x)`` for the interleaved
        layout ``(x0,x1,x2,x3) -> (-x1, x0, -x3, x2)``:

            out[2k]   = -x[2k+1]
            out[2k+1] =  x[2k]

        Note: the apply site does ``flat @ M`` (left-multiply by the activation),
        so M must be the *transpose* of the naive per-pair rotation matrix
        (``P`` with ``P @ x == rotate_half(x)`` would give ``x @ P != rotate_half(x)``).
        Swapping the two non-zero entries yields ``x @ M == P @ x == rotate_half(x)``.
        """
        P = np.zeros((dim, dim), dtype=np.float32)
        for k in range(dim // 2):
            P[2 * k, 2 * k + 1] = 1.0
            P[2 * k + 1, 2 * k] = -1.0
        return P

    def _apply_3d_rope(self, vit_mlir, in_op, cos_op, sin_op, perm_w_op, name_prefix, ip):
        """Apply interleaved 3D RoPE to q or k.

        in_op    : [1, N, vnum_heads, vhead_dim]  (vhead_dim=64)
        cos_op   : [1, N, 1, 64]   sin_op: [1, N, 1, 64]  (broadcast over heads)
        perm_w_op: shared 64x64 rotate_half permutation weight op
        returns  : [1, N, vnum_heads, vhead_dim]
        """
        T = vit_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, vit_mlir)
        N = self.num_patches
        H = self.vnum_heads
        D = self.vhead_dim
        # flatten [1,N,H,D] -> [N*H, D]
        flat = top.ReshapeOp(T([N * H, D]),
                             in_op,
                             shape=[N * H, D],
                             loc=L(name_prefix + ".rope.reshape"),
                             ip=ip).output
        # rotate_half(x) = x @ P
        rot = top.MatMulOp(T([N * H, D]),
                           flat,
                           perm_w_op,
                           vit_mlir.none_op,
                           loc=L(name_prefix + ".rope.rotate_half"),
                           ip=ip).output
        rot = top.ReshapeOp(T([1, N, H, D]),
                            rot,
                            shape=[1, N, H, D],
                            loc=L(name_prefix + ".rope.rot_back"),
                            ip=ip).output
        cos_part = top.MulOp(T([1, N, H, D]), [in_op, cos_op],
                             loc=L(name_prefix + ".rope.mul_cos"),
                             ip=ip).output
        sin_part = top.MulOp(T([1, N, H, D]), [rot, sin_op],
                             loc=L(name_prefix + ".rope.mul_sin"),
                             ip=ip).output
        out = top.AddOp(T([1, N, H, D]), [cos_part, sin_part],
                        loc=L(name_prefix + ".rope.out"),
                        ip=ip).output
        return out

    # ------------------------------------------------------------------
    # Vision weight setup
    # ------------------------------------------------------------------

    def _set_vit_qkv_weight(self, block_prefix: str, weight_dict: dict):
        """Split the fused qkv weight [3*E, E] + bias [3*E] into q/k/v.

        Mage-ViT vision attention keeps qkv in BF16 (in the AWQ ignore list),
        so this is always the plain-weight path.
        """
        E = self.embed_dim
        qkv_w_path = block_prefix + ".self_attn.qkv.weight"
        if not self.model.is_exist(qkv_w_path):
            raise RuntimeError(f"Can't find key: {qkv_w_path}")
        w = self.model.read(qkv_w_path).reshape(3 * E, E)
        q_w = np.ascontiguousarray(np.transpose(w[:E, :], (1, 0)))
        k_w = np.ascontiguousarray(np.transpose(w[E:2 * E, :], (1, 0)))
        v_w = np.ascontiguousarray(np.transpose(w[2 * E:, :], (1, 0)))
        weight_dict[block_prefix + ".self_attn.q.weight"] = q_w
        weight_dict[block_prefix + ".self_attn.k.weight"] = k_w
        weight_dict[block_prefix + ".self_attn.v.weight"] = v_w
        bias_path = block_prefix + ".self_attn.qkv.bias"
        if self.model.is_exist(bias_path):
            b = self.model.read(bias_path).reshape(3 * E)
            weight_dict[block_prefix + ".self_attn.q.bias"] = b[:E]
            weight_dict[block_prefix + ".self_attn.k.bias"] = b[E:2 * E]
            weight_dict[block_prefix + ".self_attn.v.bias"] = b[2 * E:]

    # ------------------------------------------------------------------
    # gen_vit_mlir
    # ------------------------------------------------------------------

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        name = "vit"
        vit_npz = "vit_top_weights.npz"
        P = self.vit_path
        patch_embed = f"{P}.embeddings.patch_embedding"
        ln_pre = f"{P}.layernorm_pre"
        layers = f"{P}.encoder.layers"
        merger_ln_q = f"{P}.merger.ln_q"
        merger_fc1 = f"{P}.merger.mlp.0"
        merger_fc2 = f"{P}.merger.mlp.2"
        rotary_cos = f"{P}.rotary.cos"
        rotary_sin = f"{P}.rotary.sin"
        # shared rotate_half permutation matrix
        perm_name = f"{P}.rope_perm"

        def save_weights():
            weights_dict = {}
            # 3D RoPE axis cos/sin tables
            tables = self._build_3d_rope_tables()
            weights_dict[f"{rotary_cos}.t"] = tables['t'][0]
            weights_dict[f"{rotary_cos}.h"] = tables['h'][0]
            weights_dict[f"{rotary_cos}.w"] = tables['w'][0]
            weights_dict[f"{rotary_sin}.t"] = tables['t'][1]
            weights_dict[f"{rotary_sin}.h"] = tables['h'][1]
            weights_dict[f"{rotary_sin}.w"] = tables['w'][1]
            # rotate_half permutation matrix
            weights_dict[perm_name] = self._rotate_half_perm_matrix(self.vhead_dim)
            # patch embedding conv weight [E, C, ph, pw] (no bias)
            conv_w = self.model.read(patch_embed + ".weight")  # [E,3,16,16]
            weights_dict[patch_embed + ".weight"] = np.ascontiguousarray(conv_w)
            # pre-norm / merger
            self.set_common_weight(ln_pre, weights_dict)
            self.set_common_weight(merger_ln_q, weights_dict)
            self.set_linear_weight(merger_fc1, weights_dict)
            self.set_linear_weight(merger_fc2, weights_dict)
            # blocks
            for i in range(self.vit_depth):
                bp = f"{layers}.{i}"
                self.set_common_weight(bp + ".layer_norm1", weights_dict)
                self.set_common_weight(bp + ".layer_norm2", weights_dict)
                self.set_linear_weight(bp + ".self_attn.proj", weights_dict)
                self.set_linear_weight(bp + ".mlp.fc1", weights_dict)
                self.set_linear_weight(bp + ".mlp.fc2", weights_dict)
                self._set_vit_qkv_weight(bp, weights_dict)
            self.weight_keys.extend(list(weights_dict.keys()))
            np.savez(vit_npz, **weights_dict)

        # bake weights BEFORE building the graph (shape-infer safe; pitfall #3)
        save_weights()

        N = self.num_patches
        E = self.embed_dim
        H = self.vnum_heads
        D = self.vhead_dim
        half = D // 2  # 32
        in_shape = [N, self.patch_dim]  # [N, 768]
        pos_shape = [N]  # [N] per axis
        mask_shape = [1, 1, N, N]
        out_dim = N // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.vout_hidden_size]  # [N/4, 2560]

        # inputs: pixel_values, pos_t, pos_h, pos_w, attention_mask
        input_shapes = [in_shape, pos_shape, pos_shape, pos_shape, mask_shape]
        input_types = ['F32', 'INT32', 'INT32', 'INT32', 'F32']
        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                name,
                                self.platform,
                                input_types,
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, vit_mlir)

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)  # pixel_values
        pos_t_op = vit_mlir.create_input_op(L('position_ids'), 1)  # pos_t [N]
        pos_h_op = vit_mlir.create_input_op(L('position_ids'), 2)  # pos_h [N]
        pos_w_op = vit_mlir.create_input_op(L('position_ids'), 3)  # pos_w [N]
        mask_op = vit_mlir.create_input_op(L('attention_mask'), 4)  # [1,1,N,N]

        # ---- patch embedding: [N,768] -> [N,3,16,16] -> Conv -> [N,E,1,1] -> [1,N,E]
        x = top.ReshapeOp(T([N, self.in_channels, self.patch_size, self.patch_size]),
                          in0_op,
                          shape=[N, self.in_channels, self.patch_size, self.patch_size],
                          loc=L(patch_embed + ".reshape"),
                          ip=ip).output
        conv_w = vit_mlir.create_weight_op(patch_embed + ".weight",
                                           [E, self.in_channels, self.patch_size, self.patch_size])
        x = top.ConvOp(
            T([N, E, 1, 1]),
            x,
            conv_w,
            vit_mlir.none_op,  # bias=False
            kernel_shape=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            loc=L(patch_embed),
            ip=ip).output
        x = top.ReshapeOp(T([1, N, E]),
                          x,
                          shape=[1, N, E],
                          loc=L(patch_embed + ".unsqueeze"),
                          ip=ip).output
        x = self.layer_norm(vit_mlir, x, ln_pre, eps=self.vnorm_eps)

        # ---- 3D RoPE cos/sin: gather three axis tables -> concat [N,32] -> tile [N,64]
        def build_cossin(which, pos_op, axis_key, sa):
            tbl = vit_mlir.create_weight_op(f"{which}.{axis_key}", [self.num_patches, sa])
            g = top.GatherOp(T([N, sa]),
                             tbl,
                             pos_op,
                             axis=0,
                             loc=L(f"{which}.{axis_key}.gather"),
                             ip=ip).output
            return g

        cos_t = build_cossin(rotary_cos, pos_t_op, 't', self.rope_t_size)
        cos_h = build_cossin(rotary_cos, pos_h_op, 'h', self.rope_h_size)
        cos_w = build_cossin(rotary_cos, pos_w_op, 'w', self.rope_w_size)
        cos_hw = top.ConcatOp(T([N, self.rope_h_size + self.rope_w_size]), [cos_h, cos_w],
                              axis=1,
                              loc=L(rotary_cos + ".concat.hw"),
                              ip=ip).output
        cos_full = top.ConcatOp(T([N, half]), [cos_t, cos_hw],
                                axis=1,
                                loc=L(rotary_cos + ".concat.thw"),
                                ip=ip).output
        cos_op = top.TileOp(T([N, D]), cos_full, tile=[1, 2], loc=L(rotary_cos + ".tile"),
                            ip=ip).output
        cos_op = top.ReshapeOp(T([1, N, 1, D]),
                               cos_op,
                               shape=[1, N, 1, D],
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output

        sin_t = build_cossin(rotary_sin, pos_t_op, 't', self.rope_t_size)
        sin_h = build_cossin(rotary_sin, pos_h_op, 'h', self.rope_h_size)
        sin_w = build_cossin(rotary_sin, pos_w_op, 'w', self.rope_w_size)
        sin_hw = top.ConcatOp(T([N, self.rope_h_size + self.rope_w_size]), [sin_h, sin_w],
                              axis=1,
                              loc=L(rotary_sin + ".concat.hw"),
                              ip=ip).output
        sin_full = top.ConcatOp(T([N, half]), [sin_t, sin_hw],
                                axis=1,
                                loc=L(rotary_sin + ".concat.thw"),
                                ip=ip).output
        sin_op = top.TileOp(T([N, D]), sin_full, tile=[1, 2], loc=L(rotary_sin + ".tile"),
                            ip=ip).output
        sin_op = top.ReshapeOp(T([1, N, 1, D]),
                               sin_op,
                               shape=[1, N, 1, D],
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output

        # shared rotate_half permutation weight (created once, reused per layer)
        perm_w_op = vit_mlir.create_weight_op(perm_name, [D, D])

        # ---- vision blocks
        def vision_block(idx, in_op):
            bp = f"{layers}.{idx}"
            norm1 = f"{bp}.layer_norm1"
            attn_q = f"{bp}.self_attn.q"
            attn_k = f"{bp}.self_attn.k"
            attn_v = f"{bp}.self_attn.v"
            attn_proj = f"{bp}.self_attn.proj"
            norm2 = f"{bp}.layer_norm2"
            hidden_shape = [1, N, E]

            def vision_attention(in_op):
                h = self.layer_norm(vit_mlir, in_op, norm1, eps=self.vnorm_eps)
                wshp = [E, E]
                q_op = self.linear(vit_mlir, attn_q, h, wshp, hidden_shape, force_bias=True)
                k_op = self.linear(vit_mlir, attn_k, h, wshp, hidden_shape, force_bias=True)
                v_op = self.linear(vit_mlir, attn_v, h, wshp, hidden_shape, force_bias=True)
                qkv_shape = [1, N, H, D]
                q_op = top.ReshapeOp(T(qkv_shape),
                                     q_op,
                                     shape=[1, -1, H, D],
                                     loc=L(attn_q + ".reshape"),
                                     ip=ip).output
                k_op = top.ReshapeOp(T(qkv_shape),
                                     k_op,
                                     shape=[1, -1, H, D],
                                     loc=L(attn_k + ".reshape"),
                                     ip=ip).output
                v_op = top.ReshapeOp(T(qkv_shape),
                                     v_op,
                                     shape=[1, -1, H, D],
                                     loc=L(attn_v + ".reshape"),
                                     ip=ip).output
                q_op = self._apply_3d_rope(vit_mlir, q_op, cos_op, sin_op, perm_w_op,
                                           attn_q + ".rotary", ip)
                k_op = self._apply_3d_rope(vit_mlir, k_op, cos_op, sin_op, perm_w_op,
                                           attn_k + ".rotary", ip)
                fa_op = top.FAttentionOp(
                    T(qkv_shape),
                    q_op,
                    k_op,
                    v_op,
                    mask_op,
                    vit_mlir.none_op,
                    scale=D**-0.5,
                    batch=1,
                    q_head=H,
                    kv_head=H,  # MHA
                    dim=D,
                    mq=N,
                    mk=N,
                    keep_dims=True,
                    loc=L(f"{bp}.fattention"),
                    ip=ip).output
                fa_op = top.ReshapeOp(T(hidden_shape),
                                      fa_op,
                                      shape=[1, -1, E],
                                      loc=L(f"{bp}.fattention.reshape"),
                                      ip=ip).output
                out_op = self.linear(vit_mlir,
                                     attn_proj,
                                     fa_op,
                                     wshp,
                                     hidden_shape,
                                     force_bias=True)
                out_op = top.AddOp(T(hidden_shape), [in_op, out_op],
                                   loc=L(attn_proj + ".add"),
                                   ip=ip).output
                return out_op

            def vision_mlp(in_op):
                fc1 = f"{bp}.mlp.fc1"
                fc2 = f"{bp}.mlp.fc2"
                h = self.layer_norm(vit_mlir, in_op, norm2, eps=self.vnorm_eps)
                up = self.linear(vit_mlir, fc1, h, [E, self.vintermediate_size],
                                 [1, N, self.vintermediate_size])
                up = self.activate(vit_mlir, up, ActType.GELU, fc1)
                down = self.linear(vit_mlir, fc2, up, [self.vintermediate_size, E], hidden_shape)
                return top.AddOp(T(hidden_shape), [in_op, down], loc=L(fc2 + ".add"), ip=ip).output

            in_op = vision_attention(in_op)
            in_op = vision_mlp(in_op)
            return in_op

        for idx in range(self.vit_depth):
            x = vision_block(idx, x)

        # ---- merger: ln_q -> reshape [N/4, 4*E=4096] -> fc1 + GELU + fc2
        x = self.layer_norm(vit_mlir, x, merger_ln_q, eps=self.vnorm_eps)
        merge_out = E * (self.spatial_merge_size**2)  # 4096
        x = top.ReshapeOp(T([out_dim, merge_out]),
                          x,
                          shape=[-1, merge_out],
                          loc=L(merger_ln_q + ".reshape"),
                          ip=ip).output
        x = self.linear(vit_mlir, merger_fc1, x, [merge_out, merge_out], [out_dim, merge_out])
        x = self.activate(vit_mlir, x, ActType.GELU, merger_fc1)
        x = self.linear(vit_mlir, merger_fc2, x, [merge_out, self.vout_hidden_size],
                        [out_dim, self.vout_hidden_size])

        vit_mlir.create_return_op([x])
        self.save_mlir_module(vit_mlir, name)

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------

    @override
    def compile_vit(self):
        name = "vit"
        if self.register_bmodel(name):
            return
        # vision tower stays BF16 (kept out of AWQ): quantize as bf16
        vit_q = getattr(self, "vit_quantize", None) or self.half_precision_quantize
        if self.half_precision_quantize == 'bf16' and \
                getattr(self, "vit_f16_out_bf16", True) and vit_q in ('f16', 'bf16'):
            extra_args = ['--quantize f16', '--quant_output_bf16']
        else:
            extra_args = [f'--quantize {vit_q}', '--quant_output']
        if self.quant_mode is not None:
            extra_args.append(f'--q_group_size {self.q_group_size}')
        self.submit_deploy_task(
            name,
            extra_args,
            dynamic=self.dynamic,
            symmetric=self.symmetric,
        )

    # ------------------------------------------------------------------
    # Gate (StreamMind perception path: PreNet + Mamba + PostNet)
    # ------------------------------------------------------------------

    def _load_gate_weights(self):
        """Load StreamMind Gate weights from the separate safetensors file.
        Keep BF16 dtype to match main model."""
        import os
        model_path = self.model_path
        gate_path = os.path.join(model_path, "streammind_gate.safetensors")
        if not os.path.exists(gate_path):
            raise RuntimeError(f"Gate weights not found: {gate_path}")
        # Use torch loader to keep BF16 dtype
        from safetensors.torch import load_file as torch_load
        raw = torch_load(gate_path)
        # Extract only the perception path (pre_net, mamba, post_net)
        gate_w = {}
        for k, v in raw.items():
            if k.startswith(('pre_net.', 'mamba_model.', 'post_net.')):
                gate_w[k] = v  # Keep original dtype (BF16)
        return gate_w

    def _prepare_gate_weights(self, gate_w):
        """Precompute derived weights for the gate MLIR graph.
        Keep original dtype (BF16) for all weights."""
        T = self.gate_T
        D = self.gate_d_model
        d_inner = self.gate_d_inner
        d_state = self.gate_d_state
        dt_rank = self.gate_dt_rank

        # Precompute A = -exp(A_log) for selective scan
        # A_log is numpy array with original dtype (BF16)
        A_log = gate_w["mamba_model.ssms.0.mixer.A_log"]  # [d_inner, d_state]
        # Convert to float32 for exp calculation, then back to original dtype
        A_log_f32 = A_log.astype(np.float32) if A_log.dtype != np.float32 else A_log
        gate_w["gate._A"] = (-np.exp(A_log_f32)).astype(A_log.dtype)  # [d_inner, d_state]

        # Initial state zeros [1, d_inner, d_state]
        gate_w["gate._state_init"] = np.zeros((1, d_inner, d_state), dtype=A_log.dtype)

        # LeakyReLU constants
        gate_w["gate._zero_scalar"] = np.zeros((1, ), dtype=A_log.dtype)
        gate_w["gate._leaky_slope"] = np.full((1, ), self.gate_leaky_slope, dtype=A_log.dtype)

        # Weights for softplus calculation (to avoid AddConst lowering bug)
        # Create one weight for each timestep's softplus operation in selective scan
        for t in range(T):
            gate_w[f"gate._one_mamba.scan.t{t}"] = np.ones((1, 1, d_inner), dtype=A_log.dtype)

        # Reshape conv1d.weight from [d_inner, 1, d_conv] to [d_inner, 1, 1, d_conv]
        conv1d_w = gate_w["mamba_model.ssms.0.mixer.conv1d.weight"]
        gate_w["mamba_model.ssms.0.mixer.conv1d.weight"] = \
            np.expand_dims(conv1d_w, axis=1)  # Add dimension at position 1

        # Split in_proj weight [D, 2*d_inner] into x and z parts
        # Note: set_linear_weight transposes weight to [D, 2*d_inner]
        in_proj_w = gate_w["mamba_model.ssms.0.mixer.in_proj.weight"]
        gate_w["mamba_model.ssms.0.mixer.in_proj.x.weight"] = \
            np.ascontiguousarray(in_proj_w[:, :d_inner])
        gate_w["mamba_model.ssms.0.mixer.in_proj.z.weight"] = \
            np.ascontiguousarray(in_proj_w[:, d_inner:])

        # Split x_proj weight [d_inner, dt_rank+2*d_state] into dt, B, C parts
        # Note: set_linear_weight transposes weight to [d_inner, dt_rank+2*d_state]
        x_proj_w = gate_w["mamba_model.ssms.0.mixer.x_proj.weight"]
        gate_w["mamba_model.ssms.0.mixer.x_proj.dt.weight"] = \
            np.ascontiguousarray(x_proj_w[:, :dt_rank])
        gate_w["mamba_model.ssms.0.mixer.x_proj.B.weight"] = \
            np.ascontiguousarray(x_proj_w[:, dt_rank:dt_rank + d_state])
        gate_w["mamba_model.ssms.0.mixer.x_proj.C.weight"] = \
            np.ascontiguousarray(x_proj_w[:, dt_rank + d_state:])

        return gate_w

    def _gen_mamba_block(self, gate_mlir, x, TT, L, ip, none_op, weight_op, T, D, d_inner, d_state,
                         d_conv, dt_rank):
        """Generate MLIR for the Mamba block: LayerNorm -> Mamba -> residual + norm_fn."""
        residual = x

        # -- Block LayerNorm --
        x = self.layer_norm(gate_mlir, x, "mamba_model.ssms.0.norm", eps=1e-5)

        # -- in_proj: split into 2 separate MatMuls to avoid triggering
        # SplitMatMulPattern (which crashes on gate's complex IR) --
        # x_part: [1,T,D] @ [D, d_inner] -> [1,T,d_inner]
        x_part = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.in_proj.x", x, [D, d_inner],
                             [1, T, d_inner])
        # z_part: [1,T,D] @ [D, d_inner] -> [1,T,d_inner]
        z_part = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.in_proj.z", x, [D, d_inner],
                             [1, T, d_inner])

        # -- causal conv1d: manual causal conv for T=4 --
        conv_w_raw = weight_op("mamba_model.ssms.0.mixer.conv1d.weight", [d_inner, 1, 1, d_conv])
        conv_w_2d = top.ReshapeOp(TT([d_inner, d_conv]),
                                  conv_w_raw,
                                  shape=[d_inner, d_conv],
                                  loc=L("mamba.conv.w_reshape"),
                                  ip=ip).output
        conv_b = weight_op("mamba_model.ssms.0.mixer.conv1d.bias", [d_inner])
        y_conv_steps = []
        for t in range(T):
            y_t = top.ReshapeOp(TT([1, d_inner]),
                                conv_b,
                                shape=[1, d_inner],
                                loc=L(f"mamba.conv.t{t}.bias_reshape"),
                                ip=ip).output
            for k in range(min(t + 1, d_conv)):
                x_tk = top.SliceOp(TT([1, 1, d_inner]),
                                   x_part,
                                   none_op,
                                   none_op,
                                   none_op,
                                   offset=[0, t - k, 0],
                                   steps=[1, 1, 1],
                                   ends=[1, t - k + 1, d_inner],
                                   axes=[],
                                   loc=L(f"mamba.conv.t{t}.k{k}.slice"),
                                   ip=ip).output
                x_tk = top.ReshapeOp(TT([1, d_inner]),
                                     x_tk,
                                     shape=[1, d_inner],
                                     loc=L(f"mamba.conv.t{t}.k{k}.reshape"),
                                     ip=ip).output
                # Reverse kernel index: PyTorch Conv1d uses cross-correlation
                # (no kernel flip), so kernel weight at lag k is w[d_conv-1-k]
                rev_k = d_conv - 1 - k
                w_k = top.SliceOp(TT([d_inner, 1]),
                                  conv_w_2d,
                                  none_op,
                                  none_op,
                                  none_op,
                                  offset=[0, rev_k],
                                  steps=[1, 1],
                                  ends=[d_inner, rev_k + 1],
                                  axes=[],
                                  loc=L(f"mamba.conv.t{t}.k{k}.w_slice"),
                                  ip=ip).output
                w_k = top.ReshapeOp(TT([d_inner]),
                                    w_k,
                                    shape=[d_inner],
                                    loc=L(f"mamba.conv.t{t}.k{k}.w_flat"),
                                    ip=ip).output
                wx = top.MulOp(TT([1, d_inner]), [x_tk, w_k],
                               loc=L(f"mamba.conv.t{t}.k{k}.mul"),
                               ip=ip).output
                y_t = top.AddOp(TT([1, d_inner]), [y_t, wx],
                                loc=L(f"mamba.conv.t{t}.k{k}.add"),
                                ip=ip).output
            y_conv_steps.append(
                top.ReshapeOp(TT([1, 1, d_inner]),
                              y_t,
                              shape=[1, 1, d_inner],
                              loc=L(f"mamba.conv.t{t}.out_reshape"),
                              ip=ip).output)
        x_conv = top.ConcatOp(TT([1, T, d_inner]),
                              y_conv_steps,
                              axis=1,
                              loc=L("mamba.conv.concat"),
                              ip=ip).output
        x_conv = self.activate(gate_mlir, x_conv, ActType.SILU, "mamba.conv.silu")

        # -- x_proj, dt_proj: split into 3 separate MatMuls to avoid
        # triggering SplitMatMulPattern crash --
        dt_raw = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.x_proj.dt", x_conv,
                             [d_inner, dt_rank], [1, T, dt_rank])
        B_raw = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.x_proj.B", x_conv,
                            [d_inner, d_state], [1, T, d_state])
        C_raw = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.x_proj.C", x_conv,
                            [d_inner, d_state], [1, T, d_state])
        dt_proj_raw = self.linear(gate_mlir,
                                  "mamba_model.ssms.0.mixer.dt_proj",
                                  dt_raw, [dt_rank, d_inner], [1, T, d_inner],
                                  force_bias=True)

        # -- Precomputed A and state init --
        A_op = weight_op("gate._A", [d_inner, d_state])
        state_shape = [1, d_inner, d_state]
        state_op = weight_op("gate._state_init", state_shape)

        # -- Pre-slice to avoid MatMul+Slice pattern in loop --
        x_conv_t = []
        z_part_t = []
        B_raw_t = []
        C_raw_t = []
        dt_proj_raw_t = []
        for t in range(T):
            x_conv_t.append(
                top.SliceOp(TT([1, 1, d_inner]),
                            x_conv,
                            none_op,
                            none_op,
                            none_op,
                            offset=[0, t, 0],
                            steps=[1, 1, 1],
                            ends=[1, t + 1, d_inner],
                            axes=[],
                            loc=L(f"mamba.preslice.x.t{t}"),
                            ip=ip).output)
            z_part_t.append(
                top.SliceOp(TT([1, 1, d_inner]),
                            z_part,
                            none_op,
                            none_op,
                            none_op,
                            offset=[0, t, 0],
                            steps=[1, 1, 1],
                            ends=[1, t + 1, d_inner],
                            axes=[],
                            loc=L(f"mamba.preslice.z.t{t}"),
                            ip=ip).output)
            B_raw_t.append(
                top.SliceOp(TT([1, 1, d_state]),
                            B_raw,
                            none_op,
                            none_op,
                            none_op,
                            offset=[0, t, 0],
                            steps=[1, 1, 1],
                            ends=[1, t + 1, d_state],
                            axes=[],
                            loc=L(f"mamba.preslice.B.t{t}"),
                            ip=ip).output)
            C_raw_t.append(
                top.SliceOp(TT([1, 1, d_state]),
                            C_raw,
                            none_op,
                            none_op,
                            none_op,
                            offset=[0, t, 0],
                            steps=[1, 1, 1],
                            ends=[1, t + 1, d_state],
                            axes=[],
                            loc=L(f"mamba.preslice.C.t{t}"),
                            ip=ip).output)
            dt_proj_raw_t.append(
                top.SliceOp(TT([1, 1, d_inner]),
                            dt_proj_raw,
                            none_op,
                            none_op,
                            none_op,
                            offset=[0, t, 0],
                            steps=[1, 1, 1],
                            ends=[1, t + 1, d_inner],
                            axes=[],
                            loc=L(f"mamba.preslice.dt.t{t}"),
                            ip=ip).output)

        # -- Unrolled selective scan --
        y_step_ops = []
        for t in range(T):
            p = f"mamba.scan.t{t}"
            x_t = x_conv_t[t]
            z_t = z_part_t[t]
            B_t = B_raw_t[t]
            C_t = C_raw_t[t]
            dt_t = dt_proj_raw_t[t]
            dt_exp = top.ExpOp(TT([1, 1, d_inner]), dt_t, loc=L(f"{p}.dt_exp"), ip=ip).output
            one_weight = weight_op(f"gate._one_{p}", [1, 1, d_inner])
            dt_exp1 = top.AddOp(TT([1, 1, d_inner]), [dt_exp, one_weight],
                                loc=L(f"{p}.dt_add1"),
                                ip=ip).output
            dt_sp = top.LogOp(TT([1, 1, d_inner]), dt_exp1, loc=L(f"{p}.dt_softplus"), ip=ip).output
            dt_r = top.ReshapeOp(TT([1, d_inner, 1]),
                                 dt_sp,
                                 shape=[1, d_inner, 1],
                                 loc=L(f"{p}.dt_reshape"),
                                 ip=ip).output
            x_r = top.ReshapeOp(TT([1, d_inner, 1]),
                                x_t,
                                shape=[1, d_inner, 1],
                                loc=L(f"{p}.x_reshape"),
                                ip=ip).output
            A_bcast = top.ReshapeOp(TT([1, d_inner, d_state]),
                                    A_op,
                                    shape=[1, d_inner, d_state],
                                    loc=L(f"{p}.A_reshape"),
                                    ip=ip).output
            dt_A = top.MulOp(TT([1, d_inner, d_state]), [dt_r, A_bcast], loc=L(f"{p}.dt_A"),
                             ip=ip).output
            A_bar = top.ExpOp(TT([1, d_inner, d_state]), dt_A, loc=L(f"{p}.A_bar"), ip=ip).output
            B_bar = top.MulOp(TT([1, d_inner, d_state]), [dt_r, B_t], loc=L(f"{p}.B_bar"),
                              ip=ip).output
            st_sc = top.MulOp(TT(state_shape), [A_bar, state_op], loc=L(f"{p}.state_scale"),
                              ip=ip).output
            Bx = top.MulOp(TT(state_shape), [B_bar, x_r], loc=L(f"{p}.Bx"), ip=ip).output
            state_op = top.AddOp(TT(state_shape), [st_sc, Bx], loc=L(f"{p}.state_update"),
                                 ip=ip).output
            st_C = top.MulOp(TT(state_shape), [state_op, C_t], loc=L(f"{p}.state_C"), ip=ip).output
            y_t = top.ReduceOp(TT([1, d_inner]),
                               st_C,
                               axes=[2],
                               keepdims=False,
                               mode=StringAttr.get("ReduceSum"),
                               loc=L(f"{p}.reduce"),
                               ip=ip).output
            D_op = weight_op("mamba_model.ssms.0.mixer.D", [d_inner])
            D_bc = top.ReshapeOp(TT([1, d_inner]),
                                 D_op,
                                 shape=[1, d_inner],
                                 loc=L(f"{p}.D_reshape"),
                                 ip=ip).output
            x_flat = top.ReshapeOp(TT([1, d_inner]),
                                   x_t,
                                   shape=[1, d_inner],
                                   loc=L(f"{p}.x_flat"),
                                   ip=ip).output
            Dx = top.MulOp(TT([1, d_inner]), [D_bc, x_flat], loc=L(f"{p}.Dx"), ip=ip).output
            y_t = top.AddOp(TT([1, d_inner]), [y_t, Dx], loc=L(f"{p}.y_skip"), ip=ip).output
            z_flat = top.ReshapeOp(TT([1, d_inner]),
                                   z_t,
                                   shape=[1, d_inner],
                                   loc=L(f"{p}.z_flat"),
                                   ip=ip).output
            z_gate = top.SiLUOp(TT([1, d_inner]), z_flat, loc=L(f"{p}.z_silu"), ip=ip).output
            y_t = top.MulOp(TT([1, d_inner]), [y_t, z_gate], loc=L(f"{p}.y_gate"), ip=ip).output
            y_t = top.ReshapeOp(TT([1, 1, d_inner]),
                                y_t,
                                shape=[1, 1, d_inner],
                                loc=L(f"{p}.y_reshape"),
                                ip=ip).output
            y_step_ops.append(y_t)

        y = top.ConcatOp(TT([1, T, d_inner]), y_step_ops, axis=1, loc=L("mamba.scan.concat"),
                         ip=ip).output

        # -- out_proj + residual + norm_fn --
        y = self.linear(gate_mlir, "mamba_model.ssms.0.mixer.out_proj", y, [d_inner, D], [1, T, D])
        y = top.AddOp(TT([1, T, D]), [y, residual], loc=L("mamba.block_residual"), ip=ip).output
        y = self.layer_norm(gate_mlir, y, "mamba_model.norm_fn", eps=1e-5)
        return y

    def gen_gate_mlir(self):
        """Generate MLIR for the Gate perception path:
        PreNet (Linear + LeakyReLU) -> Mamba Block -> PostNet (LeakyReLU + Linear).

        Input: [1, T, D] mean-pooled vision tokens
        Output: [1, T, D] processed perception tokens
        """
        tqdm.write("generate gate mlir ...")
        name = "gate"
        gate_npz = "gate_top_weights.npz"

        # -- load weights using ModelHandle methods (preserves dtype) --
        gate_weights = {}

        # PreNet weights (linear layer — only set_linear_weight, NOT set_common_weight
        # which would overwrite the transposed weight with the original)
        self.set_linear_weight("pre_net.fc3", gate_weights)

        # Mamba weights
        self.set_common_weight("mamba_model.ssms.0.norm", gate_weights)
        self.set_linear_weight("mamba_model.ssms.0.mixer.in_proj", gate_weights)
        self.set_common_weight("mamba_model.ssms.0.mixer.conv1d", gate_weights)
        self.set_linear_weight("mamba_model.ssms.0.mixer.x_proj", gate_weights)
        self.set_linear_weight("mamba_model.ssms.0.mixer.dt_proj", gate_weights)
        # Read A_log and D directly (not standard weight/bias)
        if self.model.is_exist("mamba_model.ssms.0.mixer.A_log"):
            gate_weights["mamba_model.ssms.0.mixer.A_log"] = self.model.read(
                "mamba_model.ssms.0.mixer.A_log")
        if self.model.is_exist("mamba_model.ssms.0.mixer.D"):
            gate_weights["mamba_model.ssms.0.mixer.D"] = self.model.read(
                "mamba_model.ssms.0.mixer.D")
        self.set_linear_weight("mamba_model.ssms.0.mixer.out_proj", gate_weights)
        self.set_common_weight("mamba_model.norm_fn", gate_weights)

        # PostNet weights (linear layer — same as PreNet, no set_common_weight)
        self.set_linear_weight("post_net.fc3", gate_weights)

        # Precompute derived weights
        gate_w = self._prepare_gate_weights(gate_weights)

        # Save weights (preserving original dtype from model)
        np.savez(gate_npz, **gate_w)
        self.weight_keys.extend(list(gate_w.keys()))

        T = self.gate_T  # 4
        D = self.gate_d_model  # 2560
        d_inner = self.gate_d_inner  # 5120
        d_state = self.gate_d_state  # 16
        d_conv = self.gate_d_conv  # 4
        dt_rank = self.gate_dt_rank  # 160

        in_shape = [1, T, D]
        out_shape = [1, T, D]

        # Use BF16 to match main model dtype
        gate_mlir = MLIRImporter([in_shape], [out_shape],
                                 name,
                                 self.platform, ['BF16'],
                                 weight_file=f"../{gate_npz}")
        ip = gate_mlir.insert_point
        TT = gate_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, gate_mlir)
        none_op = gate_mlir.none_op

        in_op = gate_mlir.create_input_op(L('gate_input'), 0)  # [1, T, D]

        # ---- helpers ----
        def weight_op(key, shape):
            return gate_mlir.create_weight_op(key, shape)

        def leaky_relu(x, shape, name_prefix):
            """LeakyReLU: max(x,0) + slope * min(x,0)"""
            zero_op = weight_op("gate._zero_scalar", [1])
            slope_op = weight_op("gate._leaky_slope", [1])
            pos = top.MaxOp(TT(shape), [x, zero_op], loc=L(f"{name_prefix}.max"), ip=ip).output
            neg = top.MinOp(TT(shape), [x, zero_op], loc=L(f"{name_prefix}.min"), ip=ip).output
            neg_s = top.MulOp(TT(shape), [neg, slope_op], loc=L(f"{name_prefix}.mul_slope"),
                              ip=ip).output
            return top.AddOp(TT(shape), [pos, neg_s], loc=L(f"{name_prefix}.leaky_relu"),
                             ip=ip).output

        # ================================================================
        # PreNet: Linear + LeakyReLU
        # ================================================================
        x = self.linear(gate_mlir, "pre_net.fc3", in_op, [D, D], [1, T, D], force_bias=True)
        x = leaky_relu(x, [1, T, D], "pre_net")

        # ================================================================
        # Mamba Block: LayerNorm -> Mamba -> residual + norm_fn
        # ================================================================
        y = self._gen_mamba_block(gate_mlir, x, TT, L, ip, none_op, weight_op, T, D, d_inner,
                                  d_state, d_conv, dt_rank)

        # ================================================================
        # PostNet: LeakyReLU + Linear
        # ================================================================
        y = leaky_relu(y, [1, T, D], "post_net")
        y = self.linear(gate_mlir, "post_net.fc3", y, [D, D], [1, T, D], force_bias=True)

        gate_mlir.create_return_op([y])
        self.save_mlir_module(gate_mlir, name)

    # ------------------------------------------------------------------
    # Gate compile
    # ------------------------------------------------------------------

    def compile_gate(self):
        name = "gate"
        if self.register_bmodel(name):
            return
        # Gate stays BF16 (no AWQ quantization on gate weights)
        self.submit_deploy_task(
            name,
            ['--quantize bf16', '--quant_output'],
            dynamic=False,
            symmetric=self.symmetric,
        )

    # ------------------------------------------------------------------
    # ClsNet (4-layer micro Qwen3 binary classifier: silent/speak)
    # ------------------------------------------------------------------

    def gen_cls_net_mlir(self):
        """Generate MLIR for ClsNet: 4-layer micro Qwen3 binary classifier.

        Reference (gate_cpu.py ClsNet.forward):
          pair = stack([perception, embed_tokens(zeros)], dim=1) → [B*T, 2, D]
          Run 4 Qwen3 layers with causal attention on seq=2 per pair.
          logits = lm_head(final_norm(output))[:, 0, :]

        MLIR: batch=T independent 2-token sequences. Each frame's perception
        token + target embedding form one pair processed independently.

        Input: [1, T, D] perception tokens from gate PostNet (T=4, D=2560)
        Output: [1, T, 2] silent/speak logits
        """
        tqdm.write("generate cls_net mlir ...")
        name = "cls_net"
        cls_npz = "cls_net_top_weights.npz"

        # -- load weights via self.model.read() (BF16, same as Gate/LLM) --
        cls_weights = {}

        T = self.gate_T  # 4
        S = 2 * T  # 8 (interleaved pair sequence)
        D = self.cls_hidden_size  # 2560
        num_layers = self.cls_num_layers  # 4
        intermediate = self.cls_intermediate_size  # 12288
        vocab = self.cls_vocab_size  # 2

        num_heads = 32
        kv_heads = 8
        head_dim = 128
        q_dim = num_heads * head_dim  # 4096
        kv_dim = kv_heads * head_dim  # 1024
        q_shape = [1, S, num_heads, head_dim]
        kv_shape = [1, S, kv_heads, head_dim]

        # Match the source weight dtype (BF16 stored as the model's read() dtype)
        # so derived weights don't introduce F32/BF16 mixed dtype.
        embed_key = "cls_net.cls_model.model.embed_tokens.weight"
        embed_w = self.model.read(embed_key)  # [vocab=2, D], BF16 (numpy)
        w_dtype = embed_w.dtype

        # RoPE tables: CUDA uses position_ids=[0,1] for each pair (batch=T).
        # After reshaping pair to [1, S=8, D], we need positions [0,1,0,1,0,1,0,1].
        cls_rope_theta = 10000.0
        inv_freq = 1.0 / (cls_rope_theta**(np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pair_freqs = np.outer(np.arange(2, dtype=np.float32), inv_freq)  # [2, 64]
        pair_cos = np.concatenate([np.cos(pair_freqs), np.cos(pair_freqs)], axis=1)
        pair_sin = np.concatenate([np.sin(pair_freqs), np.sin(pair_freqs)], axis=1)
        # Tile for T pairs: [0,1,0,1,0,1,0,1]
        rope_cos = np.tile(pair_cos, (T, 1)).reshape(1, S, 1, head_dim).astype(w_dtype)
        rope_sin = np.tile(pair_sin, (T, 1)).reshape(1, S, 1, head_dim).astype(w_dtype)
        cls_weights["cls_rope_cos.weight"] = np.ascontiguousarray(rope_cos)
        cls_weights["cls_rope_sin.weight"] = np.ascontiguousarray(rope_sin)
        # Block-diagonal causal mask [1, 1, S, S]:
        # Each pair (p_i, t_i) has its own 2×2 causal block; cross-pair masked.
        causal_mask = np.full((1, 1, S, S), -1e9, dtype=w_dtype)
        for pair_idx in range(T):
            base = pair_idx * 2
            causal_mask[0, 0, base, base] = 0  # p_i sees itself
            causal_mask[0, 0, base + 1, base] = 0  # t_i sees p_i
            causal_mask[0, 0, base + 1, base + 1] = 0  # t_i sees itself
        cls_weights["cls_causal_mask.weight"] = causal_mask
        # Target embedding for token ID 0 (embed_tokens.weight[0]).
        target_emb = embed_w[0]  # [D]
        cls_weights["cls_target_emb.weight"] = np.ascontiguousarray(
            np.broadcast_to(target_emb, (1, T, 1, D)).copy())

        # Linear + norm weights via set_linear_weight / set_common_weight
        # (transposes [out,in]→[in,out] and reads BF16 from ModelHandle).
        for i in range(num_layers):
            lp = f"cls_net.cls_model.model.layers.{i}"
            self.set_linear_weight(f"{lp}.self_attn.q_proj", cls_weights)
            self.set_linear_weight(f"{lp}.self_attn.k_proj", cls_weights)
            self.set_linear_weight(f"{lp}.self_attn.v_proj", cls_weights)
            self.set_linear_weight(f"{lp}.self_attn.o_proj", cls_weights)
            self.set_linear_weight(f"{lp}.mlp.gate_proj", cls_weights)
            self.set_linear_weight(f"{lp}.mlp.up_proj", cls_weights)
            self.set_linear_weight(f"{lp}.mlp.down_proj", cls_weights)
            self.set_common_weight(f"{lp}.input_layernorm", cls_weights)
            self.set_common_weight(f"{lp}.post_attention_layernorm", cls_weights)
            self.set_common_weight(f"{lp}.self_attn.q_norm", cls_weights)
            self.set_common_weight(f"{lp}.self_attn.k_norm", cls_weights)
        self.set_linear_weight("cls_net.cls_model.lm_head", cls_weights)
        self.set_common_weight("cls_net.cls_model.model.norm", cls_weights)

        np.savez(cls_npz, **cls_weights)
        self.weight_keys.extend(list(cls_weights.keys()))

        in_shape = [1, T, D]
        out_shape = [1, T, vocab]

        cls_mlir = MLIRImporter([in_shape], [out_shape],
                                name,
                                self.platform, ['BF16'],
                                weight_file=f"../{cls_npz}")
        ip = cls_mlir.insert_point
        TT = cls_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, cls_mlir)
        none_op = cls_mlir.none_op

        postnet_in = cls_mlir.create_input_op(L('cls_input'), 0)  # [1, T, D]

        # -- Build interleaved pair: [1, T, D] → [1, S, D] --
        # perception: [1, T, D] → [1, T, 1, D]
        p_4d = top.ReshapeOp(TT([1, T, 1, D]),
                             postnet_in,
                             shape=[1, T, 1, D],
                             loc=L("pair.p_reshape"),
                             ip=ip).output
        # target = embed_tokens(0) [1, T, 1, D]
        t_4d = cls_mlir.create_weight_op("cls_target_emb.weight", [1, T, 1, D])
        # Concat along axis=2: [1, T, 2, D]
        pair_4d = top.ConcatOp(TT([1, T, 2, D]), [p_4d, t_4d],
                               axis=2,
                               only_merge=False,
                               loc=L("pair.concat"),
                               ip=ip).output
        # Reshape to [1, 2*T, D] = [1, S, D] (interleaved p0,t0,p1,t1,...)
        x = top.ReshapeOp(TT([1, S, D]), pair_4d, shape=[1, S, D], loc=L("pair.flatten"),
                          ip=ip).output

        # Load RoPE and mask constants.
        cos_op = cls_mlir.create_weight_op("cls_rope_cos.weight", [1, S, 1, head_dim])
        sin_op = cls_mlir.create_weight_op("cls_rope_sin.weight", [1, S, 1, head_dim])
        mask_op = cls_mlir.create_weight_op("cls_causal_mask.weight", [1, 1, S, S])

        # -- 4 transformer layers: batch=1, seq=S=8, block-diagonal attention --
        for i in range(num_layers):
            lp = f"cls_net.cls_model.model.layers.{i}"
            p = f"layers.{i}"
            residual = x  # [1, S, D]

            # Input RMSNorm
            x_norm = self.rms_norm(cls_mlir, x, f"{lp}.input_layernorm", eps=1e-6)

            # Q, K, V projections
            q_op = self.linear(cls_mlir, f"{lp}.self_attn.q_proj", x_norm, [D, q_dim],
                               [1, S, q_dim])
            k_op = self.linear(cls_mlir, f"{lp}.self_attn.k_proj", x_norm, [D, kv_dim],
                               [1, S, kv_dim])
            v_op = self.linear(cls_mlir, f"{lp}.self_attn.v_proj", x_norm, [D, kv_dim],
                               [1, S, kv_dim])

            # Reshape to multi-head
            q_op = top.ReshapeOp(TT(q_shape), q_op, shape=q_shape, loc=L(f"{p}.q_reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(TT(kv_shape), k_op, shape=kv_shape, loc=L(f"{p}.k_reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(TT(kv_shape), v_op, shape=kv_shape, loc=L(f"{p}.v_reshape"),
                                 ip=ip).output

            # QK-norm
            q_op = self.rms_norm(cls_mlir, q_op, f"{lp}.self_attn.q_norm", eps=1e-6)
            k_op = self.rms_norm(cls_mlir, k_op, f"{lp}.self_attn.k_norm", eps=1e-6)

            # 1D RoPE (contiguous_halves = Qwen3 style)
            q_op = top.RopeOp(TT(q_shape),
                              q_op,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L(f"{p}.q_rope"),
                              ip=ip).output
            k_op = top.RopeOp(TT(kv_shape),
                              k_op,
                              sin_op,
                              cos_op,
                              rope_mode=StringAttr.get("contiguous_halves"),
                              loc=L(f"{p}.k_rope"),
                              ip=ip).output

            # Fused attention with block-diagonal mask
            attn_out = top.FAttentionOp(TT([1, S, q_dim]),
                                        q_op,
                                        k_op,
                                        v_op,
                                        mask_op,
                                        none_op,
                                        scale=head_dim**-0.5,
                                        batch=1,
                                        q_head=num_heads,
                                        kv_head=kv_heads,
                                        dim=head_dim,
                                        mq=S,
                                        mk=S,
                                        keep_dims=False,
                                        mask_size=0,
                                        loc=L(f"{p}.fattention"),
                                        ip=ip).output

            # O projection
            o_out = self.linear(cls_mlir, f"{lp}.self_attn.o_proj", attn_out, [q_dim, D], [1, S, D])

            # Residual
            x = top.AddOp(TT([1, S, D]), [residual, o_out], loc=L(f"{p}.attn_residual"),
                          ip=ip).output

            # Post-attention RMSNorm
            residual = x
            x_norm = self.rms_norm(cls_mlir, x, f"{lp}.post_attention_layernorm", eps=1e-6)

            # SwiGLU MLP
            gate_out = self.linear(cls_mlir, f"{lp}.mlp.gate_proj", x_norm, [D, intermediate],
                                   [1, S, intermediate])
            gate_out = self.activate(cls_mlir, gate_out, ActType.SILU, f"{p}.mlp.gate_silu")
            up_out = self.linear(cls_mlir, f"{lp}.mlp.up_proj", x_norm, [D, intermediate],
                                 [1, S, intermediate])
            mlp_hidden = top.MulOp(TT([1, S, intermediate]), [gate_out, up_out],
                                   loc=L(f"{p}.mlp.swiglu"),
                                   ip=ip).output
            mlp_out = self.linear(cls_mlir, f"{lp}.mlp.down_proj", mlp_hidden, [intermediate, D],
                                  [1, S, D])

            # Residual
            x = top.AddOp(TT([1, S, D]), [residual, mlp_out], loc=L(f"{p}.mlp_residual"),
                          ip=ip).output

        # -- Extract perception positions (0, 2, 4, 6) --
        # [1, S, D] → [1, T, 2, D] → slice [:, :, 0:1, :] → [1, T, 1, D]
        x_4d = top.ReshapeOp(TT([1, T, 2, D]),
                             x,
                             shape=[1, T, 2, D],
                             loc=L("extract.reshape"),
                             ip=ip).output
        x_perception = top.SliceOp(TT([1, T, 1, D]),
                                   x_4d,
                                   none_op,
                                   none_op,
                                   none_op,
                                   offset=[0, 0, 0, 0],
                                   steps=[1, 1, 1, 1],
                                   ends=[1, T, 1, D],
                                   axes=[],
                                   loc=L("extract.perception"),
                                   ip=ip).output
        x_perception = top.ReshapeOp(TT([1, T, D]),
                                     x_perception,
                                     shape=[1, T, D],
                                     loc=L("extract.squeeze"),
                                     ip=ip).output

        # -- Final RMSNorm + lm_head --
        x_perception = self.rms_norm(cls_mlir,
                                     x_perception,
                                     "cls_net.cls_model.model.norm",
                                     eps=1e-6)
        logits = self.linear(cls_mlir, "cls_net.cls_model.lm_head", x_perception, [D, vocab],
                             [1, T, vocab])

        cls_mlir.create_return_op([logits])
        self.save_mlir_module(cls_mlir, name)

    def compile_cls_net(self):
        name = "cls_net"
        if self.register_bmodel(name):
            return
        # ClsNet stays BF16 (no AWQ quantization)
        self.submit_deploy_task(
            name,
            ['--quantize bf16', '--quant_output'],
            dynamic=False,
            symmetric=self.symmetric,
        )

    # ------------------------------------------------------------------
    # Pipeline registration
    # ------------------------------------------------------------------

    @override
    def gen_all_mlir(self):
        super().gen_all_mlir()
        if self.do_gate:
            self.all_gen_mlirs.append(self.gen_gate_mlir)
            self.gen_gate_mlir()
            self.all_gen_mlirs.append(self.gen_cls_net_mlir)
            self.gen_cls_net_mlir()

    @override
    def compile_all(self):
        if self.do_gate:
            self.all_compiles.append(self.compile_gate)
            self.all_compiles.append(self.compile_cls_net)
        super().compile_all()
