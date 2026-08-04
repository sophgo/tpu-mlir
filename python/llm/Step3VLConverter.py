# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Step3VLConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        self._args = args
        super().__init__(args, config, loader=loader)
        # Detect weight path format: model.language_model.layers (VLM composite)
        # vs model.layers (flat). Both are valid for Step3-VL depending on the
        # checkpoint source (e.g. compressed-tensors vs original AWQ).
        index_file = os.path.join(args.model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file) as f:
                idx = json.load(f)
            if any("language_model.layers" in k for k in idx.get("weight_map", {})):
                self.model_info = QWEN3VL_INFO
        self.init_vconfig()
        # ViT prefix: model.vision_model (composite) or vision_model (flat)
        if self.model_info is not COMMON_INFO:
            self.vit_path = "model.vision_model"
        else:
            self.vit_path = "vision_model"
        # Fused QKV key: detect from actual weight map
        # qkv_proj.weight (original AWQ) vs in_proj_weight (compressed-tensors / some checkpoints)
        wm = idx.get("weight_map", {}) if os.path.exists(index_file) else {}
        if any("in_proj_weight" in k for k in wm):
            self.fused_qkv_w_key = "attn.in_proj_weight"
            self.fused_qkv_b_key = "attn.in_proj_bias"
        else:
            self.fused_qkv_w_key = "attn.qkv_proj.weight"
            self.fused_qkv_b_key = "attn.qkv_proj.bias"
        # extern mlirs
        self.all_gen_mlirs.append(self.gen_vit_global_mlir)
        self.all_compiles.append(self.compile_vit_global_mlir)
        # Only compile vit_patch if max_patches > 0 (user specified
        # --max_pixels that produces patches). Saves ~3.5GB bmodel size.
        if self.max_patches > 0:
            self.all_gen_mlirs.append(self.gen_vit_patch_mlir)
            self.all_compiles.append(self.compile_vit_patch_mlir)

    def init_vconfig(self):
        self.do_vit = False  # We handle vit_global/vit_patch manually
        vconfig = self.config.vision_config
        self.vconfig = vconfig
        # Vision parameters
        self.patch_size = vconfig.patch_size  # 14
        self.embed_dim = vconfig.width  # 1536
        self.vnum_heads = vconfig.heads  # 16
        self.vhead_dim = self.embed_dim // self.vnum_heads  # 96
        self.vit_depth = vconfig.layers  # 47
        self.image_size = vconfig.image_size  # 728
        self.mlp_ratio = getattr(vconfig, 'mlp_ratio', 8960 / 1536)
        self.vintermediate_size = int(self.embed_dim * self.mlp_ratio)  # 8960
        self.layer_norm_eps = getattr(vconfig, 'layer_norm_eps', 1e-5)
        self.ls_init_value = getattr(vconfig, 'ls_init_value', 0.1)
        self.use_ln_pre = getattr(vconfig, 'use_ln_pre', True)
        self.use_ln_post = getattr(vconfig, 'use_ln_post', False)
        # Grid sizes
        self.global_grid = self.image_size // self.patch_size  # 52
        # For patch views (504x504): 504 / 14 = 36
        self.patch_image_size = 504
        self.patch_grid = self.patch_image_size // self.patch_size  # 36
        # Derive max_patches from user-specified max_pixels.
        # Replicates HF ImagePatcher.determine_window_size + slide_window
        # to estimate the actual max patch count for the given resolution.
        # Default max_pixels (728×728) produces 0 patches → fall back to 4.
        max_shape = getattr(self._args, 'max_shape', [728, 728])
        self.max_patches = self._estimate_max_patches(int(max_shape[0]), int(max_shape[1]))
        print(f"[Step3VL] max_pixels={max_shape[0]}x{max_shape[1]} → "
              f"max_patches={self.max_patches}"
              f"{' (vit_patch skipped)' if self.max_patches == 0 else ''}")
        # After downsamplers (2× stride-2 Conv2d)
        # Conv2d(in, out, k=3, s=2, p=1): out_size = (in - 1) // 2 + 1
        self.global_ds1_h = (self.global_grid - 1) // 2 + 1  # 26
        self.global_ds1_w = self.global_ds1_h
        self.global_ds2_h = (self.global_ds1_h - 1) // 2 + 1  # 13
        self.global_ds2_w = self.global_ds2_h
        self.patch_ds1_h = (self.patch_grid - 1) // 2 + 1  # 18
        self.patch_ds1_w = self.patch_ds1_h
        self.patch_ds2_h = (self.patch_ds1_h - 1) // 2 + 1  # 9
        self.patch_ds2_w = self.patch_ds2_h
        self.global_out_tokens = self.global_ds2_h * self.global_ds2_w  # 169
        self.patch_out_tokens = self.patch_ds2_h * self.patch_ds2_w  # 81
        self.projector_in_dim = self.embed_dim * 4  # 6144
        # ViT hidden act
        self.vit_hidden_act = ActType.QUICK_GELU

    @staticmethod
    def _estimate_max_patches(h, w):
        """Estimate max patches for a given max_pixels resolution (H, W).

        Replicates HF ImagePatcher logic: determine_window_size + slide_window.
        Returns 4 as default when the resolution produces 0 patches (e.g. 728×728).
        """
        from math import ceil
        max_image_size = 3024
        long_side = max(h, w)
        short_side = min(h, w)
        # Preprocess: resize if exceeds max_image_size
        if long_side > max_image_size:
            scale = max_image_size / long_side
            h = int(h * scale)
            w = int(w * scale)
        # Determine window size
        long_s = max(h, w)
        short_s = min(h, w)
        if long_s <= 728:
            window_size = short_s if long_s / short_s > 1.5 else 0
        else:
            window_size = min(short_s, 504) if long_s / short_s > 4 else 504
        if window_size == 0:
            return 0  # No patches needed for this resolution
        # Slide window: compute number of crops
        w_ratio = w / window_size
        h_ratio = h / window_size
        w_new = window_size * (int(w_ratio) +
                               (1 if w_ratio % 1 > 0.2 else 0)) if w_ratio >= 1 else w
        h_new = window_size * (int(h_ratio) +
                               (1 if h_ratio % 1 > 0.2 else 0)) if h_ratio >= 1 else h
        x_num = max(1, ceil((w_new - window_size) / window_size + 1))
        y_num = max(1, ceil((h_new - window_size) / window_size + 1))
        return x_num * y_num

    # =========================================================================
    # 2D RoPE precomputation
    # =========================================================================

    def compute_2d_rope_cos_sin(self, grid_h, grid_w):
        """Precompute 2D RoPE cos/sin table for a given grid.

        Returns:
            cos: np.array of shape [grid_h*grid_w, head_dim], float32
            sin: np.array of shape [grid_h*grid_w, head_dim], float32
        """
        dim = self.vhead_dim  # 96
        half_dim = dim // 2  # 48
        quarter_dim = half_dim // 2  # 24
        theta = 10000.0

        # inv_freq: [24]
        inv_freq = 1.0 / (theta
                          **(np.arange(0, half_dim, 2, dtype=np.float32)[:quarter_dim] / half_dim))

        # freqs_w: [grid_w, 48] — repeat each freq to [cos_freq, cos_freq]
        freqs_w = np.einsum('w,f->wf', np.arange(grid_w, dtype=np.float32), inv_freq)
        freqs_w = np.repeat(freqs_w, 2, axis=-1)

        # freqs_h: [grid_h, 48]
        freqs_h = np.einsum('h,f->hf', np.arange(grid_h, dtype=np.float32), inv_freq)
        freqs_h = np.repeat(freqs_h, 2, axis=-1)

        # Expand to grid: freqs_w [1, W, 48] → [H, W, 48], freqs_h [H, 1, 48] → [H, W, 48]
        freqs_w_full = np.broadcast_to(freqs_w[None, :, :], (grid_h, grid_w, half_dim))
        freqs_h_full = np.broadcast_to(freqs_h[:, None, :], (grid_h, grid_w, half_dim))

        # Concatenate: [H, W, 96] = [freqs_w, freqs_h]
        freqs = np.concatenate([freqs_w_full, freqs_h_full], axis=-1).reshape(-1, dim)

        cos = np.cos(freqs).astype(np.float32)
        sin = np.sin(freqs).astype(np.float32)
        return cos, sin

    # =========================================================================
    # ViT MLIR generation (shared between global and patch)
    # =========================================================================

    def _build_vit_mlir(self, is_global):
        """Build ViT MLIR graph for either the global or patch view."""
        suffix = "global" if is_global else "patch"
        name = f"vit_{suffix}"
        grid_h = self.global_grid if is_global else self.patch_grid
        grid_w = grid_h
        num_tokens = grid_h * grid_w  # 2704 or 1296
        max_batch = 1 if is_global else self.max_patches
        input_h = self.image_size if is_global else self.patch_image_size
        input_w = input_h
        ds_tokens = self.global_out_tokens if is_global else self.patch_out_tokens
        ds1_h = self.global_ds1_h if is_global else self.patch_ds1_h
        ds1_w = self.global_ds1_w if is_global else self.patch_ds1_w
        ds2_h = self.global_ds2_h if is_global else self.patch_ds2_h
        ds2_w = self.global_ds2_w if is_global else self.patch_ds2_w

        vit_npz = f"{name}_top_f32_all_origin_weight.npz"

        # Weight paths
        patch_embed = f"{self.vit_path}.conv1"
        posemb_path = f"{self.vit_path}.positional_embedding"
        ln_pre_path = f"{self.vit_path}.ln_pre"
        ds1_path = f"{self.vit_path}.vit_downsampler1"
        ds2_path = f"{self.vit_path}.vit_downsampler2"
        proj_path = f"{'model.' if self.model_info is not COMMON_INFO else ''}vit_large_projector"

        # Precompute 2D RoPE
        cos_table, sin_table = self.compute_2d_rope_cos_sin(grid_h, grid_w)

        # Precompute absolute positional embedding for the specific grid
        # by interpolating from the original 52×52 posemb
        abs_posemb = self._interpolate_posemb(grid_h, grid_w)

        # ---- save_weights ----
        def save_weights():
            weights_dict = {}

            # Conv1 patch embedding: weight [embed_dim, 3, patch_size, patch_size]
            conv1_w = self.model.read(patch_embed + ".weight")
            weights_dict[patch_embed + ".weight"] = conv1_w

            # Absolute positional embedding for this grid: [1, num_tokens, embed_dim]
            weights_dict[f"vit_{suffix}_abs_posemb"] = abs_posemb

            # 2D RoPE cos/sin: [num_tokens, vhead_dim]
            weights_dict[f"vit_{suffix}_rope_cos"] = cos_table
            weights_dict[f"vit_{suffix}_rope_sin"] = sin_table

            # ln_pre
            if self.use_ln_pre:
                weights_dict[ln_pre_path + ".weight"] = self.model.read(ln_pre_path + ".weight")
                weights_dict[ln_pre_path + ".bias"] = self.model.read(ln_pre_path + ".bias")

            # Transformer blocks
            for i in range(self.vit_depth):
                blk = f"{self.vit_path}.transformer.resblocks.{i}"

                # LayerNorms
                for ln_name in ["ln_1", "ln_2"]:
                    weights_dict[blk + "." + ln_name +
                                 ".weight"] = self.model.read(blk + "." + ln_name + ".weight")
                    weights_dict[blk + "." + ln_name + ".bias"] = self.model.read(blk + "." +
                                                                                  ln_name + ".bias")

                # Fused QKV → split into Q, K, V (all BF16).
                # Checkpoint only has fused QKV, not separate q/k/v.
                # Manual split + transpose: HF (out, in) → MatMul (in, out).
                qkv_w = self.model.read(blk + f".{self.fused_qkv_w_key}")
                qkv_b = self.model.read(blk + f".{self.fused_qkv_b_key}")
                D = self.embed_dim
                # qkv_w shape: [3*D, D], split along dim 0
                weights_dict[blk + ".attn.q.weight"] = np.ascontiguousarray(
                    np.transpose(qkv_w[:D, :], (1, 0)))
                weights_dict[blk + ".attn.k.weight"] = np.ascontiguousarray(
                    np.transpose(qkv_w[D:2 * D, :], (1, 0)))
                weights_dict[blk + ".attn.v.weight"] = np.ascontiguousarray(
                    np.transpose(qkv_w[2 * D:, :], (1, 0)))
                # Biases are 1D, no transpose needed.
                weights_dict[blk + ".attn.q.bias"] = np.ascontiguousarray(qkv_b[:D])
                weights_dict[blk + ".attn.k.bias"] = np.ascontiguousarray(qkv_b[D:2 * D])
                weights_dict[blk + ".attn.v.bias"] = np.ascontiguousarray(qkv_b[2 * D:])

                # Out projection (BF16)
                self.set_linear_weight(blk + ".attn.out_proj", weights_dict)

                # LayerScale gammas: shape [embed_dim]
                weights_dict[blk + ".ls_1.gamma"] = self.model.read(blk + ".ls_1.gamma")
                weights_dict[blk + ".ls_2.gamma"] = self.model.read(blk + ".ls_2.gamma")

                # MLP (c_fc may be AWQ for blocks 1-46, c_proj may be AWQ)
                self.set_linear_weight(blk + ".mlp.c_fc", weights_dict)
                self.set_linear_weight(blk + ".mlp.c_proj", weights_dict)

            # Downsampler convs
            weights_dict[ds1_path + ".weight"] = self.model.read(ds1_path + ".weight")
            weights_dict[ds1_path + ".bias"] = self.model.read(ds1_path + ".bias")
            weights_dict[ds2_path + ".weight"] = self.model.read(ds2_path + ".weight")
            weights_dict[ds2_path + ".bias"] = self.model.read(ds2_path + ".bias")

            # Projector (Linear, no bias): use set_linear_weight which
            # handles the HF (out, in) → MatMul (in, out) transpose.
            self.set_linear_weight(proj_path, weights_dict)

            np.savez(vit_npz, **weights_dict)
            self.weight_keys.extend(list(weights_dict.keys()))

        # Save weights first (needed before MLIR gen for is_key_quantized)
        save_weights()

        # ---- Build MLIR graph ----
        in_shape = [max_batch, 3, input_h, input_w]
        out_shape = [max_batch, ds_tokens, self.hidden_size]

        vit_mlir = MLIRImporter([in_shape], [out_shape],
                                name,
                                self.platform, ['F32'],
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, vit_mlir)

        # Input: [max_batch, 3, H, W]
        in_op = vit_mlir.create_input_op(L('pixel_values'), 0)

        # === Patch Embedding via Conv2d ===
        # Conv2d(3, embed_dim, k=14, s=14, no bias)
        conv_w = vit_mlir.create_weight_op(patch_embed + ".weight",
                                           [self.embed_dim, 3, self.patch_size, self.patch_size])
        conv_out = top.ConvOp(T([max_batch, self.embed_dim, grid_h, grid_w]),
                              in_op,
                              conv_w,
                              vit_mlir.none_op,
                              kernel_shape=[self.patch_size, self.patch_size],
                              strides=[self.patch_size, self.patch_size],
                              pads=[0, 0, 0, 0],
                              dilations=[1, 1],
                              loc=L(patch_embed),
                              ip=ip).output
        # [max_batch, D, Gh, Gw] → flatten → [max_batch, D, Gh*Gw]
        hidden = top.ReshapeOp(T([max_batch, self.embed_dim, num_tokens]),
                               conv_out,
                               shape=[max_batch, self.embed_dim, -1],
                               loc=L(patch_embed + ".flatten"),
                               ip=ip).output
        # → permute → [max_batch, Gh*Gw, D]
        hidden = top.PermuteOp(T([max_batch, num_tokens, self.embed_dim]),
                               hidden,
                               order=[0, 2, 1],
                               loc=L(patch_embed + ".permute"),
                               ip=ip).output

        # === Absolute Positional Embedding ===
        # posemb weight: [1, num_tokens, embed_dim], broadcasts with AddOp
        posemb_w = vit_mlir.create_weight_op(f"vit_{suffix}_abs_posemb",
                                             [1, num_tokens, self.embed_dim])
        hidden = top.AddOp(T([max_batch, num_tokens, self.embed_dim]), [hidden, posemb_w],
                           loc=L(posemb_path + ".add"),
                           ip=ip).output

        # === LayerNorm pre ===
        if self.use_ln_pre:
            hidden = self.layer_norm(vit_mlir, hidden, ln_pre_path, eps=self.layer_norm_eps)

        # === 2D RoPE (precomputed, lookup once for all blocks) ===
        cos_w = vit_mlir.create_weight_op(f"vit_{suffix}_rope_cos", [num_tokens, self.vhead_dim])
        sin_w = vit_mlir.create_weight_op(f"vit_{suffix}_rope_sin", [num_tokens, self.vhead_dim])
        # Reshape to [1, num_tokens, 1, vhead_dim] for RopeOp
        cos_op = top.ReshapeOp(T([1, num_tokens, 1, self.vhead_dim]),
                               cos_w,
                               shape=[1, num_tokens, 1, self.vhead_dim],
                               loc=L("rope_cos.reshape"),
                               ip=ip).output
        sin_op = top.ReshapeOp(T([1, num_tokens, 1, self.vhead_dim]),
                               sin_w,
                               shape=[1, num_tokens, 1, self.vhead_dim],
                               loc=L("rope_sin.reshape"),
                               ip=ip).output

        # === Transformer Blocks ===
        for i in range(self.vit_depth):
            hidden = self._vision_block(vit_mlir, i, hidden, cos_op, sin_op, num_tokens, max_batch,
                                        T, L, ip)

        # === Post LayerNorm ===
        if self.use_ln_post:
            ln_post_path = f"{self.vit_path}.ln_post"
            hidden = self.layer_norm(vit_mlir, hidden, ln_post_path, eps=self.layer_norm_eps)

        # === Downsampler 1: Conv2d(embed_dim, embed_dim*2, k=3, s=2, p=1) ===
        # hidden: [max_batch, num_tokens, D] → permute → [max_batch, D, num_tokens]
        hidden = top.PermuteOp(T([max_batch, self.embed_dim, num_tokens]),
                               hidden,
                               order=[0, 2, 1],
                               loc=L(ds1_path + ".pre_permute"),
                               ip=ip).output
        # → reshape → [max_batch, D, grid_h, grid_w]
        hidden = top.ReshapeOp(T([max_batch, self.embed_dim, grid_h, grid_w]),
                               hidden,
                               shape=[max_batch, self.embed_dim, grid_h, grid_w],
                               loc=L(ds1_path + ".pre_reshape"),
                               ip=ip).output

        ds1_wt = vit_mlir.create_weight_op(ds1_path + ".weight",
                                           [self.embed_dim * 2, self.embed_dim, 3, 3])
        ds1_b = vit_mlir.create_weight_op(ds1_path + ".bias", [self.embed_dim * 2])
        hidden = top.ConvOp(T([max_batch, self.embed_dim * 2, ds1_h, ds1_w]),
                            hidden,
                            ds1_wt,
                            ds1_b,
                            kernel_shape=[3, 3],
                            strides=[2, 2],
                            pads=[1, 1, 1, 1],
                            dilations=[1, 1],
                            loc=L(ds1_path),
                            ip=ip).output  # [max_batch, 3072, ds1_h, ds1_w]

        # === Downsampler 2: Conv2d(embed_dim*2, embed_dim*4, k=3, s=2, p=1) ===
        ds2_w_op = vit_mlir.create_weight_op(ds2_path + ".weight",
                                             [self.embed_dim * 4, self.embed_dim * 2, 3, 3])
        ds2_b = vit_mlir.create_weight_op(ds2_path + ".bias", [self.embed_dim * 4])
        hidden = top.ConvOp(T([max_batch, self.embed_dim * 4, ds2_h, ds2_w]),
                            hidden,
                            ds2_w_op,
                            ds2_b,
                            kernel_shape=[3, 3],
                            strides=[2, 2],
                            pads=[1, 1, 1, 1],
                            dilations=[1, 1],
                            loc=L(ds2_path),
                            ip=ip).output  # [max_batch, 6144, ds2_h, ds2_w]

        # Reshape to [max_batch, ds_tokens, embed_dim*4]
        final_ds_tokens = ds2_h * ds2_w
        hidden = top.ReshapeOp(T([max_batch, self.projector_in_dim, final_ds_tokens]),
                               hidden,
                               shape=[max_batch, self.projector_in_dim, -1],
                               loc=L(proj_path + ".reshape1"),
                               ip=ip).output
        hidden = top.PermuteOp(T([max_batch, final_ds_tokens, self.projector_in_dim]),
                               hidden,
                               order=[0, 2, 1],
                               loc=L(proj_path + ".permute"),
                               ip=ip).output

        # === Projector: Linear(embed_dim*4, hidden_size, no bias) ===
        hidden = self.linear(vit_mlir, proj_path, hidden, [self.projector_in_dim, self.hidden_size],
                             [max_batch, final_ds_tokens, self.hidden_size])

        vit_mlir.create_return_op([hidden])
        self.save_mlir_module(vit_mlir, name)

    def _interpolate_posemb(self, target_h, target_w):
        """Interpolate absolute positional embedding from 52×52 to target grid.

        Returns: np.array of shape [1, target_h*target_w, embed_dim]
        """
        raw = self.model.read(f"{self.vit_path}.positional_embedding")
        # raw shape: [52*52, 1536] = [2704, 1536]
        orig_grid = int(raw.shape[0]**0.5)  # 52
        if orig_grid == target_h and orig_grid == target_w:
            return raw[None, :, :]  # [1, N, D]

        # Reshape to [1, D, orig_grid, orig_grid] for interpolation
        import torch
        import torch.nn.functional as F
        posemb = torch.from_numpy(raw).float().reshape(1, orig_grid, orig_grid,
                                                       -1).permute(0, 3, 1, 2).contiguous()
        posemb = F.interpolate(posemb,
                               size=(target_h, target_w),
                               mode='bilinear',
                               align_corners=False)
        posemb = posemb.permute(0, 2, 3, 1).reshape(-1, self.embed_dim).numpy()
        return posemb[None, :, :]  # [1, N, D]

    def _vision_block(self, vit_mlir, block_id, hidden, cos_op, sin_op, num_tokens, batch, T, L,
                      ip):
        """Build one vision transformer block."""
        blk = f"{self.vit_path}.transformer.resblocks.{block_id}"
        ln1_path = blk + ".ln_1"
        ln2_path = blk + ".ln_2"
        attn_q = blk + ".attn.q"
        attn_k = blk + ".attn.k"
        attn_v = blk + ".attn.v"
        attn_out_proj = blk + ".attn.out_proj"
        ls1_path = blk + ".ls_1"
        ls2_path = blk + ".ls_2"
        mlp_c_fc = blk + ".mlp.c_fc"
        mlp_c_proj = blk + ".mlp.c_proj"

        residual = hidden
        hidden_shape = [batch, num_tokens, self.embed_dim]

        # --- Attention branch ---
        norm1_op = self.layer_norm(vit_mlir, hidden, ln1_path, eps=self.layer_norm_eps)

        # Q, K, V projections (all BF16, separate after QKV split)
        q_op = self.linear(vit_mlir,
                           attn_q,
                           norm1_op, [self.embed_dim, self.embed_dim],
                           [batch, num_tokens, self.embed_dim],
                           force_bias=True)
        k_op = self.linear(vit_mlir,
                           attn_k,
                           norm1_op, [self.embed_dim, self.embed_dim],
                           [batch, num_tokens, self.embed_dim],
                           force_bias=True)
        v_op = self.linear(vit_mlir,
                           attn_v,
                           norm1_op, [self.embed_dim, self.embed_dim],
                           [batch, num_tokens, self.embed_dim],
                           force_bias=True)

        # Reshape to [batch, num_tokens, num_heads, head_dim]
        qk_shape = [batch, num_tokens, self.vnum_heads, self.vhead_dim]
        q_op = top.ReshapeOp(T(qk_shape),
                             q_op,
                             shape=[batch, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(attn_q + ".reshape"),
                             ip=ip).output
        k_op = top.ReshapeOp(T(qk_shape),
                             k_op,
                             shape=[batch, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(attn_k + ".reshape"),
                             ip=ip).output
        v_op = top.ReshapeOp(T(qk_shape),
                             v_op,
                             shape=[batch, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(attn_v + ".reshape"),
                             ip=ip).output

        # 2D RoPE on Q, K.
        # HF vision_encoder.rotate_half uses adjacent-pair rotation
        # (rearrange "... (d r) -> ... d r", r=2), which corresponds to
        # RopeOp's "interleaved_pairs" mode — NOT "contiguous_halves".
        # The cos/sin table (compute_2d_rope_cos_sin) is laid out as the
        # full-dim interleaved [f0,f0,f1,f1,...], matching this mode.
        q_op = top.RopeOp(T(qk_shape),
                          q_op,
                          sin_op,
                          cos_op,
                          force_f32=True,
                          rope_mode=StringAttr.get("interleaved_pairs"),
                          loc=L(attn_q + ".rope"),
                          ip=ip).output
        k_op = top.RopeOp(T(qk_shape),
                          k_op,
                          sin_op,
                          cos_op,
                          force_f32=True,
                          rope_mode=StringAttr.get("interleaved_pairs"),
                          loc=L(attn_k + ".rope"),
                          ip=ip).output

        # Multi-head self-attention (non-causal)
        attn_out = top.FAttentionOp(T(qk_shape),
                                    q_op,
                                    k_op,
                                    v_op,
                                    vit_mlir.none_op,
                                    vit_mlir.none_op,
                                    scale=self.vhead_dim**-0.5,
                                    batch=batch,
                                    q_head=self.vnum_heads,
                                    kv_head=self.vnum_heads,
                                    dim=self.vhead_dim,
                                    mq=num_tokens,
                                    mk=num_tokens,
                                    keep_dims=True,
                                    loc=L(blk + ".fattention"),
                                    ip=ip).output

        # Reshape back: [batch, num_tokens, num_heads, head_dim] → [batch, num_tokens, embed_dim]
        attn_out = top.ReshapeOp(T([batch, num_tokens, self.embed_dim]),
                                 attn_out,
                                 shape=[batch, -1, self.embed_dim],
                                 loc=L(blk + ".fattention.reshape"),
                                 ip=ip).output

        # Out projection
        attn_result = self.linear(vit_mlir,
                                  attn_out_proj,
                                  attn_out, [self.embed_dim, self.embed_dim],
                                  [batch, num_tokens, self.embed_dim],
                                  force_bias=True)

        # LayerScale: gamma * attn_result
        gamma1_w = vit_mlir.create_weight_op(ls1_path + ".gamma", [1, 1, self.embed_dim])
        scaled_attn = top.MulOp(T([batch, num_tokens, self.embed_dim]), [attn_result, gamma1_w],
                                loc=L(ls1_path + ".mul"),
                                ip=ip).output

        # Residual connection
        hidden = top.AddOp(T([batch, num_tokens, self.embed_dim]), [residual, scaled_attn],
                           loc=L(blk + ".attn_residual"),
                           ip=ip).output

        # --- MLP branch ---
        residual = hidden

        norm2_op = self.layer_norm(vit_mlir, hidden, ln2_path, eps=self.layer_norm_eps)

        # c_fc: embed_dim → intermediate_size (may be AWQ for blocks 1-46)
        fc1_op = self.linear(vit_mlir,
                             mlp_c_fc,
                             norm2_op, [self.embed_dim, self.vintermediate_size],
                             [batch, num_tokens, self.vintermediate_size],
                             force_bias=True)

        # QuickGELU activation
        act_op = self.activate(vit_mlir, fc1_op, self.vit_hidden_act, mlp_c_fc)

        # c_proj: intermediate_size → embed_dim (may be AWQ)
        fc2_op = self.linear(vit_mlir,
                             mlp_c_proj,
                             act_op, [self.vintermediate_size, self.embed_dim],
                             [batch, num_tokens, self.embed_dim],
                             force_bias=True)

        # LayerScale
        gamma2_w = vit_mlir.create_weight_op(ls2_path + ".gamma", [1, 1, self.embed_dim])
        scaled_mlp = top.MulOp(T([batch, num_tokens, self.embed_dim]), [fc2_op, gamma2_w],
                               loc=L(ls2_path + ".mul"),
                               ip=ip).output

        # Residual connection
        hidden = top.AddOp(T([batch, num_tokens, self.embed_dim]), [residual, scaled_mlp],
                           loc=L(blk + ".mlp_residual"),
                           ip=ip).output

        return hidden

    def gen_vit_global_mlir(self):
        tqdm.write("generate vit_global mlir ...")
        self._build_vit_mlir(is_global=True)

    def gen_vit_patch_mlir(self):
        tqdm.write("generate vit_patch mlir ...")
        self._build_vit_mlir(is_global=False)

    # =========================================================================
    # ViT compilation
    # =========================================================================

    def compile_vit_global_mlir(self):
        name = "vit_global"
        if self.register_bmodel(name):
            return
        vit_q = self.vit_quantize or self.half_precision_quantize
        self.submit_deploy_task(
            name,
            [f'--quantize {vit_q}', '--quant_output'],
            dynamic=False,
            symmetric=self.symmetric,
        )

    def compile_vit_patch_mlir(self):
        name = "vit_patch"
        if self.register_bmodel(name):
            return
        vit_q = self.vit_quantize or self.half_precision_quantize
        self.submit_deploy_task(
            name,
            [f'--quantize {vit_q}', '--quant_output'],
            dynamic=True,
            symmetric=self.symmetric,
        )
