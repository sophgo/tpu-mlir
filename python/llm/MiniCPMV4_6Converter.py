# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .Qwen3_5Converter import *
from mlir.ir import StringAttr


class MiniCPMV4_6Converter(Qwen3_5Converter):

    def __init__(self, args, config, loader=None):
        # super().__init__() calls self.init_vconfig() (overridden below)
        super().__init__(args, config, loader=loader)
        # Patch grid dimensions (from max_shape, used by gen_vit_mlir)
        self.max_shape = args.max_shape
        self.patch_grid_h = self.max_shape[0] // self.patch_size
        self.patch_grid_w = self.max_shape[1] // self.patch_size

    @override
    def init_vconfig(self):
        """Override: MiniCPM-V-4.6 vision_config differs from Qwen3.5."""
        vc = self.config.vision_config
        self.patch_size = vc.patch_size  # 14
        self.embed_dim = vc.hidden_size  # 1152
        self.vnum_heads = vc.num_attention_heads  # 16
        self.vhead_dim = self.embed_dim // self.vnum_heads  # 72
        self.vintermediate_size = vc.intermediate_size  # 4304
        self.vit_depth = vc.num_hidden_layers  # 27
        self.vit_ln_eps = vc.layer_norm_eps  # 1e-6
        self.in_channels = vc.num_channels  # 3
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size  # 588

        # Always compile both 4x and 16x; runtime selects which to use.
        # Override compile_vit_4x/compile_vit_16x to False to skip a mode.
        self.compile_vit_4x = True
        self.compile_vit_16x = True
        self.pixel_multiple = self.patch_size * (2 if not self.compile_vit_16x else 4)
        self.vit_path = "model.vision_tower"

        # ViT Merger config
        self.insert_layer_id = vc.insert_layer_id  # 6
        self.window_kernel_size = tuple(vc.window_kernel_size)  # (2, 2)

        # Top-level merger config
        self.merge_kernel_size = tuple(self.config.merge_kernel_size)  # (2, 2)
        self.merger_times = self.config.merger_times  # 1

        # Image metadata
        self.image_token_id = self.config.image_token_id  # 248056
        self.video_token_id = self.config.video_token_id  # 248057

        # Text model uses MRoPE with partial_rotary_factor=0.25
        # mrope_section defaults to [11, 11, 10] even when not in config
        self.mrope_section = [11, 11, 10]
        self.position_shape = [3, self.max_input_length]

    # ======================== gen_vit_mlir (dispatcher) ========================

    @override
    def gen_vit_mlir(self):
        self._save_vit_weights()
        if self.compile_vit_4x:
            self._build_vit("4x")
        if self.compile_vit_16x:
            self._build_vit("16x")

    # ======================== _save_vit_weights ========================

    def _save_vit_weights(self):
        """Save all ViT weights once. Called before building any MLIR."""
        tqdm.write("saving vit weights...")
        vit_npz = "vit_top_weights.npz"
        self._vit_npz = vit_npz
        vp = self.vit_path
        merger_prefix = "model.merger.mlp.0"
        h = self.patch_grid_h
        w = self.patch_grid_w

        weights_dict = dict()

        # Patch embedding: convert Conv2d weight [D, C, PS, PS] to MatMul weight [C*PS*PS, D]
        pe_w = self.model.read(f"{vp}.embeddings.patch_embedding.weight")
        D, C, PS_h, PS_w = pe_w.shape
        pe_w_matmul = pe_w.reshape(D, C * PS_h * PS_w).transpose(1, 0).copy()
        weights_dict[f"{vp}.embeddings.patch_embedding.weight"] = pe_w_matmul
        pe_b = self.model.read(f"{vp}.embeddings.patch_embedding.bias")
        weights_dict[f"{vp}.embeddings.patch_embedding.bias"] = pe_b.reshape(1, -1)

        # Position embedding
        pos_w = self.model.read(f"{vp}.embeddings.position_embedding.weight")
        self.num_position_embeddings = pos_w.shape[0]
        weights_dict[f"{vp}.embeddings.position_embedding.weight"] = pos_w

        # ViT encoder layers (0-26)
        for i in range(self.vit_depth):
            lp = f"{vp}.encoder.layers.{i}"
            self.set_common_weight(f"{lp}.layer_norm1", weights_dict)
            self.set_common_weight(f"{lp}.layer_norm2", weights_dict)
            self.set_linear_weight(f"{lp}.self_attn.q_proj", weights_dict)
            self.set_linear_weight(f"{lp}.self_attn.k_proj", weights_dict)
            self.set_linear_weight(f"{lp}.self_attn.v_proj", weights_dict)
            self.set_linear_weight(f"{lp}.self_attn.out_proj", weights_dict)
            self.set_linear_weight(f"{lp}.mlp.fc1", weights_dict)
            self.set_linear_weight(f"{lp}.mlp.fc2", weights_dict)

        # Post-layernorm
        self.set_common_weight(f"{vp}.post_layernorm", weights_dict)

        # Merger: DownsampleMLP (both modes)
        self.set_common_weight(f"{merger_prefix}.pre_norm", weights_dict)
        self.set_linear_weight(f"{merger_prefix}.linear_1", weights_dict)
        self.set_linear_weight(f"{merger_prefix}.linear_2", weights_dict)

        # ViT Window Attention Merger weights (16x mode only)
        if self.compile_vit_16x:
            vm = f"{vp}.vit_merger"
            self.set_common_weight(f"{vm}.layer_norm1", weights_dict)
            self.set_linear_weight(f"{vm}.self_attn.q_proj", weights_dict)
            self.set_linear_weight(f"{vm}.self_attn.k_proj", weights_dict)
            self.set_linear_weight(f"{vm}.self_attn.v_proj", weights_dict)
            self.set_linear_weight(f"{vm}.self_attn.out_proj", weights_dict)
            self.set_common_weight(f"{vm}.pre_norm", weights_dict)
            self.set_linear_weight(f"{vm}.linear_1", weights_dict)
            self.set_linear_weight(f"{vm}.linear_2", weights_dict)

        # Note: spatial indices (reorder_index, window_index, reverse_index)
        # are now runtime inputs, not weights

        self.weight_keys.extend(weights_dict.keys())
        np.savez(vit_npz, **weights_dict)

    # ======================== compile_vit ========================

    @override
    def compile_vit(self):
        if not self.do_vit:
            return
        vit_q = self.vit_quantize or self.half_precision_quantize
        for mode in ["4x", "16x"]:
            if not getattr(self, f"compile_vit_{mode}"):
                continue
            name = f"vit_{mode}"
            if self.register_bmodel(name):
                continue
            extra_args = [f'--quantize {vit_q}', '--quant_output']
            self.submit_deploy_task(name, extra_args, dynamic=True)

    # ======================== _build_vit (shared builder) ========================

    def _build_vit(self, mode):
        tqdm.write(f"generate vit mlir (MiniCPM-V-4.6, {mode} mode)...")
        name = f"vit_{mode}"
        patches = self.num_patches
        h = self.patch_grid_h
        w = self.patch_grid_w
        D = self.embed_dim
        D_text = self.hidden_size
        n_heads = self.vnum_heads
        d_head = self.vhead_dim
        D_ff = self.vintermediate_size

        vp = self.vit_path
        merger_prefix = "model.merger.mlp.0"

        # Mode-dependent: output token count and grid after ViT merger
        if mode == "4x":
            post_vit_patches = patches  # no ViT merger
            post_vit_h = h
            post_vit_w = w
        else:  # 16x
            post_vit_patches = patches // 4  # ViT merger: 2×2 reduce
            post_vit_h = h // 2
            post_vit_w = w // 2

        merger_N_out = (post_vit_h // 2) * (post_vit_w // 2)

        # ======================== MLIR Module ========================

        # Input shapes: shared inputs first, then 16x-specific inputs
        in_shape = [1, self.in_channels, self.patch_size, patches * self.patch_size]
        pos_ids_shape = [patches]
        reorder_idx_shape = [post_vit_patches]

        # Shared inputs (both modes): pixel_values, pos_ids, reorder_index
        in_shapes = [in_shape, pos_ids_shape, reorder_idx_shape]
        in_types = ['F32', 'INT32', 'INT32']

        # 16x mode adds: window_index, reverse_index
        if mode == "16x":
            window_idx_shape = [patches]
            reverse_idx_shape = [patches]
            in_shapes.extend([window_idx_shape, reverse_idx_shape])
            in_types.extend(['INT32', 'INT32'])

        # Output shape: final merger output
        merged_d = D * 4
        out_shapes = [
            [merger_N_out, D_text],
        ]

        vit_mlir = MLIRImporter(in_shapes,
                                out_shapes,
                                name,
                                self.platform,
                                in_types,
                                weight_file=f"../{self._vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda n: self.get_loc(n, vit_mlir)

        # ======================== Vision Embedding ========================

        def vision_embedding(pixel_op, pos_ids_op):
            # Patch embedding via im2col + MatMul (replaces Conv2d for dynamic shape support)
            C = self.in_channels  # 3
            PS = self.patch_size  # 14
            P_flat = C * PS * PS  # 3 * 14 * 14 = 588

            # 1. Reshape [1, C, PS, P*PS] -> [1, C, PS, P, PS]
            reshape1 = top.ReshapeOp(T([1, C, PS, patches, PS]),
                                     pixel_op,
                                     shape=[1, C, PS, -1, PS],
                                     loc=L(f"{vp}.embeddings.reshape_split"),
                                     ip=ip).output

            # 2. Permute [1, C, PS, P, PS] -> [1, P, C, PS, PS]
            permute1 = top.PermuteOp(T([1, patches, C, PS, PS]),
                                     reshape1,
                                     order=[0, 3, 1, 2, 4],
                                     loc=L(f"{vp}.embeddings.permute"),
                                     ip=ip).output

            # 3. Reshape [1, P, C, PS, PS] -> [P, C*PS*PS]
            reshape2 = top.ReshapeOp(T([patches, P_flat]),
                                     permute1,
                                     shape=[-1, P_flat],
                                     loc=L(f"{vp}.embeddings.reshape_flatten"),
                                     ip=ip).output

            # 4. MatMul [P, C*PS*PS] @ [C*PS*PS, D] + bias [1, D] -> [P, D]
            matmul_w = vit_mlir.create_weight_op(f"{vp}.embeddings.patch_embedding.weight",
                                                 [P_flat, D])
            matmul_b = vit_mlir.create_weight_op(f"{vp}.embeddings.patch_embedding.bias", [1, D])
            perm_out = top.MatMulOp(T([patches, D]),
                                    reshape2,
                                    matmul_w,
                                    matmul_b,
                                    loc=L(f"{vp}.embeddings.patch_embedding"),
                                    ip=ip).output

            pos_weight = vit_mlir.create_weight_op(f"{vp}.embeddings.position_embedding.weight",
                                                   [self.num_position_embeddings, D])
            # Gather: [num_pos_embeddings, D] + [patches] → [patches, D]
            pos_emb = top.GatherOp(T([patches, D]),
                                   pos_weight,
                                   pos_ids_op,
                                   axis=0,
                                   loc=L(f"{vp}.embeddings.position_embedding"),
                                   ip=ip).output

            # [patches, D] + [patches, D] → [patches, D]
            out = top.AddOp(T([patches, D]), [perm_out, pos_emb],
                            loc=L(f"{vp}.embeddings.add"),
                            ip=ip).output
            return out

        # ======================== Vision Block ========================

        def vision_block(idx, in_op, num_p):
            """Single encoder layer. num_p = current number of patches."""
            lp = f"{vp}.encoder.layers.{idx}"
            attn_p = f"{lp}.self_attn"
            mlp_p = f"{lp}.mlp"
            hidden_shape = [num_p, D]
            qkv_shape = [1, num_p, n_heads, d_head]
            proj_shape = [D, D]
            # Attention
            norm1_op = self.layer_norm(vit_mlir, in_op, f"{lp}.layer_norm1", eps=self.vit_ln_eps)

            q_op = self.linear(vit_mlir,
                               f"{attn_p}.q_proj",
                               norm1_op,
                               proj_shape,
                               hidden_shape,
                               force_bias=True)

            k_op = self.linear(vit_mlir,
                               f"{attn_p}.k_proj",
                               norm1_op,
                               proj_shape,
                               hidden_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               f"{attn_p}.v_proj",
                               norm1_op,
                               proj_shape,
                               hidden_shape,
                               force_bias=True)

            # 2D → 4D for FAttention
            q_op = top.ReshapeOp(T(qkv_shape),
                                 q_op,
                                 shape=[1, -1, n_heads, d_head],
                                 loc=L(f"{attn_p}.q.reshape"),
                                 ip=ip).output

            k_op = top.ReshapeOp(T(qkv_shape),
                                 k_op,
                                 shape=[1, -1, n_heads, d_head],
                                 loc=L(f"{attn_p}.k.reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(qkv_shape),
                                 v_op,
                                 shape=[1, -1, n_heads, d_head],
                                 loc=L(f"{attn_p}.v.reshape"),
                                 ip=ip).output

            fa_op = top.FAttentionOp(T(qkv_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     scale=d_head**-0.5,
                                     batch=1,
                                     q_head=n_heads,
                                     kv_head=n_heads,
                                     dim=d_head,
                                     mq=num_p,
                                     mk=num_p,
                                     keep_dims=True,
                                     loc=L(f"{lp}.fattention"),
                                     ip=ip).output

            # 4D → 2D
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  shape=[-1, D],
                                  loc=L(f"{lp}.fattention.reshape"),
                                  ip=ip).output

            out_op = self.linear(vit_mlir,
                                 f"{attn_p}.out_proj",
                                 fa_op,
                                 proj_shape,
                                 hidden_shape,
                                 force_bias=True)

            attn_out = top.AddOp(T(hidden_shape), [in_op, out_op],
                                 loc=L(f"{attn_p}.out.add"),
                                 ip=ip).output

            # MLP
            norm2_op = self.layer_norm(vit_mlir, attn_out, f"{lp}.layer_norm2", eps=self.vit_ln_eps)
            fc1_op = self.linear(vit_mlir,
                                 f"{mlp_p}.fc1",
                                 norm2_op, [D, D_ff], [num_p, D_ff],
                                 force_bias=True)
            act_op = self.activate(vit_mlir, fc1_op, ActType.GELU_PYTORCH_TANH, f"{mlp_p}.fc1")
            fc2_op = self.linear(vit_mlir,
                                 f"{mlp_p}.fc2",
                                 act_op, [D_ff, D], [num_p, D],
                                 force_bias=True)
            mlp_out = top.AddOp(T(hidden_shape), [attn_out, fc2_op],
                                loc=L(f"{mlp_p}.fc2.add"),
                                ip=ip).output
            return mlp_out

        # ======================== ViT Window Merger (16x only) ========================

        def vit_window_merger(in_op, win_idx_op, rev_idx_op):
            """Window attention merger: N tokens → N/4 tokens."""
            vm = f"{vp}.vit_merger"
            N = patches
            N_win = N // 4  # after 2×2 window merge
            h_win = h // 2
            w_win = w // 2
            win_d = D * 4
            D_win_ff = self.vintermediate_size * 4  # 4304 * 4 = 17216
            n_windows = N_win  # number of 2×2 windows

            # 1. LayerNorm
            ln_op = self.layer_norm(vit_mlir, in_op, f"{vm}.layer_norm1", eps=self.vit_ln_eps)

            # 2. Reorder to window order: [N, D] gather axis=0
            reordered = top.GatherOp(T([N, D]),
                                     ln_op,
                                     win_idx_op,
                                     axis=0,
                                     loc=L(f"{vm}.gather_window"),
                                     ip=ip).output

            # 3. Reshape for batch-isolated window attention: [N,D] → [N_win,4,D]
            batch_input = top.ReshapeOp(
                T([n_windows, 4, D]),
                reordered,
                shape=[-1, 4, D],  # dynamic n_windows
                loc=L(f"{vm}.reshape_batch"),
                ip=ip).output

            # 4. Window self-attention (batch=N_win, each window = 4 tokens)
            win_hidden = [n_windows, 4, D]
            win_qkv = [n_windows, 4, n_heads, d_head]
            proj = [D, D]

            q_op = self.linear(vit_mlir,
                               f"{vm}.self_attn.q_proj",
                               batch_input,
                               proj,
                               win_hidden,
                               force_bias=True)
            k_op = self.linear(vit_mlir,
                               f"{vm}.self_attn.k_proj",
                               batch_input,
                               proj,
                               win_hidden,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               f"{vm}.self_attn.v_proj",
                               batch_input,
                               proj,
                               win_hidden,
                               force_bias=True)

            q_op = top.ReshapeOp(T(win_qkv),
                                 q_op,
                                 shape=[-1, 4, n_heads, d_head],
                                 loc=L(f"{vm}.self_attn.q.reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(win_qkv),
                                 k_op,
                                 shape=[-1, 4, n_heads, d_head],
                                 loc=L(f"{vm}.self_attn.k.reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(win_qkv),
                                 v_op,
                                 shape=[-1, 4, n_heads, d_head],
                                 loc=L(f"{vm}.self_attn.v.reshape"),
                                 ip=ip).output

            attn_out = top.FAttentionOp(T(win_qkv),
                                        q_op,
                                        k_op,
                                        v_op,
                                        vit_mlir.none_op,
                                        vit_mlir.none_op,
                                        scale=d_head**-0.5,
                                        batch=n_windows,
                                        q_head=n_heads,
                                        kv_head=n_heads,
                                        dim=d_head,
                                        mq=4,
                                        mk=4,
                                        keep_dims=True,
                                        loc=L(f"{vm}.self_attn.fattention"),
                                        ip=ip).output

            attn_out = top.ReshapeOp(T(win_hidden),
                                     attn_out,
                                     shape=[-1, 4, D],
                                     loc=L(f"{vm}.self_attn.reshape"),
                                     ip=ip).output
            out_op = self.linear(vit_mlir,
                                 f"{vm}.self_attn.out_proj",
                                 attn_out,
                                 proj,
                                 win_hidden,
                                 force_bias=True)

            # 5. Reshape back: [N_win,4,D] → [N,D]
            out_flat = top.ReshapeOp(T([N, D]),
                                     out_op,
                                     shape=[-1, D],
                                     loc=L(f"{vm}.reshape_back"),
                                     ip=ip).output

            # 6. Reverse reorder: [N, D] gather axis=0
            restored = top.GatherOp(T([N, D]),
                                    out_flat,
                                    rev_idx_op,
                                    axis=0,
                                    loc=L(f"{vm}.gather_reverse"),
                                    ip=ip).output

            # 7. Residual add: [N, D] + [N, D]
            attn_result = top.AddOp(T([N, D]), [in_op, restored],
                                    loc=L(f"{vm}.residual.add"),
                                    ip=ip).output

            # 8. 2×2 spatial concat: [N,D] → [N_win, 4*D]
            gathered = top.GatherOp(T([N, D]),
                                    attn_result,
                                    win_idx_op,
                                    axis=0,
                                    loc=L(f"{vm}.gather_2x2"),
                                    ip=ip).output
            concat_out = top.ReshapeOp(T([N_win, win_d]),
                                       gathered,
                                       shape=[-1, win_d],
                                       loc=L(f"{vm}.reshape_2x2"),
                                       ip=ip).output

            # 9. mean_residual: [N,D] → reorder → [N_win,4,D] → mean → [N_win,D]
            mean_gathered = top.GatherOp(T([N, D]),
                                         attn_result,
                                         win_idx_op,
                                         axis=0,
                                         loc=L(f"{vm}.mean.gather"),
                                         ip=ip).output
            mean_reshaped = top.ReshapeOp(T([N_win, 4, D]),
                                          mean_gathered,
                                          shape=[-1, 4, D],
                                          loc=L(f"{vm}.mean.reshape"),
                                          ip=ip).output
            mean_residual = top.ReduceOp(T([N_win, D]),
                                         mean_reshaped,
                                         axes=[1],
                                         keepdims=0,
                                         mode=StringAttr.get("ReduceMean"),
                                         loc=L(f"{vm}.mean.reduce"),
                                         ip=ip).output

            # 10. MLP: pre_norm → linear_1 → GELU(tanh) → linear_2
            pre_ln = self.layer_norm(vit_mlir, concat_out, f"{vm}.pre_norm", eps=1e-6)
            fc1 = self.linear(vit_mlir,
                              f"{vm}.linear_1",
                              pre_ln, [win_d, D_win_ff], [N_win, D_win_ff],
                              force_bias=True)
            act = self.activate(vit_mlir, fc1, ActType.GELU_PYTORCH_TANH, f"{vm}.linear_1")
            fc2 = self.linear(vit_mlir,
                              f"{vm}.linear_2",
                              act, [D_win_ff, D], [N_win, D],
                              force_bias=True)

            # 11. Add mean_residual → return 2D
            result = top.AddOp(T([N_win, D]), [fc2, mean_residual],
                               loc=L(f"{vm}.mlp.add_residual"),
                               ip=ip).output
            return result

        # ======================== Merger inlined into Build Graph ========================

        # ======================== Build Graph ========================

        # Create input ops in order: pixel_values, pos_ids, reorder_index, [window_index, reverse_index]
        in0 = vit_mlir.create_input_op(L("pixel_values"), 0)
        in1 = vit_mlir.create_input_op(L("pos_ids"), 1)
        reorder_idx_op = vit_mlir.create_input_op(L("merger.reorder_index"), 2)

        if mode == "16x":
            window_idx_op = vit_mlir.create_input_op(L("vit_merger.window_index"), 3)
            reverse_idx_op = vit_mlir.create_input_op(L("vit_merger.reverse_index"), 4)

        new_op = vision_embedding(in0, in1)

        if mode == "4x":
            for i in range(self.vit_depth):
                new_op = vision_block(i, new_op, patches)
        else:  # 16x
            for i in range(self.insert_layer_id + 1):  # layers 0-6
                new_op = vision_block(i, new_op, patches)
            new_op = vit_window_merger(new_op, window_idx_op, reverse_idx_op)
            for i in range(self.insert_layer_id + 1, self.vit_depth):  # layers 7-26
                new_op = vision_block(i, new_op, post_vit_patches)

        # Post-layernorm
        new_op = self.layer_norm(vit_mlir, new_op, f"{vp}.post_layernorm", eps=self.vit_ln_eps)

        # Merger (DownsampleMLP)
        merged_d = D * 4
        mp = merger_prefix
        # Gather: [post_vit_patches, D] axis=0 → reorder → [merger_N_out, 4*D]
        gathered = top.GatherOp(T([post_vit_patches, D]),
                                new_op,
                                reorder_idx_op,
                                axis=0,
                                loc=L("merger.gather_2x2"),
                                ip=ip).output
        reshaped = top.ReshapeOp(T([merger_N_out, merged_d]),
                                 gathered,
                                 shape=[-1, merged_d],
                                 loc=L("merger.reshape_2x2"),
                                 ip=ip).output
        ln_op = self.layer_norm(vit_mlir, reshaped, f"{mp}.pre_norm", eps=1e-6)
        fc1_op = self.linear(vit_mlir,
                             f"{mp}.linear_1",
                             ln_op, [merged_d, merged_d], [merger_N_out, merged_d],
                             force_bias=True)
        act_op = self.activate(vit_mlir, fc1_op, ActType.GELU, f"{mp}.linear_1")
        fc2_op = self.linear(vit_mlir,
                             f"{mp}.linear_2",
                             act_op, [merged_d, D_text], [merger_N_out, D_text],
                             force_bias=True)

        vit_mlir.create_return_op([fc2_op])
        self.save_mlir_module(vit_mlir, name)
