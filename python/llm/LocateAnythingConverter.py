# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from .LlmInfo import JANUS_INFO
from typing_extensions import override


class LocateAnythingConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        self.max_pixels = args.max_pixels
        self.max_shape = args.max_shape
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        super().__init__(args, config, loader=loader)
        self.do_vit = True
        self.init_vconfig()
        self.vit_path = "vision_model"

    def init_vconfig(self):
        """Set MoonViT vision parameters."""
        vconfig = self.config.vision_config
        self.patch_size = vconfig.patch_size
        self.embed_dim = vconfig.hidden_size
        self.vnum_heads = vconfig.num_attention_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vit_depth = vconfig.num_hidden_layers
        self.vintermediate_size = vconfig.intermediate_size
        self.merge_kernel_size = list(vconfig.merge_kernel_size)
        self.spatial_merge_size = self.merge_kernel_size[0]
        self.init_pos_emb_h = vconfig.init_pos_emb_height
        self.init_pos_emb_w = vconfig.init_pos_emb_width
        self.max_num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        # MLP1 projector dimensions
        self.merged_dim = self.embed_dim * (self.spatial_merge_size**2)
        self.mlp1_hidden = self.hidden_size
        # Number of output tokens after merger
        self.num_merged_tokens = self.max_num_patches // (self.spatial_merge_size**2)

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_type = LlmType.QWEN2
        self.model_info = JANUS_INFO

    @override
    def compile_vit(self):
        """Always compile the ViT bmodel as dynamic.

        MoonViT handles variable image sizes at runtime (variable patch
        count), so the ViT must be dynamic regardless of the global
        ``--dynamic`` flag. Text blocks keep the standard rule: static by
        default, dynamic when ``--dynamic`` is passed.
        """
        if not self.do_vit:
            return
        name = "vit"
        if self.register_bmodel(name):
            return
        vit_q = self.vit_quantize or self.half_precision_quantize
        if self.half_precision_quantize == 'bf16' and self.vit_f16_out_bf16 and vit_q in ('f16',
                                                                                          'bf16'):
            extra_args = ['--quantize f16', '--quant_output_bf16']
        else:
            extra_args = [f'--quantize {vit_q}', '--quant_output']
        vit_info = getattr(self, 'vit_quant_info', None)
        if vit_info and vit_info.quant_bits != 16:
            extra_args.append(f'--q_group_size {vit_info.q_group_size}')
        elif isinstance(self.loader,
                        GGUFModelHandle) and not getattr(self, 'vit_gguf_float', False):
            extra_args.append(f'--q_group_size {self.q_group_size}')
        elif self.quant_mode is not None:
            extra_args.append(f'--q_group_size {self.q_group_size}')
        extra_args.append('--dynamic')
        self.submit_deploy_task(name, extra_args, symmetric=self._detect_vit_symmetric())

    def vision_rotary(self):
        """Precompute 2D RoPE cos/sin for MoonViT.

        MoonViT uses 2D RoPE where:
        - First half of head_dim (dim//4 pairs) uses h-direction frequencies
        - Second half of head_dim (dim//4 pairs) uses w-direction frequencies

        This layout is compatible with the standard rotary_pos() which uses
        contiguous halves (first_half, second_half).

        Returns:
            cos, sin: numpy arrays of shape [num_patches, dim//2]
                      Pair-wise: first dim//4 pairs use w-direction, next dim//4 use h-direction
        """
        dim = self.vhead_dim
        assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        grid_h = self.max_shape[0] // self.patch_size
        grid_w = self.max_shape[1] // self.patch_size
        theta_base = 10000.0
        half_dim = dim // 2  # 36 pairs

        # Frequency range: dim//4 for each spatial direction
        dim_range = np.arange(0, dim, 4, dtype=np.float32)[:dim // 4]
        freqs = 1.0 / (theta_base**(dim_range / dim))

        # H direction
        h_pos = np.arange(grid_h, dtype=np.float32)
        h_freqs = np.outer(h_pos, freqs)
        h_cos = np.cos(h_freqs)
        h_sin = np.sin(h_freqs)

        # W direction
        w_pos = np.arange(grid_w, dtype=np.float32)
        w_freqs = np.outer(w_pos, freqs)
        w_cos = np.cos(w_freqs)
        w_sin = np.sin(w_freqs)

        quarter = dim // 4  # 18
        # Build pair-wise cos/sin: [grid_h, grid_w, half_dim]
        # HF freqs_cis interleaves x/y: [w_cos0, h_cos0, w_cos1, h_cos1, ...]
        cos = np.zeros((grid_h, grid_w, half_dim), dtype=np.float32)
        sin = np.zeros((grid_h, grid_w, half_dim), dtype=np.float32)

        for i in range(quarter):
            # Even pairs: w-direction (x in HF), broadcast along h-axis
            cos[:, :, 2 * i] = w_cos[None, :, i]
            sin[:, :, 2 * i] = w_sin[None, :, i]
            # Odd pairs: h-direction (y in HF), broadcast along w-axis
            cos[:, :, 2 * i + 1] = h_cos[:, i:i + 1]
            sin[:, :, 2 * i + 1] = h_sin[:, i:i + 1]

        cos = cos.reshape(-1, half_dim)
        sin = sin.reshape(-1, half_dim)
        return cos, sin  # [num_patches, dim//2]

    def precompute_pos_emb(self):
        """Precompute interpolated 2D position embedding for the max grid size.

        The Learnable2DInterpPosEmb stores a [H, W, D] parameter and interpolates
        to the actual grid size using bicubic interpolation.

        Returns:
            numpy array of shape [num_patches, embed_dim]
        """
        pos_weight = self.model.read(f"{self.vit_path}.patch_embed.pos_emb.weight")
        h = self.max_shape[0] // self.patch_size
        w = self.max_shape[1] // self.patch_size

        import torch
        import torch.nn.functional as F
        pos_tensor = torch.from_numpy(pos_weight)

        if (h, w) != (pos_tensor.shape[0], pos_tensor.shape[1]):
            pos_tensor = F.interpolate(
                pos_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode='bicubic',
            ).squeeze(0).permute(1, 2, 0)

        return pos_tensor.flatten(end_dim=1).numpy()

    def compute_merger_index(self):
        """Compute 2×2 spatial merge reorder index.

        The merger groups every 2×2 spatial patch into one token:
        For grid (h, w), output position (i, j) gathers:
        - (2i)*w + 2j, (2i)*w + (2j+1), (2i+1)*w + 2j, (2i+1)*w + (2j+1)

        Returns:
            numpy array of shape [num_merged_tokens * 4], dtype float32
        """
        h = self.max_shape[0] // self.patch_size
        w = self.max_shape[1] // self.patch_size
        new_h = h // self.merge_kernel_size[0]
        new_w = w // self.merge_kernel_size[1]

        indices = np.zeros(new_h * new_w * 4, dtype=np.float32)
        for i in range(new_h):
            for j in range(new_w):
                out_pos = i * new_w + j
                base = out_pos * 4
                indices[base] = (2 * i) * w + 2 * j
                indices[base + 1] = (2 * i) * w + 2 * j + 1
                indices[base + 2] = (2 * i + 1) * w + 2 * j
                indices[base + 3] = (2 * i + 1) * w + 2 * j + 1

        return indices

    def pairwise_rope(self, mlir_gen, q_op, cos_op, sin_op, name):
        """Apply pair-wise 2D RoPE via built-in RopeOp(interleaved_pairs).

        HF apply_rope pairs (q[2k], q[2k+1]) as complex and rotates by
        freqs_cis[k] = cos_k + i*sin_k:
            result[2k]   = q[2k]*cos_k - q[2k+1]*sin_k
            result[2k+1] = q[2k]*sin_k + q[2k+1]*cos_k

        RopeOp(interleaved_pairs) computes:
            out[i] = temp[i]*w0[i] + q[i]*w1[i], where
            temp[2k]=-q[2k+1], temp[2k+1]=q[2k]
        => w0 (input2=sin) and w1 (input3=cos) must be full-dim with each
        pair value repeated: [sin0,sin0,sin1,sin1,...], [cos0,cos0,...].

        Args:
            q_op: [1, N, H, dim]
            cos_op: [1, N, 1, half_dim] (per-pair cos, runtime input reshaped)
            sin_op: [1, N, 1, half_dim] (per-pair sin)
        """
        ip = mlir_gen.insert_point
        T = mlir_gen.get_tensor_type
        L = lambda n: self.get_loc(n, mlir_gen)
        N = self.max_num_patches
        H = self.vnum_heads
        dim = self.vhead_dim
        half = dim // 2  # 36

        # Expand cos/sin [1,N,1,half] -> [1,N,1,dim] with per-element repeat
        # ([c0,c0,c1,c1,...]) via reshape->Tile->reshape. Uses only ReshapeOp
        # shape param for the dynamic dim; Tile on a static axis (size 1->2),
        # so it is safe for dynamic N.
        def expand_full(op, tag):
            r = top.ReshapeOp(T([1, N, 1, half, 1]),
                              op,
                              shape=[1, -1, 1, half, 1],
                              loc=L(name + "." + tag + ".r"),
                              ip=ip).output
            t = top.TileOp(T([1, N, 1, half, 2]),
                           r,
                           tile=[1, 1, 1, 1, 2],
                           loc=L(name + "." + tag + ".tile"),
                           ip=ip).output
            f = top.ReshapeOp(T([1, N, 1, dim]),
                              t,
                              shape=[1, -1, 1, dim],
                              loc=L(name + "." + tag + ".full"),
                              ip=ip).output
            return f

        sin_full = expand_full(sin_op, "sin")
        cos_full = expand_full(cos_op, "cos")

        # RopeOp(input1=q, input2=sin, input3=cos, interleaved_pairs)
        result = top.RopeOp(T([1, N, H, dim]),
                            q_op,
                            sin_full,
                            cos_full,
                            rope_mode=StringAttr.get("interleaved_pairs"),
                            loc=L(name + ".rope"),
                            ip=ip).output
        return result

    def vit_layer_norm(self, mlir_gen, in_op, norm_path: str, eps, name: str = ""):
        """ViT LayerNorm with per-position DC pre-subtraction (BM1684X only).

        BM1684X's bf16+dynamic LayerNorm kernel uses the unstable variance
        formula var = E[x^2] - E[x]^2, which suffers catastrophic cancellation
        when the input has a non-zero per-position mean (DC) — e.g. MoonViT
        patch embeddings. Subtracting the per-position mean beforehand makes
        the input zero-DC so the kernel's formula degenerates to E[x^2] (no
        cancellation). LN(x - mean(x)) == LN(x) mathematically, so the output
        is unchanged; only the numerical stability improves (0.7655 -> 0.9999).
        BM1688 lowers LayerNorm to F32 and is unaffected, so skip there.
        Reuses self.layer_norm for the weight/bias/LayerNorm creation.
        """
        if self.chip != "bm1684x":
            return self.layer_norm(mlir_gen, in_op, norm_path, eps, name)
        ip = mlir_gen.insert_point
        T = mlir_gen.get_tensor_type
        L = lambda n: self.get_loc(n, mlir_gen)
        input_shape = list(in_op.type.shape)
        axis = len(input_shape) - 1
        loc_name = name if name else norm_path
        mean_shape = [1 if i == axis else d for i, d in enumerate(input_shape)]
        mean_op = top.ReduceOp(T(mean_shape),
                               in_op,
                               axes=[axis],
                               keepdims=True,
                               mode=StringAttr.get("ReduceMean"),
                               loc=L(loc_name + ".dc_mean"),
                               ip=ip).output
        x0 = top.SubOp(T(input_shape), [in_op, mean_op], loc=L(loc_name + ".dc_sub"), ip=ip).output
        return self.layer_norm(mlir_gen, x0, norm_path, eps, name)

    def vision_block(self, vit_mlir, id, in_op, cos_op, sin_op):
        """Build one MoonViT encoder block: pre-norm attention + pre-norm MLP."""
        prefix = f"{self.vit_path}.encoder.blocks.{id}"
        norm0 = f"{prefix}.norm0"
        norm1 = f"{prefix}.norm1"
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        hidden_shape = [self.max_num_patches, self.embed_dim]

        # ===== Attention (pre-norm) =====
        residual = in_op
        norm_op = self.vit_layer_norm(vit_mlir, in_op, norm0, eps=1e-5)

        # Separate Q, K, V projections (packed QKV split in save_weights)
        q_path = f"{prefix}.q"
        k_path = f"{prefix}.k"
        v_path = f"{prefix}.v"
        qkv_shape = [self.embed_dim, self.embed_dim]
        qkv_out_shape = [self.max_num_patches, self.embed_dim]

        q_op = self.linear(vit_mlir, q_path, norm_op, qkv_shape, qkv_out_shape, force_bias=True)
        k_op = self.linear(vit_mlir, k_path, norm_op, qkv_shape, qkv_out_shape, force_bias=True)
        v_op = self.linear(vit_mlir, v_path, norm_op, qkv_shape, qkv_out_shape, force_bias=True)

        # Reshape to 4D [1, seq, heads, head_dim]
        N = self.max_num_patches
        q_4d = top.ReshapeOp(T([1, N, self.vnum_heads, self.vhead_dim]),
                             q_op,
                             shape=[1, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(q_path + ".4d"),
                             ip=ip).output
        k_4d = top.ReshapeOp(T([1, N, self.vnum_heads, self.vhead_dim]),
                             k_op,
                             shape=[1, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(k_path + ".4d"),
                             ip=ip).output
        v_4d = top.ReshapeOp(T([1, N, self.vnum_heads, self.vhead_dim]),
                             v_op,
                             shape=[1, -1, self.vnum_heads, self.vhead_dim],
                             loc=L(v_path + ".4d"),
                             ip=ip).output

        # Apply pair-wise 2D RoPE to Q and K
        q_4d = self.pairwise_rope(vit_mlir, q_4d, cos_op, sin_op, q_path + ".rope")
        k_4d = self.pairwise_rope(vit_mlir, k_4d, cos_op, sin_op, k_path + ".rope")

        # Full attention (non-causal, MHA)
        fa_op = top.FAttentionOp(T([1, N, self.vnum_heads, self.vhead_dim]),
                                 q_4d,
                                 k_4d,
                                 v_4d,
                                 vit_mlir.none_op,
                                 vit_mlir.none_op,
                                 scale=self.vhead_dim**-0.5,
                                 batch=1,
                                 q_head=self.vnum_heads,
                                 kv_head=self.vnum_heads,
                                 dim=self.vhead_dim,
                                 mq=N,
                                 mk=N,
                                 keep_dims=True,
                                 loc=L(f"{prefix}.fattention"),
                                 ip=ip).output

        # Reshape back and project output
        fa_flat = top.ReshapeOp(T([N, self.embed_dim]),
                                fa_op,
                                shape=[-1, self.embed_dim],
                                loc=L(f"{prefix}.fattention.reshape"),
                                ip=ip).output

        wo_path = f"{prefix}.wo"
        out_op = self.linear(vit_mlir,
                             wo_path,
                             fa_flat, [self.embed_dim, self.embed_dim],
                             hidden_shape,
                             force_bias=True)

        # Residual connection
        attn_out = top.AddOp(T(hidden_shape), [residual, out_op], loc=L(wo_path + ".add"),
                             ip=ip).output

        # ===== MLP (pre-norm) =====
        residual = attn_out
        norm_op = self.vit_layer_norm(vit_mlir, attn_out, norm1, eps=1e-5)

        fc0_path = f"{prefix}.mlp.fc0"
        fc1_path = f"{prefix}.mlp.fc1"

        fc0_op = self.linear(vit_mlir,
                             fc0_path,
                             norm_op, [self.embed_dim, self.vintermediate_size],
                             [N, self.vintermediate_size],
                             force_bias=True)
        # GELU(tanh) activation
        act_op = self.activate(vit_mlir, fc0_op, ActType.GELU_PYTORCH_TANH, fc0_path)
        fc1_op = self.linear(vit_mlir,
                             fc1_path,
                             act_op, [self.vintermediate_size, self.embed_dim],
                             hidden_shape,
                             force_bias=True)

        # Residual connection
        mlp_out = top.AddOp(T(hidden_shape), [residual, fc1_op], loc=L(fc1_path + ".add"),
                            ip=ip).output

        return mlp_out

    @override
    def gen_vit_mlir(self):
        tqdm.write("generate MoonViT + MLP1 mlir ...")
        name = "vit"
        vit_npz = "vit_top_weights.npz"

        patch_embed_path = f"{self.vit_path}.patch_embed.proj"
        rotary_cos_path = f"{self.vit_path}.rotary.cos"
        rotary_sin_path = f"{self.vit_path}.rotary.sin"
        merger_ln_path = f"{self.vit_path}.encoder.final_layernorm"
        mlp1_ln_path = "mlp1.0"
        mlp1_fc1_path = "mlp1.1"
        mlp1_fc2_path = "mlp1.3"

        def save_weights():
            weights_dict = {}

            # Export pos_emb weight to config directory for pipeline runtime interpolation
            pos_emb_path = f"{self.vit_path}.patch_embed.pos_emb"
            pos_emb = self.model.read(pos_emb_path + ".weight")  # [init_h, init_w, embed_dim]
            config_dir = getattr(self, 'config_dir', None)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                np.savez(os.path.join(config_dir, "vit_pos_emb.npz"),
                         pos_emb=pos_emb.astype(np.float32))

            # Patch embedding: Conv2d weight [out, in, kH, kW] -> MatMul weight [in*kH*kW, out]
            data = self.model.read(patch_embed_path + ".weight")
            data = data.reshape(self.embed_dim, -1)
            weights_dict[patch_embed_path + ".weight"] = np.ascontiguousarray(
                np.transpose(data, (1, 0)))
            # Patch embedding bias (don't use set_common_weight — it would overwrite the reshaped weight)
            weights_dict[patch_embed_path + ".bias"] = self.model.read(patch_embed_path + ".bias")

            # Merger MLP1 weights
            self.set_common_weight(merger_ln_path, weights_dict)
            self.set_common_weight(mlp1_ln_path, weights_dict)
            self.set_linear_weight(mlp1_fc1_path, weights_dict)
            self.set_linear_weight(mlp1_fc2_path, weights_dict)

            # Encoder blocks
            for i in range(self.vit_depth):
                bp = f"{self.vit_path}.encoder.blocks.{i}"
                # LayerNorm (weight + bias)
                self.set_common_weight(f"{bp}.norm0", weights_dict)
                self.set_common_weight(f"{bp}.norm1", weights_dict)
                # Packed QKV: split into separate Q, K, V
                qkv_w = self.model.read(f"{bp}.wqkv.weight").reshape(3 * self.embed_dim,
                                                                     self.embed_dim)
                qkv_b = self.model.read(f"{bp}.wqkv.bias").reshape(3 * self.embed_dim)
                q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
                q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
                weights_dict[f"{bp}.q.weight"] = np.ascontiguousarray(np.transpose(q_w, (1, 0)))
                weights_dict[f"{bp}.k.weight"] = np.ascontiguousarray(np.transpose(k_w, (1, 0)))
                weights_dict[f"{bp}.v.weight"] = np.ascontiguousarray(np.transpose(v_w, (1, 0)))
                weights_dict[f"{bp}.q.bias"] = q_b
                weights_dict[f"{bp}.k.bias"] = k_b
                weights_dict[f"{bp}.v.bias"] = v_b
                # Output projection
                self.set_linear_weight(f"{bp}.wo", weights_dict)
                # MLP
                self.set_linear_weight(f"{bp}.mlp.fc0", weights_dict)
                self.set_linear_weight(f"{bp}.mlp.fc1", weights_dict)

            np.savez(vit_npz, **weights_dict)

        # === Build MLIR graph ===
        patch_dim = 3 * self.patch_size * self.patch_size

        # pos_emb and RoPE are runtime inputs for dynamic ViT
        in_shape = [self.max_num_patches, patch_dim]
        merger_idx_shape = [self.max_num_patches]
        pos_emb_in_shape = [self.max_num_patches, self.embed_dim]
        rope_in_shape = [self.max_num_patches, self.vhead_dim // 2]
        input_shapes = [in_shape, merger_idx_shape, pos_emb_in_shape, rope_in_shape, rope_in_shape]
        input_types = ['F32', 'INT32', 'F32', 'F32', 'F32']

        out_shapes = [[self.num_merged_tokens, self.hidden_size]]

        vit_mlir = MLIRImporter(input_shapes,
                                out_shapes,
                                name,
                                self.platform,
                                input_types,
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        # Inputs
        patches_op = vit_mlir.create_input_op(L('pixel_values'), 0)
        merger_idx_op = vit_mlir.create_input_op(L('merger_index'), 1)
        pos_emb_op = vit_mlir.create_input_op(L('pos_emb'), 2)
        rope_cos_flat = vit_mlir.create_input_op(L('rope_cos'), 3)
        rope_sin_flat = vit_mlir.create_input_op(L('rope_sin'), 4)

        # Patch embedding: MatMul with transposed conv weight
        # Input: [N, patch_dim] @ [patch_dim, embed_dim] -> [N, embed_dim]
        weight_op = vit_mlir.create_weight_op(patch_embed_path + ".weight",
                                              [patch_dim, self.embed_dim])
        hidden_op = top.MatMulOp(T([self.max_num_patches, self.embed_dim]),
                                 patches_op,
                                 weight_op,
                                 vit_mlir.none_op,
                                 loc=L(patch_embed_path),
                                 ip=ip).output

        # Add patch embedding bias
        bias_op = vit_mlir.create_weight_op(patch_embed_path + ".bias", [1, self.embed_dim])
        hidden_op = top.AddOp(T([self.max_num_patches, self.embed_dim]), [hidden_op, bias_op],
                              loc=L(patch_embed_path + ".bias_add"),
                              ip=ip).output

        # Add position embedding (runtime input for dynamic ViT)
        hidden_op = top.AddOp(T([self.max_num_patches, self.embed_dim]), [hidden_op, pos_emb_op],
                              loc=L("pos_emb_add"),
                              ip=ip).output

        # RoPE cos/sin: runtime inputs [N, half_dim] -> reshape to [1, N, 1, half_dim]
        N = self.max_num_patches
        half_dim = self.vhead_dim // 2
        cos_weight_op = top.ReshapeOp(T([1, N, 1, half_dim]),
                                      rope_cos_flat,
                                      shape=[1, -1, 1, half_dim],
                                      loc=L("rope_cos.reshape"),
                                      ip=ip).output
        sin_weight_op = top.ReshapeOp(T([1, N, 1, half_dim]),
                                      rope_sin_flat,
                                      shape=[1, -1, 1, half_dim],
                                      loc=L("rope_sin.reshape"),
                                      ip=ip).output

        # Encoder: 27 vision blocks
        for i in range(self.vit_depth):
            hidden_op = self.vision_block(vit_mlir, i, hidden_op, cos_weight_op, sin_weight_op)

        # Final LayerNorm
        hidden_op = self.vit_layer_norm(vit_mlir, hidden_op, merger_ln_path, eps=1e-5)

        # === Patch Merger (2×2 spatial merge) ===
        # GatherOp reorders patches into 2×2 groups:
        # [N, D] gathered by [N] indices -> [N, D]
        # Then Reshape to [N/4, 4*D]
        gathered_op = top.GatherOp(T([self.max_num_patches, self.embed_dim]),
                                   hidden_op,
                                   merger_idx_op,
                                   axis=0,
                                   loc=L("merger.gather"),
                                   ip=ip).output

        merged_dim = self.embed_dim * (self.spatial_merge_size**2)
        merged_op = top.ReshapeOp(T([self.num_merged_tokens, merged_dim]),
                                  gathered_op,
                                  shape=[-1, merged_dim],
                                  loc=L("merger.reshape"),
                                  ip=ip).output

        # === MLP1 Projector ===
        # LayerNorm(merged_dim) -> Linear(merged_dim, hidden_size) -> GELU -> Linear(hidden_size, hidden_size)
        merged_op = self.vit_layer_norm(vit_mlir, merged_op, mlp1_ln_path, eps=1e-5)
        merged_op = self.linear(vit_mlir,
                                mlp1_fc1_path,
                                merged_op, [merged_dim, self.mlp1_hidden],
                                [self.num_merged_tokens, self.mlp1_hidden],
                                force_bias=True)
        merged_op = self.activate(vit_mlir, merged_op, ActType.GELU, mlp1_fc1_path)
        merged_op = self.linear(vit_mlir,
                                mlp1_fc2_path,
                                merged_op, [self.mlp1_hidden, self.mlp1_hidden],
                                [self.num_merged_tokens, self.mlp1_hidden],
                                force_bias=True)

        vit_mlir.create_return_op([merged_op])
        save_weights()
        self.save_mlir_module(vit_mlir, name)
