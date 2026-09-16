# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
from .LlmInfo import LlmList, LlmType, WeightType, QWEN3_TTS_INFO

# ---------------------------------------------------------------------------
# Qwen3-TTS
#
# Qwen3-TTS is a text-to-speech model. This converter compiles the whole
# pipeline onto TPU as one bmodel:
#   * Speaker Encoder  (ECAPA-TDNN): mel [1, T, 128] -> spk_embed [1, 1024]
#   * Speech Tokenizer (Mimi, decode only): codec codes [1, 16, T] -> wav
#   * Talker LM        (28-layer Qwen3-style decoder)
#   * CodePredictor    (5-layer Qwen3-style decoder + 15 embed + 15 lm_head)
#
# Talker:
#   - 28 layers, GQA 16/8, head_dim 128, QK-norm, MRoPE interleaved [24,20,20]
#   - text_embedding (151936x2048) + text_projection (2-layer MLP 2048->2048->1024)
#   - codec_embedding (3072x1024) lookup, codec_head (3072x1024) = lm_head
#   - vocab_size (codec) = 3072; sample_head reused from base (vocab 3072)
#   - inputs_embeds is multi-source (text + codec + speaker + trailing), assembled
#     in the dev_buffer (Qwen3_5 pattern) so block nets read input_states.
#
# CodePredictor:
#   - 5 layers, GQA 16/8, head_dim 128, QK-norm, 1D RoPE (NOT MRoPE)
#   - 15 codec_embedding (2048x1024) + 15 lm_head (2048x1024), selected per
#     generation_step -> compiled as 15 independent embed nets + 15 lm_head nets
#   - small_to_mtp_projection is Identity (hidden_size matches) -> not compiled
#   - only block_cache (decode) net is needed; embed/lm_head are independent nets
# ============================================================================


class Qwen3TTSConverter(LlmConverter):

    def __init__(self, args, config, loader=None):
        # capture code-predictor config before super().__init__ (load_pretrained
        # is called inside super().__init__, and it reads talker_config).
        self._cp_config_src = getattr(config.talker_config, "code_predictor_config", None)
        super().__init__(args, config, loader=loader)

        # CodePredictor parameters
        cp = self._cp_config_src
        self.cp_num_layers = getattr(cp, "num_hidden_layers", 5)
        self.cp_hidden_size = getattr(cp, "hidden_size", 1024)
        self.cp_num_attention_heads = getattr(cp, "num_attention_heads", 16)
        self.cp_num_key_value_heads = getattr(cp, "num_key_value_heads", 8)
        self.cp_head_dim = getattr(cp, "head_dim",
                                   self.cp_hidden_size // self.cp_num_attention_heads)
        self.cp_intermediate_size = getattr(cp, "intermediate_size", 3072)
        self.cp_vocab_size = getattr(cp, "vocab_size", 2048)
        self.cp_rms_norm_eps = getattr(cp, "rms_norm_eps", 1e-6)
        self.cp_num_code_groups = getattr(config.talker_config, "num_code_groups", 16)
        self.cp_rope_theta = getattr(cp, "rope_theta", 1000000.0)
        # CP KV is fresh per frame; actual depth = num_code_groups (prefill 2 +
        # decode num_code_groups-2). Compile history at 2x that for an
        # off-by-one / RoPE-table safety margin. Talker KV accumulates across
        # frames and must stay at self.seq_length.
        self.cp_seq_length = 2 * self.cp_num_code_groups  # 32 for 16 code groups

        # 1D RoPE cos/sin for CodePredictor
        self.cp_cos, self.cp_sin = self._cp_rotary_tables()

        # extern block weights for Talker MRoPE (interleaved gather index)
        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}

        # register CodePredictor generation/compile hooks
        self.all_gen_mlirs.append(self.gen_code_predictor_mlir)
        self.all_compiles.append(self.compile_code_predictor)

        # Speaker Encoder (ECAPA-TDNN) config. Defaults from
        # configuration_qwen3_tts.Qwen3TTSSpeakerEncoderConfig.
        se = getattr(config, "speaker_encoder_config", None)
        self.spk_mel_dim = getattr(se, "mel_dim", 128)
        self.spk_enc_dim = getattr(se, "enc_dim", 1024)
        self.spk_enc_channels = getattr(se, "enc_channels", [512, 512, 512, 512, 1536])
        self.spk_enc_kernel_sizes = getattr(se, "enc_kernel_sizes", [5, 3, 3, 3, 1])
        self.spk_enc_dilations = getattr(se, "enc_dilations", [1, 2, 3, 4, 1])
        self.spk_enc_att_channels = getattr(se, "enc_attention_channels", 128)
        self.spk_enc_res2net_scale = getattr(se, "enc_res2net_scale", 8)
        self.spk_enc_se_channels = getattr(se, "enc_se_channels", 128)
        # Max mel-frame length for the speaker-encoder net (STATIC compile). The
        # pipeline zero-pads the reference mel to this length on the host so the
        # embedding sees the full reference audio. Set via --audio_length (mel
        # frames; 1 s = 24000/256 = 93.75 frames). Default 750 = 8 s (matches
        # the bundled clone.wav). Match it to the ref's actual mel length to
        # avoid zero-padding, which dilutes the embedding.
        self.spk_mel_frames = args.audio_length if args.audio_length > 0 else 750

        # register Speaker Encoder generation/compile hooks
        self.all_gen_mlirs.append(self.gen_speaker_encoder_mlir)
        self.all_compiles.append(self.compile_speaker_encoder)

        # Mimi speech-tokenizer (decoder only). For the Base model the TTS path
        # only needs the decode direction (codec codes -> waveform). Weights live
        # in a separate speech_tokenizer/model.safetensors; loaded lazily.
        tok_cfg = self._load_mimi_config()
        dc = tok_cfg.get("decoder_config", {})
        self.mimi_latent_dim = dc.get("latent_dim", 1024)
        self.mimi_codebook_dim = dc.get("codebook_dim", 512)
        self.mimi_codebook_size = dc.get("codebook_size", 2048)
        self.mimi_decoder_dim = dc.get("decoder_dim", 1536)
        self.mimi_hidden = dc.get("hidden_size", 512)
        self.mimi_num_layers = dc.get("num_hidden_layers", 8)
        self.mimi_num_heads = dc.get("num_attention_heads", 16)
        self.mimi_num_kv_heads = dc.get("num_key_value_heads", 16)
        self.mimi_head_dim = dc.get("head_dim", 64)
        self.mimi_inter_size = dc.get("intermediate_size", 1024)
        self.mimi_rms_eps = dc.get("rms_norm_eps", 1e-5)
        self.mimi_rope_theta = dc.get("rope_theta", 10000)
        self.mimi_sliding_window = dc.get("sliding_window", 72)
        self.mimi_num_quantizers = dc.get("num_quantizers", 16)
        self.mimi_upsample_rates = dc.get("upsample_rates", [8, 5, 4, 3])
        self.mimi_upsampling_ratios = dc.get("upsampling_ratios", [2, 2])
        # Static codec-frame length (decoder streaming block). Causal conv +
        # causal attention mean end-padding with zero codes does NOT affect
        # earlier frames -> exact for the real part. 32 frames = 2.56 s
        # first-packet latency.
        self.mimi_num_frames = 32
        self.mimi_total_upsample = 1
        for r in self.mimi_upsample_rates + self.mimi_upsampling_ratios:
            self.mimi_total_upsample *= r  # 1920
        self._mimi_loader = None

        # register Mimi decoder generation/compile hooks
        self.all_gen_mlirs.append(self.gen_mimi_decoder_mlir)
        self.all_compiles.append(self.compile_mimi_decoder)

        # ---- Mimi encoder (wav -> codes) config, for ICL voice clone ----
        # The encoder shares speech_tokenizer/config.json but uses its own
        # `encoder_config` dict (LayerNorm + plain MLP, not RMSNorm + SwiGLU).
        ec = tok_cfg.get("encoder_config", {})
        self.enc_audio_channels = ec.get("audio_channels", 1)
        self.enc_num_filters = ec.get("num_filters", 64)
        self.enc_hidden = ec.get("hidden_size", 512)
        self.enc_codebook_dim = ec.get("codebook_dim", 256)  # 256 (decoder is 512)
        self.enc_codebook_size = ec.get("codebook_size", 2048)
        self.enc_compress = ec.get("compress", 2)
        self.enc_kernel_size = ec.get("kernel_size", 7)
        self.enc_last_kernel = ec.get("last_kernel_size", 3)
        self.enc_residual_kernel = ec.get("residual_kernel_size", 3)
        self.enc_dilation_growth = ec.get("dilation_growth_rate", 2)
        self.enc_upsampling_ratios = ec.get("upsampling_ratios", [8, 6, 5, 4])
        self.enc_num_residual_layers = ec.get("num_residual_layers", 1)
        self.enc_num_layers = ec.get("num_hidden_layers", 8)
        self.enc_num_heads = ec.get("num_attention_heads", 8)
        self.enc_num_kv_heads = ec.get("num_key_value_heads", 8)
        self.enc_head_dim = ec.get("head_dim", 64)
        self.enc_inter_size = ec.get("intermediate_size", 2048)
        self.enc_norm_eps = ec.get("norm_eps", 1e-5)  # LayerNorm eps (w+b)
        self.enc_rope_theta = ec.get("rope_theta", 10000.0)
        self.enc_sliding_window = ec.get("sliding_window", 250)
        # encoder_valid_num_quantizers (top-level) = 16 effective codebooks.
        self.enc_valid_quantizers = tok_cfg.get("encoder_valid_num_quantizers", 16)
        self.enc_num_semantic = ec.get("num_semantic_quantizers", 1)
        # SEANet downsampling product = prod(upsampling_ratios) = 960; the
        # post-transformer downsample conv halves 25 Hz -> 12.5 Hz (stride 2, k 4).
        self.enc_seanet_downsample = 1
        for r in self.enc_upsampling_ratios:
            self.enc_seanet_downsample *= r  # 960
        self.enc_downsample_stride = 2  # 1920 / 960
        # Static lengths aligned to all strides (exact causal conv math).
        # T_audio is the mimi_encoder input length (wav samples @ 24 kHz),
        # derived from --audio_length (mel frames; 1 frame = hop 256 samples):
        # audio_length=750 -> T_audio=192000 (~8 s). mimi_encoder is ICL-only
        # and not compiled by default; these lengths only take effect when
        # QWEN3_TTS_ENABLE_ICL=1 (ICL is not supported — see README).
        self.enc_T_audio = self.spk_mel_frames * 256
        assert self.enc_T_audio % (2 * self.enc_seanet_downsample) == 0, \
            "T_audio must be divisible by 1920 (downsample 960 x stride 2)"
        self.enc_T_12 = self.enc_T_audio // (2 * self.enc_seanet_downsample)  # 750 -> 100
        self.enc_T_25 = self.enc_T_12 * 2  # 200

        # ICL voice clone is not delivered (ref_code precision could not be
        # aligned on this toolchain). The mimi_encoder net (wav -> codes) is
        # ICL-only and unused by x-vector-only clone, so by default it is NOT
        # compiled into the bmodel (saves ~116 MB). The gen_mimi_encoder_mlir
        # / compile_mimi_encoder code is kept below for reference; set
        # QWEN3_TTS_ENABLE_ICL=1 to build it.
        if os.environ.get("QWEN3_TTS_ENABLE_ICL"):
            self.all_gen_mlirs.append(self.gen_mimi_encoder_mlir)
            self.all_compiles.append(self.compile_mimi_encoder)

    # ------------------------------------------------------------------ #
    # Talker: config / MRoPE
    # ------------------------------------------------------------------ #
    @override
    def load_pretrained(self, config):
        self.config = config
        if self.loader is None:
            self.loader = SafetensorsModelHandle(self.model_path)
        self.model = self.loader
        self.model_type = config.model_type
        self.model_info = QWEN3_TTS_INFO
        # Talker config is the LLM body (not a text_config sub-config).
        self.llm_config = config.talker_config
        self.llm_type = LlmType.QWEN3  # base block path (QK-norm + GQA)

    @override
    def init_config(self):
        super().init_config()
        # Base class sets self.vocab_size = llm_config.vocab_size = 3072 (codec
        # vocab, which is the codec_head / sample_head vocab). Keep it.
        self.codec_vocab_size = self.vocab_size  # 3072
        self.text_vocab_size = getattr(self.llm_config, "text_vocab_size", 151936)
        self.text_hidden_size = getattr(self.llm_config, "text_hidden_size", 2048)
        # MRoPE needs 3-axis position_ids (temporal/height/width), same layout as
        # Qwen3VLConverter: [1, 3, max_input_length] (use_insert) or [3, max_input_length].
        if self.use_insert:
            self.position_shape = [1, 3, self.max_input_length]
        else:
            self.position_shape = [3, self.max_input_length]

    @override
    def rotary_embedding(self):
        # Same as Qwen3VLConverter: text_rotary_cos_sin + half slice for MRoPE.
        from .transformers_compat import text_rotary_cos_sin
        cos, sin = text_rotary_cos_sin(self.llm_config, self.seq_length)
        assert (cos.shape[-1] == self.head_dim)
        assert (sin.shape[-1] == self.head_dim)
        cos = cos[:, :, :self.head_dim // 2]
        sin = sin[:, :, :self.head_dim // 2]
        return cos, sin  # [seq, 1, 64]

    def get_mrope_index(self):
        # Identical to Qwen3VLConverter: build the interleaved frequency index
        # baked into the cos/sin gather weight.
        freqs = np.arange(0, 3 * self.head_dim // 2,
                          dtype=np.int32).reshape(3, 1, self.head_dim // 2)
        freqs_t = self.apply_interleaved_mrope(freqs)  # [1, 64]
        freqs_t = np.tile(freqs_t, (1, 2))  # [1, 128]
        return freqs_t.astype(np.float32)

    def apply_interleaved_mrope(self, freqs):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @property
    def mrope_section(self):
        rs = getattr(self.llm_config, "rope_scaling", None)
        if rs is None:
            return [24, 20, 20]
        if isinstance(rs, dict):
            return rs.get("mrope_section", [24, 20, 20])
        return getattr(rs, "mrope_section", [24, 20, 20])

    def mrope_batch(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, self.head_dim // 2])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([self.batch, 3, dim, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, self.batch, dim]),
                               in_op,
                               order=[1, 3, 0, 2],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, self.batch, dim]),
                               new_op,
                               shape=[3 * self.head_dim // 2, self.batch, -1],
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, self.batch, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([self.batch, dim, 1, self.head_dim]),
                               new_op,
                               order=[2, 3, 0, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    def mrope(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, 1, dim]),
                               in_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, 1, dim]),
                               new_op,
                               shape=[3 * self.head_dim // 2, 1, -1],
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, 1, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                               new_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        # MRoPE interleaved for the Talker (ported from Qwen3VLConverter).
        is_decode = pos_op.type.shape[-1] == 1
        if is_decode and self.use_insert:
            cos_op = self.mrope_batch(mlir_gen, pos_op, rotary_cos)
            sin_op = self.mrope_batch(mlir_gen, pos_op, rotary_sin)
        else:
            cos_op = self.mrope(mlir_gen, pos_op, rotary_cos)
            sin_op = self.mrope(mlir_gen, pos_op, rotary_sin)
        q_op = top.RopeOp(mlir_gen.get_tensor_type(q_op.type.shape),
                          q_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("q_proj", mlir_gen),
                          ip=mlir_gen.insert_point).output
        k_op = top.RopeOp(mlir_gen.get_tensor_type(k_op.type.shape),
                          k_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("k_cache", mlir_gen),
                          ip=mlir_gen.insert_point).output
        return q_op, k_op

    # ------------------------------------------------------------------ #
    # Talker: custom embedding (text_embedding + text_projection MLP) and
    # codec_embedding / codec_head. Block nets reuse the base class.
    # ------------------------------------------------------------------ #
    @override
    def gen_embedding_lmhead_mlir(self):
        tqdm.write("generate talker embedding / codec_embedding / codec_head mlir ...")
        # ---- weights ----
        text_emb_path = "talker.model.text_embedding.weight"
        text_emb = self.model.read(text_emb_path)  # (151936, 2048), GatherOp, no transpose
        # fc1/fc2 are nn.Linear: HF stores weight [out, in]; self.linear (BF16
        # path) declares weight [in, out] and emits MatMulOp without
        # right_transpose. Use set_linear_weight (same path as block projs) so
        # the weight is transposed and the bias is filled into the weight_dict
        # in one shot.
        text_wdict = {text_emb_path: text_emb}
        self.set_linear_weight("talker.text_projection.linear_fc1", text_wdict)
        self.set_linear_weight("talker.text_projection.linear_fc2", text_wdict)
        text_npz = "text_embedding_top_weights.npz"
        np.savez(text_npz, **text_wdict)

        codec_emb_path = "talker.model.codec_embedding.weight"
        codec_emb = self.model.read(codec_emb_path)  # (3072, 1024)
        codec_npz = "codec_embedding_top_weights.npz"
        np.savez(codec_npz, **{codec_emb_path: codec_emb})

        codec_head_path = "talker.codec_head.weight"
        codec_head_w = self.model.read(codec_head_path)  # (3072, 1024)
        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **{codec_head_path: codec_head_w})

        text_out_dim = self.hidden_size  # 1024
        # fc1.weight HF shape [out=2048, in=2048]; declared as [in, out] by
        # set_linear_weight (transposed). text_emb_dim = fc1 in = 2048,
        # fc1_out = fc1 out = 2048.
        _fc1_w_shape = self.model.read("talker.text_projection.linear_fc1.weight").shape
        text_emb_dim = _fc1_w_shape[1]  # 2048
        fc1_out = _fc1_w_shape[0]  # 2048

        def gen_text_embedding(name: str, seq_length: int):
            out_shape = [1, seq_length, text_out_dim]
            mlir = MLIRImporter([[1, seq_length]], [out_shape],
                                name,
                                self.platform,
                                input_types=["INT32"],
                                weight_file=f"../{text_npz}")
            ip = mlir.insert_point
            T = mlir.get_tensor_type
            L = lambda n: self.get_loc(n, mlir)
            ids = mlir.create_input_op(L("input_ids"), 0)
            w = mlir.create_weight_op(text_emb_path, [self.text_vocab_size, text_emb_dim])
            x = top.GatherOp(T([1, seq_length, text_emb_dim]),
                             w,
                             ids,
                             axis=0,
                             loc=L("text_embedding"),
                             ip=ip).output
            # text_projection: fc1 (2048->2048) + SiLU + fc2 (2048->1024).
            # Use self.linear so weight shape/bias/quant conventions match the
            # base block projs (weight [in, out], no right_transpose).
            x = self.linear(mlir, "talker.text_projection.linear_fc1", x, [text_emb_dim, fc1_out],
                            [1, seq_length, fc1_out])
            x = top.SiLUOp(T([1, seq_length, fc1_out]), x, loc=L("text_proj.act"), ip=ip).output
            x = self.linear(mlir, "talker.text_projection.linear_fc2", x, [fc1_out, text_out_dim],
                            out_shape)
            mlir.create_return_op([x])
            self.save_mlir_module(mlir, name)

        def gen_codec_embedding(name: str, seq_length: int):
            out_shape = [1, seq_length, text_out_dim]
            mlir = MLIRImporter([[1, seq_length]], [out_shape],
                                name,
                                self.platform,
                                input_types=["INT32"],
                                weight_file=f"../{codec_npz}")
            ip = mlir.insert_point
            ids = mlir.create_input_op(self.get_loc("input_ids", mlir), 0)
            w = mlir.create_weight_op(codec_emb_path, [self.codec_vocab_size, text_out_dim])
            x = top.GatherOp(mlir.get_tensor_type(out_shape),
                             w,
                             ids,
                             axis=0,
                             loc=self.get_loc("codec_embedding", mlir),
                             ip=ip).output
            mlir.create_return_op([x])
            self.save_mlir_module(mlir, name)

        def gen_codec_head():
            name = "lm_head"
            out_shape = [[1, self.codec_vocab_size]]
            if self.lmhead_with_topk:
                out_shape = [[1, 1]]
            mlir = MLIRImporter([[1, self.hidden_size]],
                                out_shape,
                                name,
                                self.platform,
                                weight_file=f"../{lmhead_npz}")
            ip = mlir.insert_point
            inp = mlir.create_input_op(self.get_loc("hidden_states", mlir), 0)
            w = mlir.create_weight_op(codec_head_path, [self.codec_vocab_size, self.hidden_size])
            x = top.MatMulOp(mlir.get_tensor_type([self.codec_vocab_size, 1]),
                             w,
                             inp,
                             mlir.none_op,
                             do_relu=False,
                             right_transpose=True,
                             loc=self.get_loc("codec_head", mlir),
                             ip=ip).output
            x = top.ReshapeOp(mlir.get_tensor_type([1, self.codec_vocab_size]),
                              x,
                              loc=self.get_loc("codec_head.reshape", mlir),
                              ip=ip).output
            if self.lmhead_with_topk:
                topk = top.TopKOp(*mlir.get_tensor_type([[1, 1], [1, 1]]),
                                  x,
                                  axis=1,
                                  K=1,
                                  loc=self.get_loc(["token_value", "token_id"], mlir),
                                  ip=ip)
                mlir.create_return_op([topk.indices])
            else:
                mlir.create_return_op([x])
            self.save_mlir_module(mlir, name)

        gen_text_embedding("embedding", self.max_input_length)
        gen_text_embedding("embedding_cache", 1)
        gen_codec_embedding("codec_embedding", self.max_input_length)
        gen_codec_embedding("codec_embedding_cache", 1)
        gen_codec_head()
        # sample_head / greedy_head (vocab = codec_vocab_size)
        self.gen_sample_head_mlir()

    @override
    def gen_sample_head_mlir(self, max_top_k=50, min_tokens_to_keep=5):
        # Force vocab_size to the codec vocab for sample_head / greedy_head.
        saved = self.vocab_size
        self.vocab_size = self.codec_vocab_size
        try:
            super().gen_sample_head_mlir(max_top_k, min_tokens_to_keep)
        finally:
            self.vocab_size = saved

    # ------------------------------------------------------------------ #
    # Speaker Encoder (ECAPA-TDNN)
    #
    # Input  : mel [1, T, 128]  (F32, host-computed STFT/mel; T = spk_mel_frames)
    # Output : spk_embed [1, 1024] (F32)
    #
    # All Conv1d use padding="same" (kept length). The HF reference uses
    # padding_mode="reflect"; TPU ConvOp only supports zero pad, so we use zero
    # padding here -- a small boundary approximation.
    # The net works in 4D [1, C, 1, T] for ConvOp (kernel_shape=[1, k]).
    # ------------------------------------------------------------------ #
    def gen_speaker_encoder_mlir(self):
        tqdm.write("generate speaker_encoder (ECAPA-TDNN) mlir ...")
        T = self.spk_mel_frames
        mel_dim = self.spk_mel_dim
        enc_channels = self.spk_enc_channels
        enc_kernels = self.spk_enc_kernel_sizes
        enc_dilations = self.spk_enc_dilations
        scale = self.spk_enc_res2net_scale
        se_channels = self.spk_enc_se_channels
        att_channels = self.spk_enc_att_channels
        enc_dim = self.spk_enc_dim

        # ---- collect weights (reshape conv [o,i,k] -> [o,i,1,k] for 4D ConvOp) ----
        spk_npz = "speaker_encoder_top_weights.npz"
        wd = {}

        def add_conv(path):
            w = self.model.read(path + ".weight")  # [o, i, k]
            b = self.model.read(path + ".bias")  # [o]
            wd[path + ".weight"] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            wd[path + ".bias"] = b

        # blocks[0]: TDNN(mel_dim -> enc_channels[0])
        add_conv("speaker_encoder.blocks.0.conv")
        # blocks[1..3]: SE-Res2Net
        for i in range(1, len(enc_channels) - 1):
            bp = f"speaker_encoder.blocks.{i}"
            add_conv(bp + ".tdnn1.conv")
            add_conv(bp + ".tdnn2.conv")
            for j in range(scale - 1):
                add_conv(f"{bp}.res2net_block.blocks.{j}.conv")
            add_conv(bp + ".se_block.conv1")
            add_conv(bp + ".se_block.conv2")
        add_conv("speaker_encoder.mfa.conv")
        add_conv("speaker_encoder.asp.tdnn.conv")
        add_conv("speaker_encoder.asp.conv")
        add_conv("speaker_encoder.fc")
        np.savez(spk_npz, **wd)

        # ---- build MLIR ----
        name = "speaker_encoder"
        in_shape = [1, T, mel_dim]
        out_shape = [1, enc_dim]
        mlir = MLIRImporter([in_shape], [out_shape],
                            name,
                            self.platform,
                            input_types=["F32"],
                            weight_file=f"../{spk_npz}")
        ip = mlir.insert_point
        TT = mlir.get_tensor_type
        L = lambda n: self.get_loc(n, mlir)

        def conv1d(x, path, C_in, C_out, k, d, loc, Tlen=None):
            # x: [1, C_in, 1, Tlen] -> [1, C_out, 1, Tlen], "same" pad (zero), no activation
            if Tlen is None:
                Tlen = T
            pad = (k - 1) * d
            left = pad // 2
            right = pad - left
            w = mlir.create_weight_op(path + ".weight", [C_out, C_in, 1, k])
            b = mlir.create_weight_op(path + ".bias", [C_out])
            return top.ConvOp(TT([1, C_out, 1, Tlen]),
                              x,
                              w,
                              b,
                              kernel_shape=[1, k],
                              strides=[1, 1],
                              dilations=[1, d],
                              pads=[0, left, 0, right],
                              group=1,
                              do_relu=False,
                              loc=L(loc),
                              ip=ip).output

        def tdnn(x, path, C_in, C_out, k, d, loc, Tlen=None):
            # TimeDelayNetBlock = Conv1d + ReLU
            y = conv1d(x, path, C_in, C_out, k, d, loc, Tlen=Tlen)
            return top.ReluOp(TT([1, C_out, 1, Tlen if Tlen else T]),
                              y,
                              loc=L(loc + ".relu"),
                              ip=ip).output

        def slice_channels(x, C_in, start, n, loc):
            # x: [1, C_in, 1, T] -> [1, n, 1, T]  (slice along channel axis 1)
            return top.SliceOp(TT([1, n, 1, T]),
                               x,
                               mlir.none_op,
                               mlir.none_op,
                               mlir.none_op,
                               offset=[0, start, 0, 0],
                               steps=[1, 1, 1, 1],
                               ends=[1, start + n, 1, T],
                               axes=[],
                               loc=L(loc),
                               ip=ip).output

        def res2net(x, C_in, k, d, blk_path, loc):
            # x: [1, C_in, 1, T]; scale chunks along channel; residual chain.
            sub = C_in // scale  # 64
            parts = [slice_channels(x, C_in, i * sub, sub, f"{loc}.chunk{i}") for i in range(scale)]
            out = [parts[0]]  # y0 = x0
            for i in range(1, scale):
                inp = parts[i]
                if i > 1:
                    inp = top.AddOp(TT([1, sub, 1, T]), [parts[i], out[i - 1]],
                                    loc=L(f"{loc}.add{i}"),
                                    ip=ip).output
                y = tdnn(inp, f"{blk_path}.blocks.{i - 1}.conv", sub, sub, k, d, f"{loc}.tdnn{i}")
                out.append(y)
            return top.ConcatOp(TT([1, C_in, 1, T]), out, axis=1, loc=L(loc + ".cat"), ip=ip).output

        def se_block(x, C_in, blk_path, loc):
            # SqueezeExcitation: mean over T -> conv1 -> relu -> conv2 -> sigmoid -> mul
            m = top.ReduceOp(TT([1, C_in, 1, 1]),
                             x,
                             axes=[3],
                             keepdims=True,
                             mode=StringAttr.get("ReduceMean"),
                             loc=L(loc + ".mean"),
                             ip=ip).output
            m = conv1d(m, blk_path + ".conv1", C_in, se_channels, 1, 1, loc + ".conv1", Tlen=1)
            m = top.ReluOp(TT([1, se_channels, 1, 1]), m, loc=L(loc + ".relu"), ip=ip).output
            m = conv1d(m, blk_path + ".conv2", se_channels, C_in, 1, 1, loc + ".conv2", Tlen=1)
            m = top.SigmoidOp(TT([1, C_in, 1, 1]), m, loc=L(loc + ".sigmoid"), ip=ip).output
            m = top.TileOp(TT([1, C_in, 1, T]), m, tile=[1, 1, 1, T], loc=L(loc + ".tile"),
                           ip=ip).output
            return top.MulOp(TT([1, C_in, 1, T]), [x, m], loc=L(loc + ".mul"), ip=ip).output

        def se_res2net(x, C_in, C_out, k, d, blk_path, loc):
            # tdnn1 -> res2net -> tdnn2 -> se -> + residual
            res = x
            y = tdnn(x, blk_path + ".tdnn1.conv", C_in, C_out, 1, 1, loc + ".tdnn1")
            y = res2net(y, C_out, k, d, blk_path + ".res2net_block", loc + ".res2net")
            y = tdnn(y, blk_path + ".tdnn2.conv", C_out, C_out, 1, 1, loc + ".tdnn2")
            y = se_block(y, C_out, blk_path + ".se_block", loc + ".se")
            return top.AddOp(TT([1, C_out, 1, T]), [y, res], loc=L(loc + ".res"), ip=ip).output

        # ---- forward ----
        mel = mlir.create_input_op(L("input_states"), 0)  # [1, T, 128]
        # transpose to [1, 128, T] then reshape to 4D [1, 128, 1, T]
        x = top.PermuteOp(TT([1, mel_dim, T]), mel, order=[0, 2, 1], loc=L("mel.permute"),
                          ip=ip).output
        x = top.ReshapeOp(TT([1, mel_dim, 1, T]), x, loc=L("mel.reshape4d"), ip=ip).output

        # blocks[0]: TDNN(mel_dim -> enc_channels[0])
        x = tdnn(x, "speaker_encoder.blocks.0.conv", mel_dim, enc_channels[0], enc_kernels[0],
                 enc_dilations[0], "spk.block0")
        block_outs = [x]
        # blocks[1..3]: SE-Res2Net(enc_channels[i-1] -> enc_channels[i])
        for i in range(1, len(enc_channels) - 1):
            x = se_res2net(x, enc_channels[i - 1], enc_channels[i], enc_kernels[i],
                           enc_dilations[i], f"speaker_encoder.blocks.{i}", f"spk.block{i}")
            block_outs.append(x)

        # mfa: TDNN(cat(block_outs[1:]) -> enc_channels[-1])
        mfa_in = top.ConcatOp(TT([1, enc_channels[-1], 1, T]),
                              block_outs[1:],
                              axis=1,
                              loc=L("spk.mfa.cat"),
                              ip=ip).output
        x = tdnn(mfa_in, "speaker_encoder.mfa.conv", enc_channels[-1], enc_channels[-1],
                 enc_kernels[-1], enc_dilations[-1], "spk.mfa")

        # AttentiveStatisticsPooling
        x = self._spk_asp(mlir, x, enc_channels[-1], att_channels, T, ip, L)
        # x: [1, 2*enc_channels[-1], 1, 1] = [1, 3072, 1, 1]

        # fc: Conv1d(3072 -> enc_dim, k=1) then squeeze
        x = conv1d(x, "speaker_encoder.fc", 2 * enc_channels[-1], enc_dim, 1, 1, "spk.fc", Tlen=1)
        # [1, 1024, 1, 1] -> [1, 1024]
        x = top.ReshapeOp(TT([1, enc_dim]), x, loc=L("spk.fc.squeeze"), ip=ip).output
        mlir.create_return_op([x])
        self.save_mlir_module(mlir, name)

    def _spk_asp(self, mlir, x, C, att_channels, T, ip, L):
        # AttentiveStatisticsPooling (mask all-ones, full length T).
        # x: [1, C, 1, T] -> [1, 2C, 1, 1]
        TT = mlir.get_tensor_type
        eps = 1e-12
        # uniform mean/std over T
        mean = top.ReduceOp(TT([1, C, 1, 1]),
                            x,
                            axes=[3],
                            keepdims=True,
                            mode=StringAttr.get("ReduceMean"),
                            loc=L("asp.umean"),
                            ip=ip).output
        mean_t = top.TileOp(TT([1, C, 1, T]),
                            mean,
                            tile=[1, 1, 1, T],
                            loc=L("asp.umean.tile"),
                            ip=ip).output
        sub = top.SubOp(TT([1, C, 1, T]), [x, mean_t], loc=L("asp.usub"), ip=ip).output
        sq = top.MulOp(TT([1, C, 1, T]), [sub, sub], loc=L("asp.usq"), ip=ip).output
        var = top.ReduceOp(TT([1, C, 1, 1]),
                           sq,
                           axes=[3],
                           keepdims=True,
                           mode=StringAttr.get("ReduceMean"),
                           loc=L("asp.uvar"),
                           ip=ip).output
        var = top.AddConstOp(TT([1, C, 1, 1]), var, const_val=eps, loc=L("asp.uvar.eps"),
                             ip=ip).output
        std = top.SqrtOp(TT([1, C, 1, 1]), var, loc=L("asp.ustd"), ip=ip).output
        std_t = top.TileOp(TT([1, C, 1, T]), std, tile=[1, 1, 1, T], loc=L("asp.ustd.tile"),
                           ip=ip).output
        att = top.ConcatOp(TT([1, 3 * C, 1, T]), [x, mean_t, std_t],
                           axis=1,
                           loc=L("asp.cat"),
                           ip=ip).output
        # tdnn: Conv1d(3C -> att, k=1) + ReLU
        w = mlir.create_weight_op("speaker_encoder.asp.tdnn.conv.weight",
                                  [att_channels, 3 * C, 1, 1])
        b = mlir.create_weight_op("speaker_encoder.asp.tdnn.conv.bias", [att_channels])
        att = top.ConvOp(TT([1, att_channels, 1, T]),
                         att,
                         w,
                         b,
                         kernel_shape=[1, 1],
                         strides=[1, 1],
                         dilations=[1, 1],
                         pads=[0, 0, 0, 0],
                         group=1,
                         do_relu=False,
                         loc=L("asp.tdnn"),
                         ip=ip).output
        att = top.ReluOp(TT([1, att_channels, 1, T]), att, loc=L("asp.tdnn.relu"), ip=ip).output
        att = top.TanhOp(TT([1, att_channels, 1, T]), att, loc=L("asp.tanh"), ip=ip).output
        # conv: Conv1d(att -> C, k=1), no activation
        w2 = mlir.create_weight_op("speaker_encoder.asp.conv.weight", [C, att_channels, 1, 1])
        b2 = mlir.create_weight_op("speaker_encoder.asp.conv.bias", [C])
        att = top.ConvOp(TT([1, C, 1, T]),
                         att,
                         w2,
                         b2,
                         kernel_shape=[1, 1],
                         strides=[1, 1],
                         dilations=[1, 1],
                         pads=[0, 0, 0, 0],
                         group=1,
                         do_relu=False,
                         loc=L("asp.conv"),
                         ip=ip).output
        # softmax over T (mask all-ones -> no masked_fill needed)
        att = top.SoftmaxOp(TT([1, C, 1, T]), att, axis=3, loc=L("asp.softmax"), ip=ip).output
        # weighted mean/std
        wx = top.MulOp(TT([1, C, 1, T]), [att, x], loc=L("asp.wx"), ip=ip).output
        wmean = top.ReduceOp(TT([1, C, 1, 1]),
                             wx,
                             axes=[3],
                             keepdims=True,
                             mode=StringAttr.get("ReduceSum"),
                             loc=L("asp.wmean"),
                             ip=ip).output
        wmean_t = top.TileOp(TT([1, C, 1, T]),
                             wmean,
                             tile=[1, 1, 1, T],
                             loc=L("asp.wmean.tile"),
                             ip=ip).output
        wsub = top.SubOp(TT([1, C, 1, T]), [x, wmean_t], loc=L("asp.wsub"), ip=ip).output
        wsq = top.MulOp(TT([1, C, 1, T]), [att, wsub], loc=L("asp.wsq"), ip=ip).output
        wsq = top.MulOp(TT([1, C, 1, T]), [wsq, wsub], loc=L("asp.wsq2"), ip=ip).output
        wvar = top.ReduceOp(TT([1, C, 1, 1]),
                            wsq,
                            axes=[3],
                            keepdims=True,
                            mode=StringAttr.get("ReduceSum"),
                            loc=L("asp.wvar"),
                            ip=ip).output
        wvar = top.AddConstOp(TT([1, C, 1, 1]), wvar, const_val=eps, loc=L("asp.wvar.eps"),
                              ip=ip).output
        wstd = top.SqrtOp(TT([1, C, 1, 1]), wvar, loc=L("asp.wstd"), ip=ip).output
        pooled = top.ConcatOp(TT([1, 2 * C, 1, 1]), [wmean, wstd],
                              axis=1,
                              loc=L("asp.pooled"),
                              ip=ip).output
        return pooled

    def compile_speaker_encoder(self):
        name = "speaker_encoder"
        if self.register_bmodel(name):
            return
        # STATIC compile at T = spk_mel_frames (default 750 ~= 8s). The pipeline
        # zero-pads the reference mel to spk_mel_frames frames on the host
        # (full-length mel); the padded tail-frames are silence and ECAPA is
        # robust to that.
        self.submit_deploy_task(
            name,
            [f'--quantize {self.half_precision_quantize}', '--quant_output'],
        )

    # ------------------------------------------------------------------ #
    # Mimi speech-tokenizer DECODER (codec codes -> waveform)
    #
    # Input  : codes [1, 16, T]  (INT32, T = mimi_num_frames)
    # Output : wav  [1, 1, 1920*T] (F32, clamped to [-1, 1])
    #
    # Only the decode direction is needed for the Base TTS path. All convs are
    # causal (left-pad); ConvTranspose1d crops the right tail. SnakeBeta uses
    # precomputed exp(alpha) and 1/(exp(beta)+eps) folded into constants. The
    # pre_transformer uses sliding-window causal attention (window=72) with a
    # precomputed additive mask. End-padding with zero codes is exact for the
    # real frames (causal).
    # ------------------------------------------------------------------ #
    def _load_mimi_config(self):
        import json
        cfg_path = f"{self.model_path}/speech_tokenizer/config.json"
        with open(cfg_path) as f:
            return json.load(f)

    def _mimi_model(self):
        if self._mimi_loader is None:
            self._mimi_loader = SafetensorsModelHandle(f"{self.model_path}/speech_tokenizer")
        return self._mimi_loader

    def gen_mimi_decoder_mlir(self):
        tqdm.write("generate mimi_decoder (codec codes -> wav) mlir ...")
        m = self._mimi_model()
        T = self.mimi_num_frames
        U = self.mimi_total_upsample
        cd = self.mimi_codebook_dim  # 512
        ld = self.mimi_latent_dim  # 1024
        dd = self.mimi_decoder_dim  # 1536
        cb_size = self.mimi_codebook_size  # 2048
        n_q = self.mimi_num_quantizers  # 16
        hidden = self.mimi_hidden  # 512
        n_heads = self.mimi_num_heads  # 16
        n_kv = self.mimi_num_kv_heads  # 16
        hd = self.mimi_head_dim  # 64
        inter = self.mimi_inter_size  # 1024
        rope_theta = self.mimi_rope_theta
        window = self.mimi_sliding_window
        up_rates = self.mimi_upsample_rates  # [8,5,4,3]
        up_ratios = self.mimi_upsampling_ratios  # [2,2]

        # ---- collect weights ----
        mimi_npz = "mimi_decoder_top_weights.npz"
        wd = {}

        def add_conv1d(prefix, in_c, out_c, k, groups=1, transpose=False):
            # PyTorch Conv1d weight [out, in/g, k] -> [out, in/g, 1, k].
            # For ConvTranspose1d PyTorch stores [in, out, k]; DeconvOp expects
            # [out, in, k] (Conv convention), so transpose (0,1) when requested.
            w = m.read(prefix + ".weight")
            if transpose:
                w = np.ascontiguousarray(np.transpose(w, (1, 0, 2)))
            wd[prefix + ".weight"] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            if m.is_exist(prefix + ".bias"):
                wd[prefix + ".bias"] = m.read(prefix + ".bias")

        def add_linear(prefix):
            # PyTorch Linear.weight is [out, in]; transpose to [in, out] to match
            # the base set_linear_weight convention (declared [in, out], no right_transpose).
            wd[prefix + ".weight"] = np.ascontiguousarray(m.read(prefix + ".weight").T)
            if m.is_exist(prefix + ".bias"):
                wd[prefix + ".bias"] = m.read(prefix + ".bias")

        def add_snake(prefix, dim):
            alpha = m.read(prefix + ".alpha")
            beta = m.read(prefix + ".beta")
            wd[prefix + ".alpha_e"] = np.exp(alpha).reshape(1, dim, 1, 1).astype(np.float32)
            wd[prefix + ".beta_inv"] = (1.0 / (np.exp(beta) + 1e-9)).reshape(1, dim, 1,
                                                                             1).astype(np.float32)

        # --- RVQ: 16 effective codebooks + 2 output_proj convs ---
        # rvq_first: 1 codebook (semantic), output_proj Conv1d(256->512, no bias)
        # rvq_rest:  15 codebooks (acoustic), output_proj Conv1d(256->512, no bias)
        cb_dim = cd // 2  # 256
        for k in range(n_q):
            if k == 0:
                cb = m.read("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
                cu = m.read("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
            else:
                cb = m.read(f"decoder.quantizer.rvq_rest.vq.layers.{k-1}._codebook.embedding_sum")
                cu = m.read(f"decoder.quantizer.rvq_rest.vq.layers.{k-1}._codebook.cluster_usage")
            eff = (cb / np.clip(cu, 1e-5, None)[:, None]).astype(np.float32)  # [2048, 256]
            wd[f"mimi.cb.{k}"] = eff
        # output_proj weights are [out, in, 1] (Conv1d); store as [out, in, 1, 1]
        for tag in ("rvq_first", "rvq_rest"):
            w = m.read(f"decoder.quantizer.{tag}.output_proj.weight")  # [512, 256, 1]
            wd[f"mimi.{tag}.out_proj.weight"] = w.reshape(w.shape[0], w.shape[1], 1, 1)

        # --- pre_conv: CausalConvNet(cd=512 -> ld=1024, k=3) ---
        add_conv1d("decoder.pre_conv.conv", cd, ld, 3)

        # --- pre_transformer ---
        add_linear("decoder.pre_transformer.input_proj")  # 1024 -> 512
        add_linear("decoder.pre_transformer.output_proj")  # 512 -> 1024
        wd["mimi.pre_tfm.norm.weight"] = m.read("decoder.pre_transformer.norm.weight")
        if m.is_exist("decoder.pre_transformer.norm.bias"):
            wd["mimi.pre_tfm.norm.bias"] = m.read("decoder.pre_transformer.norm.bias")
        for i in range(self.mimi_num_layers):
            p = f"decoder.pre_transformer.layers.{i}"
            add_linear(p + ".self_attn.q_proj")
            add_linear(p + ".self_attn.k_proj")
            add_linear(p + ".self_attn.v_proj")
            add_linear(p + ".self_attn.o_proj")
            add_linear(p + ".mlp.gate_proj")
            add_linear(p + ".mlp.up_proj")
            add_linear(p + ".mlp.down_proj")
            wd[f"mimi.pre_tfm.l{i}.inorm.weight"] = m.read(p + ".input_layernorm.weight")
            wd[f"mimi.pre_tfm.l{i}.pnorm.weight"] = m.read(p + ".post_attention_layernorm.weight")
            wd[f"mimi.pre_tfm.l{i}.ascale"] = m.read(p + ".self_attn_layer_scale.scale").reshape(
                1, 1, hidden)
            wd[f"mimi.pre_tfm.l{i}.mscale"] = m.read(p + ".mlp_layer_scale.scale").reshape(
                1, 1, hidden)
        # RoPE cos/sin [T, 1, hd]
        inv_freq = 1.0 / (rope_theta**(np.arange(0, hd, 2, dtype=np.float32) / hd))
        t = np.arange(T, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=1)
        wd["mimi.rope.cos"] = np.cos(emb).astype(np.float32).reshape(1, T, 1, hd)
        wd["mimi.rope.sin"] = np.sin(emb).astype(np.float32).reshape(1, T, 1, hd)
        # sliding-window causal additive mask [1, 1, T, T] (0 inside, -1e9 outside)
        mask = np.zeros((T, T), dtype=np.float32)
        for i in range(T):
            lo = max(0, i - window + 1)
            mask[i, :lo] = -1e9
            mask[i, i + 1:] = -1e9
        wd["mimi.attn.mask"] = mask.reshape(1, 1, T, T)

        # --- upsample: 2 stages (CausalTransConvNet + ConvNeXt) ---
        for s, ratio in enumerate(up_ratios):
            add_conv1d(f"decoder.upsample.{s}.0.conv", ld, ld, ratio,
                       transpose=True)  # ConvTranspose k=ratio, stride=ratio
            add_conv1d(f"decoder.upsample.{s}.1.dwconv.conv", ld, 1, 7, groups=ld)  # depthwise
            wd[f"mimi.up.{s}.norm.weight"] = m.read(f"decoder.upsample.{s}.1.norm.weight")
            wd[f"mimi.up.{s}.norm.bias"] = m.read(f"decoder.upsample.{s}.1.norm.bias")
            add_linear(f"decoder.upsample.{s}.1.pwconv1")  # 1024 -> 4096
            add_linear(f"decoder.upsample.{s}.1.pwconv2")  # 4096 -> 1024
            wd[f"mimi.up.{s}.gamma"] = m.read(f"decoder.upsample.{s}.1.gamma").reshape(1, ld, 1, 1)

        # --- decoder list ---
        # block 0: CausalConvNet(ld=1024 -> dd=1536, k=7)
        add_conv1d("decoder.decoder.0.conv", ld, dd, 7)
        # blocks 1..4: DecoderDecoderBlock
        for i in range(len(up_rates)):
            in_dim = dd // (2**i)
            out_dim = dd // (2**(i + 1))
            rate = up_rates[i]
            add_snake(f"decoder.decoder.{i+1}.block.0", in_dim)
            add_conv1d(f"decoder.decoder.{i+1}.block.1.conv",
                       in_dim,
                       out_dim,
                       2 * rate,
                       transpose=True)  # ConvTranspose
            for j, dil in enumerate((1, 3, 9)):
                bp = f"decoder.decoder.{i+1}.block.{j+2}"
                add_snake(bp + ".act1", out_dim)
                add_snake(bp + ".act2", out_dim)
                add_conv1d(bp + ".conv1.conv", out_dim, out_dim, 7)  # dilation applied in op
                add_conv1d(bp + ".conv2.conv", out_dim, out_dim, 1)
        out_dim_final = dd // (2**len(up_rates))  # 96
        add_snake("decoder.decoder.5", out_dim_final)
        add_conv1d("decoder.decoder.6.conv", out_dim_final, 1, 7)

        np.savez(mimi_npz, **wd)

        # ---- build MLIR ----
        name = "mimi_decoder"
        in_shape = [1, n_q, T]
        out_shape = [1, 1, U * T]
        mlir = MLIRImporter([in_shape], [out_shape],
                            name,
                            self.platform,
                            input_types=["INT32"],
                            weight_file=f"../{mimi_npz}")
        ip = mlir.insert_point
        TT = mlir.get_tensor_type
        L = lambda n: self.get_loc(n, mlir)
        none = mlir.none_op

        def W(key, shape):
            return mlir.create_weight_op(key, shape)

        def mlinear(x, prefix, in_f, out_f, shape):
            # Linear with weight [in,out] (stored raw [out,in] PyTorch, declared
            # [in,out] -> row-major reinterpret transposes). Bias only if present.
            w = W(prefix + ".weight", [in_f, out_f])
            if m.is_exist(prefix + ".bias"):
                b = W(prefix + ".bias", [1] * (len(shape) - 1) + [shape[-1]])
            else:
                b = none
            return top.MatMulOp(TT(shape), x, w, b, do_relu=False, loc=L(prefix), ip=ip).output

        def m_rms_norm(x, prefix, shape, eps):
            w = W(prefix + ".weight", [1] * (len(shape) - 1) + [shape[-1]])
            return top.RMSNormOp(TT(shape),
                                 x,
                                 w,
                                 eps=eps,
                                 weight_keep_f32=False,
                                 loc=L(prefix),
                                 ip=ip).output

        def m_layer_norm(x, prefix, shape, eps):
            w = W(prefix + ".weight", [1] * (len(shape) - 1) + [shape[-1]])
            b = W(prefix + ".bias", [1] * (len(shape) - 1) + [shape[-1]])
            return top.LayerNormOp(TT(shape),
                                   x,
                                   w,
                                   b,
                                   normalized_shape=[shape[-1]],
                                   axis=len(shape) - 1,
                                   eps=eps,
                                   loc=L(prefix),
                                   ip=ip).output

        def conv1d_causal(x, prefix, in_c, out_c, k, dilation=1, groups=1, Tlen=T):
            # x: [1, in_c, 1, Tlen] -> [1, out_c, 1, Tlen]; causal left-pad (k-1)*d
            pad = (k - 1) * dilation
            w = W(prefix + ".weight", [out_c, in_c // groups, 1, k])
            b = W(prefix + ".bias", [out_c]) if (prefix + ".bias") in wd else none
            return top.ConvOp(TT([1, out_c, 1, Tlen]),
                              x,
                              w,
                              b,
                              kernel_shape=[1, k],
                              strides=[1, 1],
                              dilations=[1, dilation],
                              pads=[0, pad, 0, 0],
                              group=groups,
                              do_relu=False,
                              loc=L(prefix),
                              ip=ip).output

        def deconv1d_causal(x, prefix, in_c, out_c, k, stride, Tlen):
            # ConvTranspose1d then crop right by (k-stride). x: [1, in_c, 1, Tlen]
            # -> [1, out_c, 1, Tlen*stride]. Weight stored transposed [out, in, 1, k].
            w = W(prefix + ".weight", [out_c, in_c, 1, k])
            b = W(prefix + ".bias", [out_c]) if (prefix + ".bias") in wd else none
            out_len = (Tlen - 1) * stride + k
            y = top.DeconvOp(TT([1, out_c, 1, out_len]),
                             x,
                             w,
                             b,
                             kernel_shape=[1, k],
                             strides=[1, stride],
                             dilations=[1, 1],
                             pads=[0, 0, 0, 0],
                             output_padding=[0, 0],
                             group=1,
                             do_relu=False,
                             loc=L(prefix),
                             ip=ip).output
            crop = k - stride
            if crop > 0:
                y = top.SliceOp(TT([1, out_c, 1, Tlen * stride]),
                                y,
                                none,
                                none,
                                none,
                                offset=[0, 0, 0, 0],
                                steps=[1, 1, 1, 1],
                                ends=[1, out_c, 1, Tlen * stride],
                                axes=[],
                                loc=L(prefix + ".crop"),
                                ip=ip).output
            return y

        def snake(x, prefix, dim, Tlen):
            # x: [1, dim, 1, Tlen] -> [1, dim, 1, Tlen]
            ae = W(prefix + ".alpha_e", [1, dim, 1, 1])
            bi = W(prefix + ".beta_inv", [1, dim, 1, 1])
            t1 = top.MulOp(TT([1, dim, 1, Tlen]), [x, ae], loc=L(prefix + ".mul_a"), ip=ip).output
            t2 = top.SinOp(TT([1, dim, 1, Tlen]), t1, loc=L(prefix + ".sin"), ip=ip).output
            t3 = top.MulOp(TT([1, dim, 1, Tlen]), [t2, t2], loc=L(prefix + ".sq"), ip=ip).output
            t4 = top.MulOp(TT([1, dim, 1, Tlen]), [t3, bi], loc=L(prefix + ".mul_b"), ip=ip).output
            return top.AddOp(TT([1, dim, 1, Tlen]), [x, t4], loc=L(prefix + ".add"), ip=ip).output

        # =================== forward ===================
        codes = mlir.create_input_op(L("input_states"), 0)  # [1, 16, T]

        # --- RVQ decode -> [1, cd=512, 1, T] ---
        # gather each codebook: codes[:,k] [1,T] -> gather cb[k] -> [1, T, 256] -> permute -> [1,256,1,T]
        gathers = []
        for k in range(n_q):
            ck = top.SliceOp(TT([1, 1, T]),
                             codes,
                             none,
                             none,
                             none,
                             offset=[0, k, 0],
                             steps=[1, 1, 1],
                             ends=[1, k + 1, T],
                             axes=[],
                             loc=L(f"rvq.slice{k}"),
                             ip=ip).output
            ck = top.ReshapeOp(TT([1, T]), ck, loc=L(f"rvq.rs{k}"), ip=ip).output
            cb = W(f"mimi.cb.{k}", [cb_size, cb_dim])
            g = top.GatherOp(TT([1, T, cb_dim]), cb, ck, axis=0, loc=L(f"rvq.gather{k}"),
                             ip=ip).output  # [1, T, 256]
            g = top.PermuteOp(TT([1, cb_dim, T]), g, order=[0, 2, 1], loc=L(f"rvq.perm{k}"),
                              ip=ip).output  # [1, 256, T]
            g = top.ReshapeOp(TT([1, cb_dim, 1, T]), g, loc=L(f"rv4d.{k}"), ip=ip).output
            gathers.append(g)
        g0 = gathers[0]
        g_rest = gathers[1]
        for k in range(2, n_q):
            g_rest = top.AddOp(TT([1, cb_dim, 1, T]), [g_rest, gathers[k]],
                               loc=L(f"rvq.sum{k}"),
                               ip=ip).output
        # output_proj: Conv1d(256->512, no bias) for each, then add
        def out_proj(g, tag):
            w = W(f"mimi.{tag}.out_proj.weight", [cd, cb_dim, 1, 1])
            return top.ConvOp(TT([1, cd, 1, T]),
                              g,
                              w,
                              none,
                              kernel_shape=[1, 1],
                              strides=[1, 1],
                              dilations=[1, 1],
                              pads=[0, 0, 0, 0],
                              group=1,
                              do_relu=False,
                              loc=L(f"rvq.{tag}.oproj"),
                              ip=ip).output

        hidden_x = top.AddOp(
            TT([1, cd, 1, T]),
            [out_proj(g0, "rvq_first"), out_proj(g_rest, "rvq_rest")],
            loc=L("rvq.add"),
            ip=ip).output  # [1, 512, 1, T]

        # --- pre_conv: CausalConvNet(512 -> 1024, k=3) -> transpose -> [1, T, 1024] ---
        hidden_x = conv1d_causal(hidden_x, "decoder.pre_conv.conv", cd, ld, 3)  # [1, 1024, 1, T]
        # [1, 1024, 1, T] -> [1, 1024, T] -> [1, T, 1024]
        hidden_x = top.ReshapeOp(TT([1, ld, T]), hidden_x, loc=L("preconv.rs3d"), ip=ip).output
        hidden_x = top.PermuteOp(TT([1, T, ld]),
                                 hidden_x,
                                 order=[0, 2, 1],
                                 loc=L("preconv.perm"),
                                 ip=ip).output

        # --- pre_transformer (8 layers) ---
        hidden_x = mlinear(hidden_x, "decoder.pre_transformer.input_proj", ld, hidden,
                           [1, T, hidden])
        cos_op = W("mimi.rope.cos", [1, T, 1, hd])
        sin_op = W("mimi.rope.sin", [1, T, 1, hd])
        mask_w = W("mimi.attn.mask", [1, 1, T, T])
        q_dim = n_heads * hd
        kv_dim = n_kv * hd
        for i in range(self.mimi_num_layers):
            p = f"decoder.pre_transformer.layers.{i}"
            residual = hidden_x
            ln = m_rms_norm(hidden_x, f"mimi.pre_tfm.l{i}.inorm", [1, T, hidden], self.mimi_rms_eps)
            q = mlinear(ln, p + ".self_attn.q_proj", hidden, q_dim, [1, T, q_dim])
            k = mlinear(ln, p + ".self_attn.k_proj", hidden, kv_dim, [1, T, kv_dim])
            v = mlinear(ln, p + ".self_attn.v_proj", hidden, kv_dim, [1, T, kv_dim])
            q = top.ReshapeOp(TT([1, T, n_heads, hd]),
                              q,
                              shape=[1, -1, n_heads, hd],
                              loc=L(f"tfm{i}.qrs"),
                              ip=ip).output
            k = top.ReshapeOp(TT([1, T, n_kv, hd]),
                              k,
                              shape=[1, -1, n_kv, hd],
                              loc=L(f"tfm{i}.krs"),
                              ip=ip).output
            v = top.ReshapeOp(TT([1, T, n_kv, hd]),
                              v,
                              shape=[1, -1, n_kv, hd],
                              loc=L(f"tfm{i}.vrs"),
                              ip=ip).output
            # 1D RoPE: positions are 0..T-1, cos/sin precomputed as [1,T,1,hd]
            q = top.RopeOp(TT([1, T, n_heads, hd]),
                           q,
                           sin_op,
                           cos_op,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=L(f"tfm{i}.qrope"),
                           ip=ip).output
            k = top.RopeOp(TT([1, T, n_kv, hd]),
                           k,
                           sin_op,
                           cos_op,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=L(f"tfm{i}.krope"),
                           ip=ip).output
            fa = top.FAttentionOp(TT([1, T, q_dim]),
                                  q,
                                  k,
                                  v,
                                  mask_w,
                                  none,
                                  scale=hd**-0.5,
                                  batch=1,
                                  q_head=n_heads,
                                  kv_head=n_kv,
                                  dim=hd,
                                  mq=T,
                                  mk=T,
                                  keep_dims=False,
                                  mask_size=0,
                                  loc=L(f"tfm{i}.fa"),
                                  ip=ip).output
            o = mlinear(fa, p + ".self_attn.o_proj", q_dim, hidden, [1, T, hidden])
            ascale = W(f"mimi.pre_tfm.l{i}.ascale", [1, 1, hidden])
            o = top.MulOp(TT([1, T, hidden]), [o, ascale], loc=L(f"tfm{i}.ascale"), ip=ip).output
            hidden_x = top.AddOp(TT([1, T, hidden]), [residual, o], loc=L(f"tfm{i}.ares"),
                                 ip=ip).output
            residual = hidden_x
            ln2 = m_rms_norm(hidden_x, f"mimi.pre_tfm.l{i}.pnorm", [1, T, hidden],
                             self.mimi_rms_eps)
            gate = mlinear(ln2, p + ".mlp.gate_proj", hidden, inter, [1, T, inter])
            up = mlinear(ln2, p + ".mlp.up_proj", hidden, inter, [1, T, inter])
            act = top.SiLUOp(TT([1, T, inter]), gate, loc=L(f"tfm{i}.mact"), ip=ip).output
            mul = top.MulOp(TT([1, T, inter]), [act, up], loc=L(f"tfm{i}.mmul"), ip=ip).output
            down = mlinear(mul, p + ".mlp.down_proj", inter, hidden, [1, T, hidden])
            mscale = W(f"mimi.pre_tfm.l{i}.mscale", [1, 1, hidden])
            down = top.MulOp(TT([1, T, hidden]), [down, mscale], loc=L(f"tfm{i}.mscale"),
                             ip=ip).output
            hidden_x = top.AddOp(TT([1, T, hidden]), [residual, down], loc=L(f"tfm{i}.mres"),
                                 ip=ip).output
        hidden_x = m_rms_norm(hidden_x, "mimi.pre_tfm.norm", [1, T, hidden], self.mimi_rms_eps)
        hidden_x = mlinear(hidden_x, "decoder.pre_transformer.output_proj", hidden, ld, [1, T, ld])

        # -> [1, 1024, 1, T]  (permute back)
        hidden_x = top.PermuteOp(TT([1, ld, T]),
                                 hidden_x,
                                 order=[0, 2, 1],
                                 loc=L("tfm.perm"),
                                 ip=ip).output
        hidden_x = top.ReshapeOp(TT([1, ld, 1, T]), hidden_x, loc=L("tfm.rs4d"), ip=ip).output

        # --- upsample: 2 stages (ConvTranspose k=ratio,stride=ratio + ConvNeXt) ---
        Tlen = T
        for s, ratio in enumerate(up_ratios):
            hidden_x = deconv1d_causal(hidden_x, f"decoder.upsample.{s}.0.conv", ld, ld, ratio,
                                       ratio, Tlen)
            Tlen = Tlen * ratio
            # ConvNeXt block (operates in [1, ld, 1, Tlen])
            inp = hidden_x
            x = conv1d_causal(hidden_x,
                              f"decoder.upsample.{s}.1.dwconv.conv",
                              ld,
                              ld,
                              7,
                              dilation=1,
                              groups=ld,
                              Tlen=Tlen)  # depthwise
            # -> [1, Tlen, ld] for LayerNorm + Linears
            x = top.ReshapeOp(TT([1, ld, Tlen]), x, loc=L(f"cnx{s}.rs"), ip=ip).output
            x = top.PermuteOp(TT([1, Tlen, ld]), x, order=[0, 2, 1], loc=L(f"cnx{s}.perm"),
                              ip=ip).output
            x = m_layer_norm(x, f"mimi.up.{s}.norm", [1, Tlen, ld], 1e-6)
            x = mlinear(x, f"decoder.upsample.{s}.1.pwconv1", ld, 4 * ld, [1, Tlen, 4 * ld])
            x = top.GELUOp(TT([1, Tlen, 4 * ld]), x, loc=L(f"cnx{s}.gelu"), ip=ip).output
            x = mlinear(x, f"decoder.upsample.{s}.1.pwconv2", 4 * ld, ld, [1, Tlen, ld])
            gamma = W(f"mimi.up.{s}.gamma", [1, ld, 1, 1])
            x = top.PermuteOp(TT([1, ld, Tlen]), x, order=[0, 2, 1], loc=L(f"cnx{s}.perm2"),
                              ip=ip).output
            x = top.ReshapeOp(TT([1, ld, 1, Tlen]), x, loc=L(f"cnx{s}.rs4d"), ip=ip).output
            x = top.MulOp(TT([1, ld, 1, Tlen]), [x, gamma], loc=L(f"cnx{s}.gamma"), ip=ip).output
            hidden_x = top.AddOp(TT([1, ld, 1, Tlen]), [inp, x], loc=L(f"cnx{s}.res"), ip=ip).output

        # --- decoder list ---
        # block 0: CausalConvNet(ld=1024 -> dd=1536, k=7)
        wav = conv1d_causal(hidden_x, "decoder.decoder.0.conv", ld, dd, 7,
                            Tlen=Tlen)  # [1, 1536, 1, Tlen]
        # blocks 1..4: DecoderDecoderBlock
        for i in range(len(up_rates)):
            in_dim = dd // (2**i)
            out_dim = dd // (2**(i + 1))
            rate = up_rates[i]
            wav = snake(wav, f"decoder.decoder.{i+1}.block.0", in_dim, Tlen)
            wav = deconv1d_causal(wav, f"decoder.decoder.{i+1}.block.1.conv", in_dim, out_dim,
                                  2 * rate, rate, Tlen)
            Tlen = Tlen * rate
            for j, dil in enumerate((1, 3, 9)):
                bp = f"decoder.decoder.{i+1}.block.{j+2}"
                res = wav
                wav = snake(wav, bp + ".act1", out_dim, Tlen)
                wav = conv1d_causal(wav,
                                    bp + ".conv1.conv",
                                    out_dim,
                                    out_dim,
                                    7,
                                    dilation=dil,
                                    Tlen=Tlen)
                wav = snake(wav, bp + ".act2", out_dim, Tlen)
                wav = conv1d_causal(wav, bp + ".conv2.conv", out_dim, out_dim, 1, Tlen=Tlen)
                wav = top.AddOp(TT([1, out_dim, 1, Tlen]), [wav, res],
                                loc=L(f"dec{i}.ru{j}.res"),
                                ip=ip).output
        out_dim_final = dd // (2**len(up_rates))  # 96
        wav = snake(wav, "decoder.decoder.5", out_dim_final, Tlen)
        wav = conv1d_causal(wav, "decoder.decoder.6.conv", out_dim_final, 1, 7,
                            Tlen=Tlen)  # [1, 1, 1, Tlen]
        # -> [1, 1, Tlen]
        wav = top.ReshapeOp(TT([1, 1, Tlen]), wav, loc=L("wav.rs"), ip=ip).output
        # clamp [-1, 1]
        wav = top.ClipOp(TT([1, 1, Tlen]), wav, min=-1.0, max=1.0, loc=L("wav.clamp"), ip=ip).output
        mlir.create_return_op([wav])
        self.save_mlir_module(mlir, name)

    def compile_mimi_decoder(self):
        name = "mimi_decoder"
        if self.register_bmodel(name):
            return
        self.submit_deploy_task(
            name,
            [f'--quantize {self.half_precision_quantize}', '--quant_output'],
        )

    def gen_mimi_encoder_mlir(self):
        """Mimi encoder: wav [1,1,T_audio] (24 kHz) -> codes [16, T_12.5] (12.5 Hz).

        Pipeline: SEANet encoder (causal conv, ÷960) -> 8-layer transformer
        (LayerNorm + GELU MLP, sliding-window 250) -> downsample conv (÷2,
        replicate pad) -> SplitRVQ encode (1 semantic + 15 acoustic = 16
        codebooks, argmin-cdist via argmax(matmul+bias) trick).
        """
        tqdm.write("generate mimi_encoder (wav -> codes) mlir ...")
        m = self._mimi_model()
        T_audio = self.enc_T_audio  # 192000 default (spk_mel_frames * 256)
        T_25 = self.enc_T_25  # 512
        T_12 = self.enc_T_12  # 256
        hidden = self.enc_hidden  # 512
        n_filters = self.enc_num_filters  # 64
        cb_dim = self.enc_codebook_dim  # 256
        cb_size = self.enc_codebook_size  # 2048
        n_q = self.enc_valid_quantizers  # 16
        n_sem = self.enc_num_semantic  # 1
        n_aco = n_q - n_sem  # 15
        n_heads = self.enc_num_heads  # 8
        n_kv = self.enc_num_kv_heads  # 8
        hd = self.enc_head_dim  # 64
        inter = self.enc_inter_size  # 2048
        eps = self.enc_norm_eps  # 1e-5
        rope_theta = self.enc_rope_theta
        window = self.enc_sliding_window  # 250
        comp = self.enc_compress  # 2
        k_se = self.enc_kernel_size  # 7
        k_last = self.enc_last_kernel  # 3
        k_res = self.enc_residual_kernel  # 3
        dil_growth = self.enc_dilation_growth  # 2
        ratios = list(reversed(self.enc_upsampling_ratios))  # [4,5,6,8]
        ds_stride = self.enc_downsample_stride  # 2

        # ---- collect weights ----
        enc_npz = "mimi_encoder_top_weights.npz"
        wd = {}

        def add_conv1d(prefix, in_c, out_c, k, groups=1):
            # Conv1d weight [out, in/g, k] -> [out, in/g, 1, k].
            w = m.read(prefix + ".weight")
            wd[prefix + ".weight"] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            if m.is_exist(prefix + ".bias"):
                wd[prefix + ".bias"] = m.read(prefix + ".bias")

        def add_linear(prefix):
            # Linear [out,in] -> [in,out] for MatMulOp convention.
            wd[prefix + ".weight"] = np.ascontiguousarray(m.read(prefix + ".weight").T)
            if m.is_exist(prefix + ".bias"):
                wd[prefix + ".bias"] = m.read(prefix + ".bias")

        # --- SEANet encoder layers (0..14) ---
        # Layout: 0=init conv; for each ratio in [4,5,6,8]: resnet block, ELU,
        # down conv; then ELU + final conv (14). resnet at 1,4,7,10; down at 3,6,9,12.
        add_conv1d("encoder.encoder.layers.0.conv", 1, n_filters, k_se)
        resnet_layers = [1, 4, 7, 10]
        down_layers = [3, 6, 9, 12]
        for i, ratio in enumerate(ratios):
            cur = (2**i) * n_filters
            rbp = f"encoder.encoder.layers.{resnet_layers[i]}.block"
            add_conv1d(rbp + ".1.conv", cur, cur // comp, k_res)
            add_conv1d(rbp + ".3.conv", cur // comp, cur, 1)
            add_conv1d(f"encoder.encoder.layers.{down_layers[i]}.conv", cur, cur * 2, 2 * ratio)
        add_conv1d("encoder.encoder.layers.14.conv", (2**len(ratios)) * n_filters, hidden, k_last)

        # --- encoder_transformer (8 layers, LayerNorm w+b, plain MLP fc1+GELU+fc2) ---
        for i in range(self.enc_num_layers):
            p = f"encoder.encoder_transformer.layers.{i}"
            add_linear(p + ".self_attn.q_proj")
            add_linear(p + ".self_attn.k_proj")
            add_linear(p + ".self_attn.v_proj")
            add_linear(p + ".self_attn.o_proj")
            add_linear(p + ".mlp.fc1")
            add_linear(p + ".mlp.fc2")
            wd[f"mimi.enc.l{i}.inorm.weight"] = m.read(p + ".input_layernorm.weight")
            wd[f"mimi.enc.l{i}.inorm.bias"] = m.read(p + ".input_layernorm.bias")
            wd[f"mimi.enc.l{i}.pnorm.weight"] = m.read(p + ".post_attention_layernorm.weight")
            wd[f"mimi.enc.l{i}.pnorm.bias"] = m.read(p + ".post_attention_layernorm.bias")
            wd[f"mimi.enc.l{i}.ascale"] = m.read(p + ".self_attn_layer_scale.scale").reshape(
                1, 1, hidden)
            wd[f"mimi.enc.l{i}.mscale"] = m.read(p + ".mlp_layer_scale.scale").reshape(1, 1, hidden)
        # RoPE cos/sin [T_25, 1, hd]
        inv_freq = 1.0 / (rope_theta**(np.arange(0, hd, 2, dtype=np.float32) / hd))
        t = np.arange(T_25, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=1)
        wd["mimi.enc.rope.cos"] = np.cos(emb).astype(np.float32).reshape(1, T_25, 1, hd)
        wd["mimi.enc.rope.sin"] = np.sin(emb).astype(np.float32).reshape(1, T_25, 1, hd)
        # sliding-window causal additive mask [1, 1, T_25, T_25]
        mask = np.zeros((T_25, T_25), dtype=np.float32)
        for i in range(T_25):
            lo = max(0, i - window + 1)
            mask[i, :lo] = -1e9
            mask[i, i + 1:] = -1e9
        wd["mimi.enc.attn.mask"] = mask.reshape(1, 1, T_25, T_25)

        # --- downsample conv (512->512, k=4, stride=2, no bias, replicate pad) ---
        w = m.read("encoder.downsample.conv.weight")  # [512, 512, 4]
        wd["mimi.enc.downsample.weight"] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])

        # --- RVQ: 16 effective codebooks (1 semantic + 15 acoustic) ---
        # Each branch has its own input_proj Conv1d(512->256, k=1, no bias).
        # codebook effective = embed_sum / cluster_usage.
        def add_rvq_branch(tag, n_codes):
            wip = m.read(f"encoder.quantizer.{tag}_residual_vector_quantizer.input_proj.weight")
            wd[f"mimi.enc.{tag}.input_proj.weight"] = wip.reshape(wip.shape[0], wip.shape[1], 1, 1)
            for i in range(n_codes):
                cb = m.read(
                    f"encoder.quantizer.{tag}_residual_vector_quantizer.layers.{i}.codebook.embed_sum"
                )
                cu = m.read(
                    f"encoder.quantizer.{tag}_residual_vector_quantizer.layers.{i}.codebook.cluster_usage"
                )
                eff = (cb / np.clip(cu, 1e-5, None)[:, None]).astype(np.float32)  # [2048, 256]
                wd[f"mimi.enc.{tag}.cb.{i}"] = eff  # for gather
                wd[f"mimi.enc.{tag}.cbT.{i}"] = np.ascontiguousarray(
                    eff.T)  # [256, 2048] for matmul
                wd[f"mimi.enc.{tag}.bias.{i}"] = (-0.5 * (eff**2).sum(axis=1)).reshape(
                    1, 1, cb_size)

        add_rvq_branch("semantic", n_sem)
        add_rvq_branch("acoustic", n_aco)

        np.savez(enc_npz, **wd)

        # ---- build MLIR ----
        name = "mimi_encoder"
        in_shape = [1, self.enc_audio_channels, T_audio]
        out_shape = [[n_q, T_12]]  # output 0 = codes
        mlir = MLIRImporter([in_shape],
                            out_shape,
                            name,
                            self.platform,
                            input_types=["F32"],
                            weight_file=f"../{enc_npz}")
        ip = mlir.insert_point
        TT = mlir.get_tensor_type

        L = lambda n: self.get_loc(n, mlir)
        none = mlir.none_op

        def W(key, shape):
            return mlir.create_weight_op(key, shape)

        def mlinear(x, prefix, in_f, out_f, shape):
            # Bias only if present in npz (encoder_transformer q/k/v/o/fc1/fc2 are
            # bias-free per config attention_bias=False + plain MLP w/o bias).
            w = W(prefix + ".weight", [in_f, out_f])
            b = W(prefix + ".bias", [1] * (len(shape) - 1) + [shape[-1]]) \
                if m.is_exist(prefix + ".bias") else none
            return top.MatMulOp(TT(shape), x, w, b, do_relu=False, loc=L(prefix), ip=ip).output

        def m_layer_norm(x, prefix, shape, eps):
            w = W(prefix + ".weight", [1] * (len(shape) - 1) + [shape[-1]])
            b = W(prefix + ".bias", [1] * (len(shape) - 1) + [shape[-1]])
            return top.LayerNormOp(TT(shape),
                                   x,
                                   w,
                                   b,
                                   normalized_shape=[shape[-1]],
                                   axis=len(shape) - 1,
                                   eps=eps,
                                   loc=L(prefix),
                                   ip=ip).output

        def conv1d_causal(x, prefix, in_c, out_c, k, dilation=1, groups=1, Tlen=T_audio):
            # x: [1, in_c, 1, Tlen] -> [1, out_c, 1, Tlen]; causal left-pad (k-1)*d (zero)
            pad = (k - 1) * dilation
            w = W(prefix + ".weight", [out_c, in_c // groups, 1, k])
            b = W(prefix + ".bias", [out_c]) if (prefix + ".bias") in wd else none
            return top.ConvOp(TT([1, out_c, 1, Tlen]),
                              x,
                              w,
                              b,
                              kernel_shape=[1, k],
                              strides=[1, 1],
                              dilations=[1, dilation],
                              pads=[0, pad, 0, 0],
                              group=groups,
                              do_relu=False,
                              loc=L(prefix),
                              ip=ip).output

        def conv1d_causal_down(x, prefix, in_c, out_c, k, stride, Tin, Tout):
            # causal down conv: left-pad (k-stride) zero, stride>1, aligned input.
            pad = k - stride
            w = W(prefix + ".weight", [out_c, in_c, 1, k])
            b = W(prefix + ".bias", [out_c]) if (prefix + ".bias") in wd else none
            return top.ConvOp(TT([1, out_c, 1, Tout]),
                              x,
                              w,
                              b,
                              kernel_shape=[1, k],
                              strides=[1, stride],
                              dilations=[1, 1],
                              pads=[0, pad, 0, 0],
                              group=1,
                              do_relu=False,
                              loc=L(prefix),
                              ip=ip).output

        def resnet_block(x, layer_idx, dim, Tin, tag=None):
            # ELU -> Conv1d(dim->dim/2, k=3, dil=1) -> ELU -> Conv1d(dim/2->dim, k=1) + residual
            res = x
            bp = f"encoder.encoder.layers.{layer_idx}.block"
            h = top.EluOp(TT([1, dim, 1, Tin]), x, alpha=1.0, loc=L(bp + ".elu0"), ip=ip).output
            h = conv1d_causal(h, bp + ".1.conv", dim, dim // comp, k_res, Tlen=Tin)
            h = top.EluOp(TT([1, dim // comp, 1, Tin]), h, alpha=1.0, loc=L(bp + ".elu1"),
                          ip=ip).output
            h = conv1d_causal(h, bp + ".3.conv", dim // comp, dim, 1, Tlen=Tin)
            out = top.AddOp(TT([1, dim, 1, Tin]), [res, h], loc=L(bp + ".res"), ip=ip).output
            return out

        # =================== forward ===================
        wav = mlir.create_input_op(L("input_states"), 0)  # [1, 1, T_audio]
        x = top.ReshapeOp(TT([1, 1, 1, T_audio]), wav, loc=L("in.rs4d"), ip=ip).output  # [1,1,1,T]

        # --- SEANet encoder ---
        x = conv1d_causal(x, "encoder.encoder.layers.0.conv", 1, n_filters, k_se)  # [1,64,1,T]
        Tlen = T_audio
        for i, ratio in enumerate(ratios):
            dim = (2**i) * n_filters
            x = resnet_block(x, resnet_layers[i], dim, Tlen, tag=f"resnet{i}")
            x = top.EluOp(TT([1, dim, 1, Tlen]), x, alpha=1.0, loc=L(f"se.elu{i}"), ip=ip).output
            Tout = Tlen // ratio
            x = conv1d_causal_down(x, f"encoder.encoder.layers.{down_layers[i]}.conv", dim, dim * 2,
                                   2 * ratio, ratio, Tlen, Tout)
            Tlen = Tout
        # Tlen == T_25 here
        x = top.EluOp(TT([1, (2**len(ratios)) * n_filters, 1, Tlen]),
                      x,
                      alpha=1.0,
                      loc=L("se.elu_final"),
                      ip=ip).output
        x = conv1d_causal(x,
                          "encoder.encoder.layers.14.conv", (2**len(ratios)) * n_filters,
                          hidden,
                          k_last,
                          Tlen=Tlen)  # [1,512,1,T_25]

        # --- encoder_transformer (8 layers) ---
        # [1, hidden, 1, T_25] -> [1, T_25, hidden]
        x = top.ReshapeOp(TT([1, hidden, T_25]), x, loc=L("tfm.in.rs3d"), ip=ip).output
        x = top.PermuteOp(TT([1, T_25, hidden]), x, order=[0, 2, 1], loc=L("tfm.in.perm"),
                          ip=ip).output
        cos_op = W("mimi.enc.rope.cos", [1, T_25, 1, hd])
        sin_op = W("mimi.enc.rope.sin", [1, T_25, 1, hd])
        mask_w = W("mimi.enc.attn.mask", [1, 1, T_25, T_25])
        q_dim = n_heads * hd
        kv_dim = n_kv * hd
        for i in range(self.enc_num_layers):
            p = f"encoder.encoder_transformer.layers.{i}"
            residual = x
            ln = m_layer_norm(x, f"mimi.enc.l{i}.inorm", [1, T_25, hidden], eps)
            q = mlinear(ln, p + ".self_attn.q_proj", hidden, q_dim, [1, T_25, q_dim])
            k = mlinear(ln, p + ".self_attn.k_proj", hidden, kv_dim, [1, T_25, kv_dim])
            v = mlinear(ln, p + ".self_attn.v_proj", hidden, kv_dim, [1, T_25, kv_dim])
            q = top.ReshapeOp(TT([1, T_25, n_heads, hd]),
                              q,
                              shape=[1, -1, n_heads, hd],
                              loc=L(f"tfm{i}.qrs"),
                              ip=ip).output
            k = top.ReshapeOp(TT([1, T_25, n_kv, hd]),
                              k,
                              shape=[1, -1, n_kv, hd],
                              loc=L(f"tfm{i}.krs"),
                              ip=ip).output
            v = top.ReshapeOp(TT([1, T_25, n_kv, hd]),
                              v,
                              shape=[1, -1, n_kv, hd],
                              loc=L(f"tfm{i}.vrs"),
                              ip=ip).output
            q = top.RopeOp(TT([1, T_25, n_heads, hd]),
                           q,
                           sin_op,
                           cos_op,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=L(f"tfm{i}.qrope"),
                           ip=ip).output
            k = top.RopeOp(TT([1, T_25, n_kv, hd]),
                           k,
                           sin_op,
                           cos_op,
                           rope_mode=StringAttr.get("contiguous_halves"),
                           loc=L(f"tfm{i}.krope"),
                           ip=ip).output
            fa = top.FAttentionOp(TT([1, T_25, q_dim]),
                                  q,
                                  k,
                                  v,
                                  mask_w,
                                  none,
                                  scale=hd**-0.5,
                                  batch=1,
                                  q_head=n_heads,
                                  kv_head=n_kv,
                                  dim=hd,
                                  mq=T_25,
                                  mk=T_25,
                                  keep_dims=False,
                                  mask_size=0,
                                  loc=L(f"tfm{i}.fa"),
                                  ip=ip).output
            o = mlinear(fa, p + ".self_attn.o_proj", q_dim, hidden, [1, T_25, hidden])
            ascale = W(f"mimi.enc.l{i}.ascale", [1, 1, hidden])
            o = top.MulOp(TT([1, T_25, hidden]), [o, ascale], loc=L(f"tfm{i}.ascale"), ip=ip).output
            x = top.AddOp(TT([1, T_25, hidden]), [residual, o], loc=L(f"tfm{i}.ares"), ip=ip).output
            residual = x
            ln2 = m_layer_norm(x, f"mimi.enc.l{i}.pnorm", [1, T_25, hidden], eps)
            h = mlinear(ln2, p + ".mlp.fc1", hidden, inter, [1, T_25, inter])
            h = top.GELUOp(TT([1, T_25, inter]), h, loc=L(f"tfm{i}.mact"), ip=ip).output
            h = mlinear(h, p + ".mlp.fc2", inter, hidden, [1, T_25, hidden])
            mscale = W(f"mimi.enc.l{i}.mscale", [1, 1, hidden])
            h = top.MulOp(TT([1, T_25, hidden]), [h, mscale], loc=L(f"tfm{i}.mscale"), ip=ip).output
            x = top.AddOp(TT([1, T_25, hidden]), [residual, h], loc=L(f"tfm{i}.mres"), ip=ip).output
        # -> [1, hidden, 1, T_25]
        x = top.PermuteOp(TT([1, hidden, T_25]), x, order=[0, 2, 1], loc=L("tfm.out.perm"),
                          ip=ip).output
        x = top.ReshapeOp(TT([1, hidden, 1, T_25]), x, loc=L("tfm.out.rs4d"), ip=ip).output

        # --- downsample conv (512->512, k=4, stride=2, replicate pad, no bias) ---
        # causal left-pad 2 (replicate) on W dim, then conv stride 2 -> [1,512,1,T_12]
        left = 2  # k - stride
        x = top.PadOp(TT([1, hidden, 1, T_25 + left]),
                      x,
                      paddings=[0, 0, 0, left, 0, 0, 0, 0],
                      val=0.0,
                      mode=StringAttr.get("edge"),
                      loc=L("ds.pad"),
                      ip=ip).output
        w_ds = W("mimi.enc.downsample.weight", [hidden, hidden, 1, 4])
        x = top.ConvOp(TT([1, hidden, 1, T_12]),
                       x,
                       w_ds,
                       none,
                       kernel_shape=[1, 4],
                       strides=[1, ds_stride],
                       dilations=[1, 1],
                       pads=[0, 0, 0, 0],
                       group=1,
                       do_relu=False,
                       loc=L("ds.conv"),
                       ip=ip).output  # [1,512,1,T_12]

        # --- SplitRVQ encode ---
        # emb = x [1, 512, 1, T_12]. Each branch: input_proj Conv1d(512->256,k=1)
        # -> [1,256,1,T_12] -> permute [1,T_12,256]; then residual codebook loop.
        def rvq_branch(emb4d, tag, n_codes):
            wip = W(f"mimi.enc.{tag}.input_proj.weight", [cb_dim, hidden, 1, 1])
            r = top.ConvOp(TT([1, cb_dim, 1, T_12]),
                           emb4d,
                           wip,
                           none,
                           kernel_shape=[1, 1],
                           strides=[1, 1],
                           dilations=[1, 1],
                           pads=[0, 0, 0, 0],
                           group=1,
                           do_relu=False,
                           loc=L(f"rvq.{tag}.iproj"),
                           ip=ip).output  # [1,256,1,T_12]
            r = top.ReshapeOp(TT([1, cb_dim, T_12]), r, loc=L(f"rvq.{tag}.rs"), ip=ip).output
            r = top.PermuteOp(TT([1, T_12, cb_dim]),
                              r,
                              order=[0, 2, 1],
                              loc=L(f"rvq.{tag}.perm"),
                              ip=ip).output  # [1,T_12,256]
            idxs = []
            for i in range(n_codes):
                cbT = W(f"mimi.enc.{tag}.cbT.{i}", [cb_dim, cb_size])
                bias = W(f"mimi.enc.{tag}.bias.{i}", [1, 1, cb_size])
                # scores = residual @ cbT + bias  -> [1, T_12, 2048]
                scores = top.MatMulOp(TT([1, T_12, cb_size]),
                                      r,
                                      cbT,
                                      bias,
                                      do_relu=False,
                                      loc=L(f"rvq.{tag}.mm{i}"),
                                      ip=ip).output
                topk = top.TopKOp(TT([1, T_12, 1]),
                                  TT([1, T_12, 1]),
                                  scores,
                                  axis=2,
                                  K=1,
                                  loc=L([f"rvq.{tag}.val{i}", f"rvq.{tag}.idx{i}"]),
                                  ip=ip)
                idx = top.ReshapeOp(TT([1, T_12]),
                                    topk.indices,
                                    loc=L(f"rvq.{tag}.idxrs{i}"),
                                    ip=ip).output
                idxs.append(idx)
                cb = W(f"mimi.enc.{tag}.cb.{i}", [cb_size, cb_dim])
                quant = top.GatherOp(TT([1, T_12, cb_dim]),
                                     cb,
                                     idx,
                                     axis=0,
                                     loc=L(f"rvq.{tag}.gather{i}"),
                                     ip=ip).output  # [1,T_12,256]
                r = top.SubOp(TT([1, T_12, cb_dim]), [r, quant], loc=L(f"rvq.{tag}.sub{i}"),
                              ip=ip).output
            return idxs  # list of [1, T_12]

        sem_idxs = rvq_branch(x, "semantic", n_sem)  # 1 codebook
        aco_idxs = rvq_branch(x, "acoustic", n_aco)  # 15 codebooks
        all_idxs = sem_idxs + aco_idxs  # 16 x [1, T_12]
        codes = top.ConcatOp(TT([n_q, T_12]), all_idxs, axis=0, loc=L("rvq.concat"),
                             ip=ip).output  # [16, T_12]
        mlir.create_return_op([codes])
        self.save_mlir_module(mlir, name)

    def compile_mimi_encoder(self):
        name = "mimi_encoder"
        if self.register_bmodel(name):
            return
        # No --quant_output: the net's only output is RVQ TopK indices, which
        # must stay int32 (quant_output would bf16-quantize them -> wrong codes).
        # Compile in bf16 (half_precision_quantize); this net is ICL-only and
        # not built by default (QWEN3_TTS_ENABLE_ICL=1 to build it).
        self.submit_deploy_task(
            name,
            [f'--quantize {self.half_precision_quantize}'],
        )

    # ------------------------------------------------------------------ #
    # CodePredictor
    # ------------------------------------------------------------------ #
    def _cp_rotary_tables(self):
        # Plain 1D RoPE cos/sin for CodePredictor (no MRoPE). Sized to
        # cp_seq_length (NOT self.seq_length) — CP KV is fresh per frame and
        # only ever reaches num_code_groups deep.
        inv_freq = 1.0 / (self.cp_rope_theta**(np.arange(0, self.cp_head_dim, 2, dtype=np.float32) /
                                               self.cp_head_dim))
        t = np.arange(self.cp_seq_length, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=1)  # [seq, head_dim]
        cos = np.cos(emb).astype(np.float32).reshape(self.cp_seq_length, 1, self.cp_head_dim)
        sin = np.sin(emb).astype(np.float32).reshape(self.cp_seq_length, 1, self.cp_head_dim)
        return cos, sin

    def gen_code_predictor_mlir(self):
        tqdm.write("generate code_predictor (block_cache + 15 embed + 15 lm_head) mlir ...")
        self._gen_cp_block_cache()
        self._gen_cp_embed_nets()
        self._gen_cp_lm_head_nets()

    # ---- CP block_cache (5-layer decode block, 1D RoPE + QK-norm) ----
    def _gen_cp_block_cache(self):
        from .LlmInfo import LlmList
        layers_prefix = "talker.code_predictor.model.layers"
        norm_path = "talker.code_predictor.model.norm"
        cp_kv_heads = self.cp_num_key_value_heads
        cp_q_heads = self.cp_num_attention_heads
        cp_head_dim = self.cp_head_dim
        cp_hidden = self.cp_hidden_size
        cp_kv_dim = cp_kv_heads * cp_head_dim
        cp_q_dim = cp_q_heads * cp_head_dim

        for idx in range(self.cp_num_layers):
            name = f"cp_block_cache_{idx}"
            top_path = f"{layers_prefix}.{idx}."
            weight_file = f"cp_block_cache_{idx}_top_weights.npz"
            weight_dict = {}
            # rotary cos/sin (1D)
            weight_dict["rotary_cos.weight"] = self.cp_cos
            weight_dict["rotary_sin.weight"] = self.cp_sin
            # norms
            self.set_common_weight(top_path + "input_layernorm", weight_dict, self.rmsnorm_type)
            self.set_common_weight(top_path + "post_attention_layernorm", weight_dict,
                                   self.rmsnorm_type)
            # qk_proj + qk_norm
            self.set_linear_weight(top_path + "self_attn.q_proj", weight_dict)
            self.set_linear_weight(top_path + "self_attn.k_proj", weight_dict)
            self.set_linear_weight(top_path + "self_attn.v_proj", weight_dict)
            self.set_linear_weight(top_path + "self_attn.o_proj", weight_dict)
            self.set_common_weight(top_path + "self_attn.q_norm", weight_dict, self.rmsnorm_type)
            self.set_common_weight(top_path + "self_attn.k_norm", weight_dict, self.rmsnorm_type)
            # mlp
            self.set_linear_weight(top_path + "mlp.gate_proj", weight_dict)
            self.set_linear_weight(top_path + "mlp.up_proj", weight_dict)
            self.set_linear_weight(top_path + "mlp.down_proj", weight_dict)
            # final norm on last layer
            if idx == self.cp_num_layers - 1:
                self.set_common_weight(norm_path, weight_dict, self.rmsnorm_type)
            np.savez(weight_file, **weight_dict)

            # build MLIR for a single decode step (input_len=1). CP KV history
            # is sized to cp_seq_length (2x num_code_groups), NOT self.seq_length
            # — CP KV is fresh per frame, only ever reaches num_code_groups deep.
            # The base apply_rotary_pos sizes the rotary cos/sin weight to
            # seq_length, so pass cp_seq_length explicitly (do NOT mutate
            # self.seq_length — MLIR gen runs in a thread pool under --no-debug
            # and an instance-state swap races with concurrent Talker net gen,
            # corrupting their KV history shape).
            input_len = 1
            max_kv_len = self.cp_seq_length + input_len
            input_shape = [1, input_len, cp_hidden]
            id_shape = [1, input_len]
            mask_shape = [1, 1, input_len, max_kv_len]
            hist_shape = [1, self.cp_seq_length, cp_kv_heads, cp_head_dim]
            out_shape = input_shape
            kv_shape = [1, input_len, cp_kv_heads, cp_head_dim]
            _ret_shapes = [out_shape, kv_shape, kv_shape]
            mlir = MLIRImporter([input_shape, id_shape, mask_shape, hist_shape, hist_shape],
                                _ret_shapes,
                                name,
                                self.platform,
                                input_types=["F32", "INT32", "F32", "F32", "F32"],
                                weight_file=f"../{weight_file}")
            T = mlir.get_tensor_type
            L = lambda n: self.get_loc(n, mlir)
            ip = mlir.insert_point

            in0 = mlir.create_input_op(L("input_states"), 0)
            in1 = mlir.create_input_op(L("position_ids"), 1)
            in2 = mlir.create_input_op(L("attention_mask"), 2)
            in3 = mlir.create_input_op(L("history_k"), 3)
            in4 = mlir.create_input_op(L("history_v"), 4)

            ln_op = self.rms_norm(mlir, in0, top_path + "input_layernorm", eps=self.cp_rms_norm_eps)

            q_op = self.linear(mlir, top_path + "self_attn.q_proj", ln_op, [cp_hidden, cp_q_dim],
                               [1, input_len, cp_q_dim])
            k_op = self.linear(mlir, top_path + "self_attn.k_proj", ln_op, [cp_hidden, cp_kv_dim],
                               [1, input_len, cp_kv_dim])
            v_op = self.linear(mlir, top_path + "self_attn.v_proj", ln_op, [cp_hidden, cp_kv_dim],
                               [1, input_len, cp_kv_dim])

            q_op = top.ReshapeOp(T([1, input_len, cp_q_heads, cp_head_dim]),
                                 q_op,
                                 shape=[1, -1, cp_q_heads, cp_head_dim],
                                 loc=L("q_proj.reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T([1, input_len, cp_kv_heads, cp_head_dim]),
                                 k_op,
                                 shape=[1, -1, cp_kv_heads, cp_head_dim],
                                 loc=L("k_proj.reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T([1, input_len, cp_kv_heads, cp_head_dim]),
                                 v_op,
                                 shape=[1, -1, cp_kv_heads, cp_head_dim],
                                 loc=L("v_cache"),
                                 ip=ip).output
            # QK-norm
            q_op = self.rms_norm(mlir,
                                 q_op,
                                 top_path + "self_attn.q_norm",
                                 eps=self.cp_rms_norm_eps)
            k_op = self.rms_norm(mlir,
                                 k_op,
                                 top_path + "self_attn.k_norm",
                                 eps=self.cp_rms_norm_eps)

            # 1D RoPE (base class apply_rotary_pos, NOT the MRoPE override).
            # Pass seq_length=cp_seq_length so the rotary cos/sin weight matches
            # the CP KV history depth (see note above on thread safety).
            q_op, k_op = LlmConverter.apply_rotary_pos(self,
                                                       mlir,
                                                       in1,
                                                       q_op,
                                                       k_op,
                                                       "rotary_cos",
                                                       "rotary_sin",
                                                       seq_length=self.cp_seq_length)

            return_ops = [k_op, v_op]
            k_op = top.ConcatOp(T([1, max_kv_len, cp_kv_heads, cp_head_dim]), [in3, k_op],
                                axis=1,
                                only_merge=True,
                                loc=L("k_proj.concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, max_kv_len, cp_kv_heads, cp_head_dim]), [in4, v_op],
                                axis=1,
                                only_merge=True,
                                loc=L("v_proj.concat"),
                                ip=ip).output

            mask_op, mask_size = self.get_fattention_mask_op(mlir, in2)
            fa = top.FAttentionOp(T([1, input_len, cp_q_dim]),
                                  q_op,
                                  k_op,
                                  v_op,
                                  mask_op,
                                  mlir.none_op,
                                  scale=cp_head_dim**-0.5,
                                  batch=1,
                                  q_head=cp_q_heads,
                                  kv_head=cp_kv_heads,
                                  dim=cp_head_dim,
                                  mq=input_len,
                                  mk=max_kv_len,
                                  keep_dims=False,
                                  mask_size=mask_size,
                                  loc=L("cp_fattention"),
                                  ip=ip).output
            o_op = self.linear(mlir, top_path + "self_attn.o_proj", fa, [cp_q_dim, cp_hidden],
                               [1, input_len, cp_hidden])

            new_op = top.AddOp(T(input_shape), [in0, o_op], loc=L("attn_resid"), ip=ip).output

            ln2 = self.rms_norm(mlir,
                                new_op,
                                top_path + "post_attention_layernorm",
                                eps=self.cp_rms_norm_eps)
            gate = self.linear(mlir, top_path + "mlp.gate_proj", ln2,
                               [cp_hidden, self.cp_intermediate_size],
                               [1, input_len, self.cp_intermediate_size])
            up = self.linear(mlir, top_path + "mlp.up_proj", ln2,
                             [cp_hidden, self.cp_intermediate_size],
                             [1, input_len, self.cp_intermediate_size])
            act = top.SiLUOp(T([1, input_len, self.cp_intermediate_size]),
                             gate,
                             loc=L("mlp.act"),
                             ip=ip).output
            mul = top.MulOp(T([1, input_len, self.cp_intermediate_size]), [act, up],
                            loc=L("mlp.mul"),
                            ip=ip).output
            down = self.linear(mlir, top_path + "mlp.down_proj", mul,
                               [self.cp_intermediate_size, cp_hidden], [1, input_len, cp_hidden])
            new_op = top.AddOp(T(input_shape), [new_op, down], loc=L("mlp_resid"), ip=ip).output

            if idx == self.cp_num_layers - 1:
                new_op = self.rms_norm(mlir, new_op, norm_path, eps=self.cp_rms_norm_eps)

            mlir.create_return_op([new_op] + return_ops)
            self.save_mlir_module(mlir, name)

    # ---- CP 15 embed nets (Gather, input_ids[1,1] -> [1,1,1024]) ----
    def _gen_cp_embed_nets(self):
        for j in range(self.cp_num_code_groups - 1):
            name = f"cp_embed_{j}"
            path = f"talker.code_predictor.model.codec_embedding.{j}.weight"
            w = self.model.read(path)  # (2048, 1024)
            npz = f"cp_embed_{j}_top_weights.npz"
            np.savez(npz, **{path: w})
            out_shape = [1, 1, self.cp_hidden_size]
            mlir = MLIRImporter([[1, 1]], [out_shape],
                                name,
                                self.platform,
                                input_types=["INT32"],
                                weight_file=f"../{npz}")
            ip = mlir.insert_point
            ids = mlir.create_input_op(self.get_loc("input_ids", mlir), 0)
            wop = mlir.create_weight_op(path, [self.cp_vocab_size, self.cp_hidden_size])
            x = top.GatherOp(mlir.get_tensor_type(out_shape),
                             wop,
                             ids,
                             axis=0,
                             loc=self.get_loc(name, mlir),
                             ip=ip).output
            mlir.create_return_op([x])
            self.save_mlir_module(mlir, name)

    # ---- CP 15 lm_head nets (hidden[1,1024] -> logits[1,2048]) ----
    def _gen_cp_lm_head_nets(self):
        for j in range(self.cp_num_code_groups - 1):
            name = f"cp_lm_head_{j}"
            path = f"talker.code_predictor.lm_head.{j}.weight"
            w = self.model.read(path)  # (2048, 1024)
            npz = f"cp_lm_head_{j}_top_weights.npz"
            np.savez(npz, **{path: w})
            mlir = MLIRImporter([[1, self.cp_hidden_size]], [[1, self.cp_vocab_size]],
                                name,
                                self.platform,
                                weight_file=f"../{npz}")
            ip = mlir.insert_point
            inp = mlir.create_input_op(self.get_loc("hidden_states", mlir), 0)
            wop = mlir.create_weight_op(path, [self.cp_vocab_size, self.cp_hidden_size])
            x = top.MatMulOp(mlir.get_tensor_type([self.cp_vocab_size, 1]),
                             wop,
                             inp,
                             mlir.none_op,
                             do_relu=False,
                             right_transpose=True,
                             loc=self.get_loc(name, mlir),
                             ip=ip).output
            x = top.ReshapeOp(mlir.get_tensor_type([1, self.cp_vocab_size]),
                              x,
                              loc=self.get_loc(name + ".reshape", mlir),
                              ip=ip).output
            mlir.create_return_op([x])
            self.save_mlir_module(mlir, name)

    # ---- compile CodePredictor nets ----
    def compile_code_predictor(self):
        # 5 block_cache nets. FAttention uses the high-precision (v2) kernel.
        for idx in range(self.cp_num_layers):
            name = f"cp_block_cache_{idx}"
            if self.register_bmodel(name, with_size=True):
                continue
            extra = [f'--quantize {self.quantize}', '--quant_input', '--quant_output']
            self.submit_deploy_task(name,
                                    extra,
                                    symmetric=self.symmetric,
                                    dynamic=False,
                                    addr_mode='basic')
        # 15 embed nets (small lookup -> with_size=False, shared weights)
        for j in range(self.cp_num_code_groups - 1):
            name = f"cp_embed_{j}"
            if self.register_bmodel(name, with_size=False):
                continue
            self.submit_deploy_task(
                name, [f'--quantize {self.quantize}', '--quant_input', '--quant_output'],
                symmetric=self.symmetric,
                dynamic=False,
                addr_mode='basic')
        # 15 lm_head nets
        for j in range(self.cp_num_code_groups - 1):
            name = f"cp_lm_head_{j}"
            if self.register_bmodel(name, with_size=True):
                continue
            self.submit_deploy_task(
                name, [f'--quantize {self.quantize}', '--quant_input', '--quant_output'],
                symmetric=self.symmetric,
                dynamic=False,
                addr_mode='basic')

    def compile_codec_embedding(self):
        # Talker codec_embedding / codec_embedding_cache: lookup of the
        # 3072x1024 table. Same shape pattern as the text embedding nets
        # (max_input_length multi-token + single-token cache). The base
        # compile_all only registers "embedding"/"embedding_cache", so these
        # must be registered+deployed here, otherwise the combined bmodel is
        # missing the codec-embedding nets that the decode step and the prompt
        # assembly both need.
        for name, with_size in (("codec_embedding", True), ("codec_embedding_cache", False)):
            if self.register_bmodel(name, with_size=with_size):
                continue
            self.submit_deploy_task(
                name, [f'--quantize {self.quantize}', '--quant_input', '--quant_output'],
                symmetric=self.symmetric,
                dynamic=False,
                addr_mode='basic')

    def compile_all(self):
        # Inject codec_embedding compiles before the base class runs the
        # all_compiles queue (base compile_all does not know about them).
        self.all_compiles.append(self.compile_codec_embedding)
        super().compile_all()
