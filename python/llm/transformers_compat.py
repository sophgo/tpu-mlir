# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Pure-Python replacements for the small subset of ``transformers``
functionality used by the LLM conversion pipeline.

The compiler only relies on ``transformers`` for two things:

1. *Loading model configs* -- ``AutoConfig.from_pretrained`` and the per-model
   ``XxxConfig`` classes.  These objects are consumed exclusively through
   attribute access, the ``in`` operator and ``.get()``, so a recursive
   attribute-access ``dict`` (``Config``) is a drop-in replacement.
2. *A handful of rotary-embedding classes* (``LlamaRotaryEmbedding``,
   ``Qwen2VLRotaryEmbedding``, ``VisionRotaryEmbedding``,
   ``Glm4vText/VisionRotaryEmbedding``, ``Qwen3_5TextRotaryEmbedding``) plus
   ``ROPE_INIT_FUNCTIONS``.  The forward pass of every one of these reduces,
   for the *identical-position* table that the converters generate, to the
   standard RoPE formula implemented in :func:`text_rotary_cos_sin`.

The RoPE math here is a line-for-line port of ``transformers``
``modeling_rope_utils.ROPE_INIT_FUNCTIONS`` and has been validated against the
installed ``transformers`` to float32 precision for every rope type in use
(default, linear, yarn, longrope, llama3, proportional and the mrope variant
which has no init function and therefore behaves like ``default``).
"""

import json
import math
import os
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Config object
# --------------------------------------------------------------------------- #


class Config(dict):
    """A ``dict`` subclass that supports recursive attribute access.

    Nested dicts (and lists of dicts) are wrapped on construction so that
    ``config.text_config.rope_theta`` works just like on a ``transformers``
    config object.  Attribute writes are proxied to dict items, which is
    required by :class:`LlmConverter.init_config` (it does
    ``self.llm_config.max_position_embeddings = seq_length``).
    """

    def __init__(self, data=None, **kwargs):
        super().__init__()
        if data is None:
            data = {}
        if isinstance(data, Config):
            data = dict(data)
        for key, value in data.items():
            self[key] = self._wrap(value)
        for key, value in kwargs.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value):
        if isinstance(value, Config):
            return value
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config(v) if isinstance(v, dict) else v for v in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def update(self, *args, **kwargs):
        """Override ``update`` so newly inserted nested dicts stay wrapped."""
        other = {}
        if args:
            other.update(args[0])
        other.update(kwargs)
        for key, value in other.items():
            self[key] = self._wrap(value)

    def to_dict(self):
        """Return a plain (recursively unwrapped) ``dict``."""

        def unwrap(value):
            if isinstance(value, Config):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, dict):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return unwrap(self)


# --------------------------------------------------------------------------- #
# Vision-config class defaults
# --------------------------------------------------------------------------- #
# The Qwen-VL / mllama / glm4v families ship a *minimal* ``vision_config`` in
# ``config.json`` -- only the fields that differ from the transformers vision
# config class are serialized.  ``AutoConfig.from_pretrained`` reconstructs the
# full object by instantiating the class, which fills in the class-level
# defaults; a plain dict (our ``Config``) does not.  This table replicates those
# defaults so attribute access (``vconfig.patch_size`` etc.) behaves identically.
# Values mirror transformers 5.6.2 class attributes (verified by instantiating
# each ``XxxVisionConfig()`` with no args).  Only keys *missing* from the raw
# ``vision_config`` are filled -- on-disk values always win.
_VISION_CONFIG_DEFAULTS = {
    "qwen2_vl": {
        "depth": 32,
        "embed_dim": 1280,
        "hidden_size": 3584,
        "hidden_act": "quick_gelu",
        "mlp_ratio": 4,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "initializer_range": 0.02,
    },
    "qwen2_5_vl": {
        "depth": 32,
        "hidden_size": 3584,
        "hidden_act": "silu",
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "tokens_per_second": 4,
        "window_size": 112,
        "out_hidden_size": 3584,
        "fullatt_block_indexes": (7, 15, 23, 31),
        "initializer_range": 0.02,
    },
    "qwen3_vl": {
        "depth": 27,
        "hidden_size": 1152,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 3584,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": (8, 16, 24),
        "initializer_range": 0.02,
    },
    # Qwen3-VL-MoE shares the same vision config defaults as Qwen3-VL.
    "qwen3_vl_moe": {
        "depth": 27,
        "hidden_size": 1152,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 3584,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": (8, 16, 24),
        "initializer_range": 0.02,
    },
    "qwen3_5": {
        "depth": 27,
        "hidden_size": 1152,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 3584,
        "num_position_embeddings": 2304,
        "initializer_range": 0.02,
    },
    # Qwen3.5-MoE shares the same vision config defaults as Qwen3.5.
    "qwen3_5_moe": {
        "depth": 27,
        "hidden_size": 1152,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 3584,
        "num_position_embeddings": 2304,
        "initializer_range": 0.02,
    },
    "mllama": {
        "hidden_size": 1280,
        "hidden_act": "gelu",
        "num_hidden_layers": 32,
        "num_global_layers": 8,
        "attention_heads": 16,
        "num_channels": 3,
        "intermediate_size": 5120,
        "vision_output_dim": 7680,
        "image_size": 448,
        "patch_size": 14,
        "norm_eps": 1e-5,
        "max_num_tiles": 4,
        "intermediate_layers_indices": [3, 7, 15, 23, 30],
        "supported_aspect_ratios": [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]],
        "initializer_range": 0.02,
    },
    "glm4v": {
        "depth": 24,
        "hidden_size": 1536,
        "hidden_act": "silu",
        "attention_bias": False,
        "num_heads": 12,
        "in_channels": 3,
        "image_size": 336,
        "patch_size": 14,
        "rms_norm_eps": 1e-5,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 4096,
        "intermediate_size": 13696,
        "initializer_range": 0.02,
    },
    "minicpmv4_6": {
        "insert_layer_id": 6,
        "window_kernel_size": (2, 2),
    },
}


def _apply_vision_defaults(data):
    """Fill in transformers vision-config class defaults for ``data``.

    ``data`` is the raw ``config.json`` dict.  If ``data["model_type"]`` is a
    known VL family and ``data`` contains a ``vision_config`` dict, every
    default key that is *absent* from the on-disk ``vision_config`` is inserted.
    On-disk values always take precedence (matching ``AutoConfig`` behaviour).
    """
    model_type = data.get("model_type")
    defaults = _VISION_CONFIG_DEFAULTS.get(model_type)
    if not defaults or not isinstance(data.get("vision_config"), dict):
        return
    vcfg = data["vision_config"]
    for key, value in defaults.items():
        if key not in vcfg:
            vcfg[key] = value


# --------------------------------------------------------------------------- #
# Top-level config defaults
# --------------------------------------------------------------------------- #
# Some VL models have class-level defaults at the top level of the config
# (not inside vision_config) that are not serialized in config.json.
# AutoConfig fills these from the class __init__ defaults; we replicate them.
_CONFIG_DEFAULTS = {
    "minicpmv4_6": {
        "merge_kernel_size": (2, 2),
        "merger_times": 1,
    },
}


def _apply_config_defaults(data):
    """Fill in transformers top-level config class defaults for ``data``."""
    model_type = data.get("model_type")
    defaults = _CONFIG_DEFAULTS.get(model_type)
    if not defaults:
        return
    for key, value in defaults.items():
        if key not in data:
            data[key] = value


def load_auto_config(model_path, trust_remote_code=True, **kwargs):
    """Read ``config.json`` directly and return a :class:`Config`.

    Prefers ``transformers.AutoConfig.from_pretrained`` when ``transformers``
    is importable, so that class-level defaults for *both* ``text_config`` and
    ``vision_config`` (e.g. Gemma3's ``num_attention_heads``/``head_dim``/
    ``rms_norm_eps`` which are absent from the on-disk minimal config) are
    filled in natively -- exactly as a real ``AutoConfig`` load would.  The
    resulting object is flattened with ``to_dict()`` so downstream code still
    sees a plain (nested) dict.  Falls back to a direct ``json.load`` when
    ``transformers`` is unavailable or the load raises.
    """
    config_path = model_path if os.path.isfile(model_path) else os.path.join(
        model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found for model path '{model_path}'")

    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:  # pragma: no cover - transformers optional
        AutoConfig = None

    if AutoConfig is not None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            return cfg.to_dict()
        except Exception:
            # Fall through to the plain-json path below.
            pass

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_auto_config(model_path, trust_remote_code=True, **kwargs):
    """Read ``config.json`` and return a :class:`Config`.

    Replaces ``transformers.AutoConfig.from_pretrained``.  When ``transformers``
    is installed, the config is loaded through ``AutoConfig`` first so that
    class-level defaults for nested sub-configs (``text_config`` /
    ``vision_config``) are filled in natively; this matters for families such
    as Gemma3 whose on-disk ``text_config`` is minimal and relies on
    ``Gemma3TextConfig`` defaults (``num_attention_heads``, ``head_dim``,
    ``rms_norm_eps``, ...).  Otherwise the raw ``config.json`` dict is read
    directly and :func:`_apply_vision_defaults` replicates the vision-config
    defaults for known VL families.  ``trust_remote_code`` is forwarded to
    ``AutoConfig`` when used and otherwise accepted for API compatibility.
    """
    del kwargs  # accepted for drop-in compatibility
    data = _load_config_data(model_path)
    _apply_vision_defaults(data)
    _apply_config_defaults(data)
    return Config(data)


# --------------------------------------------------------------------------- #
# RoPE helpers
# --------------------------------------------------------------------------- #


def build_rope_parameters(config):
    """Build the standardized ``rope_parameters`` dict from a config.

    This mirrors ``PretrainedConfig.standardize_rope_params``: it merges the
    top-level ``rope_theta`` / ``partial_rotary_factor`` and the
    ``rope_scaling`` (or existing ``rope_parameters``) dict, derives the
    ``rope_type`` and, for the scaling variants that need it, the
    ``original_max_position_embeddings``.
    """
    rope_theta = _get(config, "rope_theta", None)
    partial_rotary_factor = _get(config, "partial_rotary_factor", None)

    rope_parameters = _get(config, "rope_parameters", None)
    if rope_parameters:
        params = dict(rope_parameters)
    else:
        rope_scaling = _get(config, "rope_scaling", None)
        params = dict(rope_scaling) if rope_scaling else {}

    if not params and rope_theta is None:
        return {"rope_type": "default", "rope_theta": 10000.0}

    params.setdefault("rope_type", params.get("type", "default"))
    if "type" in params and "rope_type" not in params:
        params["rope_type"] = params["type"]
    if rope_theta is not None:
        params.setdefault("rope_theta", rope_theta)
    if partial_rotary_factor is not None:
        params["partial_rotary_factor"] = partial_rotary_factor

    if params["rope_type"] in ("llama3", "yarn", "longrope"):
        ompe = _get(config, "original_max_position_embeddings", None)
        if ompe is not None:
            params["original_max_position_embeddings"] = ompe
        else:
            params.setdefault("original_max_position_embeddings",
                              _get(config, "max_position_embeddings", 2048))

    return params


def _get(obj, key, default=None):
    """``getattr``-style access that works on dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compute_inv_freq_and_scaling(config, seq_len=None):
    """Compute ``(inv_freq, attention_scaling)`` for a config.

    Port of ``ROPE_INIT_FUNCTIONS``.  ``inv_freq`` is a 1-D ``numpy`` array of
    length ``dim // 2`` where ``dim = int(head_dim * partial_rotary_factor)``.
    The ``mrope`` rope type has no init function in ``transformers`` (it would
    raise ``KeyError``); the models that use it set ``rope_type='default'`` on
    the config, so it is handled by the default branch, which matches the
    intended behaviour (standard inverse frequencies, no scaling).
    """
    params = build_rope_parameters(config)
    rope_type = params.get("rope_type", params.get("type", "default"))
    base = params.get("rope_theta", 10000.0)
    partial_rotary_factor = params.get("partial_rotary_factor", 1.0)

    head_dim = _get(config, "head_dim", None)
    if not head_dim:
        hidden_size = _get(config, "hidden_size", 0)
        num_heads = _get(config, "num_attention_heads", 1) or 1
        head_dim = hidden_size // num_heads
    dim = int(head_dim * partial_rotary_factor)
    max_pos = _get(config, "max_position_embeddings", 2048)

    if rope_type in ("default", "mrope"):
        inv_freq = 1.0 / (base**(np.arange(0, dim, 2, dtype=np.float32) / dim))
        return inv_freq, 1.0

    if rope_type == "linear":
        inv_freq = 1.0 / (base**(np.arange(0, dim, 2, dtype=np.float32) / dim))
        inv_freq = inv_freq / params["factor"]
        return inv_freq, 1.0

    if rope_type == "llama3":
        inv_freq = 1.0 / (base**(np.arange(0, dim, 2, dtype=np.float32) / dim))
        factor = params["factor"]
        low_freq_factor = params["low_freq_factor"]
        high_freq_factor = params["high_freq_factor"]
        old_ctx = params["original_max_position_embeddings"]
        low_freq_wavelen = old_ctx / low_freq_factor
        high_freq_wavelen = old_ctx / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + \
            smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama, 1.0

    if rope_type == "longrope":
        long_factor = params["long_factor"]
        short_factor = params["short_factor"]
        original_max = params["original_max_position_embeddings"]
        factor = params.get("factor")
        if factor is None:
            factor = max_pos / original_max
        attention_factor = params.get("attention_factor")
        if attention_factor is None:
            if factor <= 1.0:
                attention_factor = 1.0
            else:
                attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max))
        ext_factors = np.asarray(
            long_factor if (seq_len and seq_len > original_max) else short_factor,
            dtype=np.float32,
        )
        inv_freq_shape = np.arange(0, dim, 2, dtype=np.float32) / dim
        inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
        return inv_freq, attention_factor

    if rope_type == "yarn":
        original_max = params["original_max_position_embeddings"]
        factor = params.get("factor")
        if factor is None:
            factor = max_pos / original_max
        attention_factor = params.get("attention_factor")
        mscale = params.get("mscale")
        mscale_all_dim = params.get("mscale_all_dim")

        def _get_mscale(scale, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        if attention_factor is None:
            if mscale and mscale_all_dim:
                attention_factor = float(
                    _get_mscale(factor, mscale) / _get_mscale(factor, mscale_all_dim))
            else:
                attention_factor = _get_mscale(factor)

        beta_fast = params.get("beta_fast") or 32
        beta_slow = params.get("beta_slow") or 1
        truncate = params.get("truncate", True)

        def _find_correction_dim(num_rotations, dim, base, max_position):
            return (dim * math.log(
                max_position / (num_rotations * 2 * math.pi))) / \
                (2 * math.log(base))

        def _find_correction_range(low_rot, high_rot, dim, base, max_position, truncate):
            low = _find_correction_dim(low_rot, dim, base, max_position)
            high = _find_correction_dim(high_rot, dim, base, max_position)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dim - 1)

        def _linear_ramp_factor(low, high, dim):
            if low == high:
                high += 0.001
            linear_func = (np.arange(dim, dtype=np.float32) - low) / (high - low)
            return np.clip(linear_func, 0, 1)

        pos_freqs = base**(np.arange(0, dim, 2, dtype=np.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        low, high = _find_correction_range(beta_fast, beta_slow, dim, base, original_max, truncate)
        inv_freq_extrapolation_factor = 1 - _linear_ramp_factor(low, high, dim // 2)
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_extrapolation_factor) + \
            inv_freq_extrapolation * inv_freq_extrapolation_factor
        return inv_freq, attention_factor

    if rope_type == "proportional":
        factor = params.get("factor", 1.0)
        rope_proportion = params.get("partial_rotary_factor", 1.0)
        rope_angles = int(rope_proportion * head_dim // 2)
        inv_freq_rotated = 1.0 / (base
                                  **(np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = np.concatenate([inv_freq_rotated, np.zeros(nope_angles, dtype=np.float32)])
        else:
            inv_freq = inv_freq_rotated
        inv_freq = inv_freq / factor
        return inv_freq, 1.0

    # Unknown rope type: fall back to default behaviour.
    inv_freq = 1.0 / (base**(np.arange(0, dim, 2, dtype=np.float32) / dim))
    return inv_freq, 1.0


def text_rotary_cos_sin(config, seq_length):
    """Compute RoPE ``(cos, sin)`` for a 1-D position sequence.

    Returns two ``numpy`` arrays of shape ``(seq_length, 1, dim)`` where
    ``dim = int(head_dim * partial_rotary_factor)`` and ``emb`` is built as
    ``cat(freqs, freqs)`` exactly like ``LlamaRotaryEmbedding.forward``.  The
    result is the *full* (un-halved) table; callers that need the halved form
    (the vision-language converters) slice ``[..., :dim // 2]`` themselves.

    This replaces ``LlamaRotaryEmbedding``, ``Qwen2VLRotaryEmbedding``,
    ``Glm4vTextRotaryEmbedding`` and ``Qwen3_5TextRotaryEmbedding`` for the
    identical-position tables the converters generate: with ``t == h == w``
    every mrope variant (``apply_mrope`` / ``apply_interleaved_mrope``) is a
    no-op, so the result coincides with plain 1-D RoPE.
    """
    inv_freq, attention_scaling = compute_inv_freq_and_scaling(config, seq_len=seq_length)
    t = np.arange(seq_length, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # [seq, dim//2]
    emb = np.concatenate([freqs, freqs], axis=1)  # [seq, dim]
    cos = (np.cos(emb) * attention_scaling).astype(np.float32)
    sin = (np.sin(emb) * attention_scaling).astype(np.float32)
    return cos.reshape(seq_length, 1, -1), sin.reshape(seq_length, 1, -1)


def vision_rotary_cos_sin(dim, seqlen, theta=10000.0):
    """Compute RoPE ``(cos, sin)`` for a vision tower.

    Replaces ``VisionRotaryEmbedding`` / ``Glm4vVisionRotaryEmbedding``.  The
    transformers classes return ``outer(arange(seqlen), inv_freq)`` of shape
    ``(seqlen, dim)`` and the callers do ``freqs.cos()`` / ``freqs.sin()``;
    here we return the final ``(cos, sin)`` arrays directly.
    """
    inv_freq = 1.0 / (theta**(np.arange(0, dim, 2, dtype=np.float32) / dim))
    seq = np.arange(seqlen, dtype=np.float32)
    freqs = np.outer(seq, inv_freq)  # [seqlen, dim]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)
