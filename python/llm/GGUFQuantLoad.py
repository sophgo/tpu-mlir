# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import re
import threading
import numpy as np
import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Add gguf-py to path

from gguf import GGUFReader, GGMLQuantizationType, ReaderTensor

# ---------------------------------------------------------------------------
# Per-architecture name mapping declarations
# ---------------------------------------------------------------------------

ComponentSpec = namedtuple("ComponentSpec",
                           ["gguf_name", "gguf_bias_name", "has_weight_suffix", "has_bias"])

NS_PREFIXES = {
    "standard": ("model.layers.{N}", "blk.{N}"),
    "vlm": ("model.language_model.layers.{N}", "blk.{N}"),
    "internvl": ("language_model.model.layers.{N}", "blk.{N}"),
}

BASE_LLM_COMPONENTS = {
    "input_layernorm": ComponentSpec("attn_norm.weight", None, True, False),
    "self_attn.q_proj": ComponentSpec("attn_q.weight", "attn_q.bias", True, True),
    "self_attn.k_proj": ComponentSpec("attn_k.weight", "attn_k.bias", True, True),
    "self_attn.v_proj": ComponentSpec("attn_v.weight", "attn_v.bias", True, True),
    "self_attn.o_proj": ComponentSpec("attn_output.weight", "attn_output.bias", True, True),
    "post_attention_layernorm": ComponentSpec("ffn_norm.weight", None, True, False),
    "mlp.gate_proj": ComponentSpec("ffn_gate.weight", "ffn_gate.bias", True, True),
    "mlp.up_proj": ComponentSpec("ffn_up.weight", "ffn_up.bias", True, True),
    "mlp.down_proj": ComponentSpec("ffn_down.weight", "ffn_down.bias", True, True),
}

ARCH_NAME_MAPS = {
    "llama": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "llama3": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "qwen2": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "qwen2_moe": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "qwen3": {
        "prefixes": ["standard", "vlm"],
        "components": {
            **BASE_LLM_COMPONENTS,
            "self_attn.q_norm":
            ComponentSpec("attn_q_norm.weight", None, True, False),
            "self_attn.k_norm":
            ComponentSpec("attn_k_norm.weight", None, True, False),
        },
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.language_model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "model.language_model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "qwen3vl": {
        "prefixes": ["standard", "vlm"],
        "components": {
            **BASE_LLM_COMPONENTS,
            "self_attn.q_norm":
            ComponentSpec("attn_q_norm.weight", None, True, False),
            "self_attn.k_norm":
            ComponentSpec("attn_k_norm.weight", None, True, False),
        },
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.language_model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "model.language_model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
        "vision_flat_prefix": "model.visual",
        "vision_flat": {
            "patch_embed.proj": ComponentSpec("v.patch_embd.weight", "v.patch_embd.bias", True,
                                              True),
            "pos_embed": ComponentSpec("v.position_embd.weight", None, True, False),
            "post_ln": ComponentSpec("v.post_ln.weight", "v.post_ln.bias", True, True),
            "merger.norm": ComponentSpec("v.post_ln.weight", "v.post_ln.bias", True, True),
            "merger.linear_fc1": ComponentSpec("mm.0.weight", "mm.0.bias", True, True),
            "merger.linear_fc2": ComponentSpec("mm.2.weight", "mm.2.bias", True, True),
        },
        "vision_block_prefix": ("model.visual.blocks.{N}", "v.blk.{N}"),
        "vision_blocks": {
            "norm1": ComponentSpec("ln1.weight", "ln1.bias", True, True),
            "norm2": ComponentSpec("ln2.weight", "ln2.bias", True, True),
            "attn.qkv": ComponentSpec("attn_qkv.weight", "attn_qkv.bias", True, True),
            "attn.proj": ComponentSpec("attn_out.weight", "attn_out.bias", True, True),
            "mlp.linear_fc1": ComponentSpec("ffn_up.weight", "ffn_up.bias", True, True),
            "mlp.linear_fc2": ComponentSpec("ffn_down.weight", "ffn_down.bias", True, True),
        },
        "vision_deepstack": True,
    },
    "qwen35": {
        "prefixes": ["standard", "vlm"],
        "components": {
            **BASE_LLM_COMPONENTS,
            "post_attention_layernorm":
            ComponentSpec("post_attention_norm.weight", None, True, False),
            "self_attn.q_norm":
            ComponentSpec("attn_q_norm.weight", None, True, False),
            "self_attn.k_norm":
            ComponentSpec("attn_k_norm.weight", None, True, False),
            "linear_attn.in_proj_a":
            ComponentSpec("ssm_alpha.weight", None, True, False),
            "linear_attn.in_proj_a.bias":
            ComponentSpec("ssm_dt.bias", None, False, False),
            "linear_attn.in_proj_b":
            ComponentSpec("ssm_beta.weight", None, True, False),
            "linear_attn.in_proj_qkv":
            ComponentSpec("attn_qkv.weight", None, True, False),
            "linear_attn.in_proj_z":
            ComponentSpec("attn_gate.weight", None, True, False),
            "linear_attn.out_proj":
            ComponentSpec("ssm_out.weight", None, True, False),
            "linear_attn.norm":
            ComponentSpec("ssm_norm.weight", None, True, False),
            "linear_attn.conv1d":
            ComponentSpec("ssm_conv1d.weight", None, True, False),
            "linear_attn.dt_bias":
            ComponentSpec("ssm_dt.bias", None, True, False),
            "linear_attn.A_log":
            ComponentSpec("ssm_a", None, True, False),
        },
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.language_model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "model.language_model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
        "vision_flat_prefix": "model.visual",
        "vision_flat": {
            "patch_embed.proj": ComponentSpec("v.patch_embd.weight", "v.patch_embd.bias", True,
                                              True),
            "pos_embed": ComponentSpec("v.position_embd.weight", None, True, False),
            "post_ln": ComponentSpec("v.post_ln.weight", "v.post_ln.bias", True, True),
            "merger.norm": ComponentSpec("v.post_ln.weight", "v.post_ln.bias", True, True),
            "merger.linear_fc1": ComponentSpec("mm.0.weight", "mm.0.bias", True, True),
            "merger.linear_fc2": ComponentSpec("mm.2.weight", "mm.2.bias", True, True),
        },
        "vision_block_prefix": ("model.visual.blocks.{N}", "v.blk.{N}"),
        "vision_blocks": {
            "norm1": ComponentSpec("ln1.weight", "ln1.bias", True, True),
            "norm2": ComponentSpec("ln2.weight", "ln2.bias", True, True),
            "attn.qkv": ComponentSpec("attn_qkv.weight", "attn_qkv.bias", True, True),
            "attn.proj": ComponentSpec("attn_out.weight", "attn_out.bias", True, True),
            "mlp.linear_fc1": ComponentSpec("ffn_up.weight", "ffn_up.bias", True, True),
            "mlp.linear_fc2": ComponentSpec("ffn_down.weight", "ffn_down.bias", True, True),
        },
        "vision_deepstack": True,
    },
    "internvl_chat": {
        "prefixes": ["internvl"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "language_model.model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "language_model.model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "language_model.lm_head": ComponentSpec("output.weight", None, True, False),
        },
        "vision_flat_prefix": "vision_model",
        "vision_flat": {
            "embeddings.class_embedding":
            ComponentSpec("v.class_embd", None, True, False),
            "embeddings.patch_embedding":
            ComponentSpec("v.patch_embd.weight", "v.patch_embd.bias", True, True),
            "embeddings.position_embedding":
            ComponentSpec("__POS_EMBD_TRANSPOSE__", None, True, False),
        },
        "vision_block_prefix": ("vision_model.encoder.layers.{N}", "v.blk.{N}"),
        "vision_blocks": {
            "norm1": ComponentSpec("ln1.weight", "ln1.bias", True, True),
            "norm2": ComponentSpec("ln2.weight", "ln2.bias", True, True),
            "ls1": ComponentSpec("ls1.weight", None, True, False),
            "ls2": ComponentSpec("ls2.weight", None, True, False),
            "attn.qkv": ComponentSpec("__QKV_MERGE__:{N}", None, True, False),
            "attn.qkv.bias": ComponentSpec("__QKV_MERGE_BIAS__:{N}", None, False, False),
            "attn.proj": ComponentSpec("attn_out.weight", "attn_out.bias", True, True),
            "mlp.fc1": ComponentSpec("ffn_up.weight", "ffn_up.bias", True, True),
            "mlp.fc2": ComponentSpec("ffn_down.weight", "ffn_down.bias", True, True),
        },
        "vision_projector": {
            "mlp1.0": ComponentSpec("mm.model.mlp.0.weight", "mm.model.mlp.0.bias", True, True),
            "mlp1.1": ComponentSpec("mm.model.mlp.1.weight", "mm.model.mlp.1.bias", True, True),
            "mlp1.3": ComponentSpec("mm.model.mlp.3.weight", "mm.model.mlp.3.bias", True, True),
        },
    },
    "clip": {
        "prefixes": ["internvl"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "language_model.model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "language_model.model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "language_model.lm_head": ComponentSpec("output.weight", None, True, False),
        },
        "vision_flat_prefix": "vision_model",
        "vision_flat": {
            "embeddings.class_embedding":
            ComponentSpec("v.class_embd", None, True, False),
            "embeddings.patch_embedding":
            ComponentSpec("v.patch_embd.weight", "v.patch_embd.bias", True, True),
            "embeddings.position_embedding":
            ComponentSpec("__POS_EMBD_TRANSPOSE__", None, True, False),
        },
        "vision_block_prefix": ("vision_model.encoder.layers.{N}", "v.blk.{N}"),
        "vision_blocks": {
            "norm1": ComponentSpec("ln1.weight", "ln1.bias", True, True),
            "norm2": ComponentSpec("ln2.weight", "ln2.bias", True, True),
            "ls1": ComponentSpec("ls1.weight", None, True, False),
            "ls2": ComponentSpec("ls2.weight", None, True, False),
            "attn.qkv": ComponentSpec("__QKV_MERGE__:{N}", None, True, False),
            "attn.qkv.bias": ComponentSpec("__QKV_MERGE_BIAS__:{N}", None, False, False),
            "attn.proj": ComponentSpec("attn_out.weight", "attn_out.bias", True, True),
            "mlp.fc1": ComponentSpec("ffn_up.weight", "ffn_up.bias", True, True),
            "mlp.fc2": ComponentSpec("ffn_down.weight", "ffn_down.bias", True, True),
        },
        "vision_projector": {
            "mlp1.0": ComponentSpec("mm.model.mlp.0.weight", "mm.model.mlp.0.bias", True, True),
            "mlp1.1": ComponentSpec("mm.model.mlp.1.weight", "mm.model.mlp.1.bias", True, True),
            "mlp1.3": ComponentSpec("mm.model.mlp.3.weight", "mm.model.mlp.3.bias", True, True),
        },
    },
    "gemma": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "gemma2": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "chatglm": {
        "prefixes": ["standard"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "model.embed_tokens": ComponentSpec("token_embd.weight", None, True, False),
            "model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "lm_head": ComponentSpec("token_embd.weight", None, True, False),
        },
    },
    "mllama": {
        "prefixes": ["internvl"],
        "components": BASE_LLM_COMPONENTS,
        "top_level": {
            "language_model.model.embed_tokens": ComponentSpec("token_embd.weight", None, True,
                                                               False),
            "language_model.model.norm": ComponentSpec("output_norm.weight", None, True, False),
            "language_model.lm_head": ComponentSpec("output.weight", None, True, False),
        },
    },
}


def _build_regex_map(arch_entry):
    """Build compiled regex pattern list from per-architecture mapping data.

    Returns a list of (compiled_regex, gguf_template) tuples, ordered so that
    more specific patterns (bias) come before less specific ones (weight).
    """
    patterns = []

    def _add(llm_key_regex, gguf_template):
        patterns.append((re.compile(llm_key_regex), gguf_template))

    def _prefix_to_regex(prefix_template):
        return re.escape(prefix_template.replace("{N}", "_PLACEHOLDER_N_")).replace(
            "_PLACEHOLDER_N_", "(\\d+)")

    # 1. Top-level entries
    for llm_key, spec in arch_entry.get("top_level", {}).items():
        llm_regex = re.escape(llm_key)
        if spec.has_bias and spec.gguf_bias_name:
            _add(llm_regex + "\\.bias$", spec.gguf_bias_name)
        if spec.has_weight_suffix:
            _add(llm_regex + "(?:\\.weight)?$", spec.gguf_name)
        else:
            _add(llm_regex + "$", spec.gguf_name)

    # 2. Layer component entries (per namespace prefix)
    for prefix_label in arch_entry.get("prefixes", []):
        llm_prefix_t, gguf_prefix_t = NS_PREFIXES[prefix_label]
        llm_prefix_r = _prefix_to_regex(llm_prefix_t)

        for llm_sub, spec in arch_entry.get("components", {}).items():
            llm_sub_r = re.escape(llm_sub)
            full_llm_r = llm_prefix_r + "\\." + llm_sub_r

            if spec.gguf_name.startswith("__"):
                gguf_tmpl = spec.gguf_name
            else:
                gguf_tmpl = gguf_prefix_t + "." + spec.gguf_name

            if spec.has_bias and spec.gguf_bias_name and not llm_sub.endswith(".bias"):
                if spec.gguf_bias_name.startswith("__"):
                    bias_gguf = spec.gguf_bias_name
                else:
                    bias_gguf = gguf_prefix_t + "." + spec.gguf_bias_name
                _add(full_llm_r + "\\.bias$", bias_gguf)

            if spec.has_weight_suffix:
                _add(full_llm_r + "(?:\\.weight)?$", gguf_tmpl)
            else:
                _add(full_llm_r + "$", gguf_tmpl)

    # 3. Vision flat entries
    vflat_prefix = arch_entry.get("vision_flat_prefix", "")
    for llm_sub, spec in arch_entry.get("vision_flat", {}).items():
        if vflat_prefix:
            full_llm_regex = re.escape(vflat_prefix) + "\\." + re.escape(llm_sub)
        else:
            full_llm_regex = re.escape(llm_sub)
        if spec.has_bias and spec.gguf_bias_name:
            _add(full_llm_regex + "\\.bias$", spec.gguf_bias_name)
        if spec.has_weight_suffix:
            _add(full_llm_regex + "(?:\\.weight)?$", spec.gguf_name)
        else:
            _add(full_llm_regex + "$", spec.gguf_name)

    # 4. Vision block entries
    vblock_prefix = arch_entry.get("vision_block_prefix", None)
    if vblock_prefix:
        llm_vb_t, gguf_vb_t = vblock_prefix
        llm_vb_r = _prefix_to_regex(llm_vb_t)

        for llm_sub, spec in arch_entry.get("vision_blocks", {}).items():
            llm_sub_r = re.escape(llm_sub)
            full_llm_r = llm_vb_r + "\\." + llm_sub_r

            if spec.gguf_name.startswith("__"):
                gguf_tmpl = spec.gguf_name
            else:
                gguf_tmpl = gguf_vb_t + "." + spec.gguf_name

            if spec.has_bias and spec.gguf_bias_name and not llm_sub.endswith(".bias"):
                if spec.gguf_bias_name.startswith("__"):
                    bias_gguf = spec.gguf_bias_name
                else:
                    bias_gguf = gguf_vb_t + "." + spec.gguf_bias_name
                _add(full_llm_r + "\\.bias$", bias_gguf)

            if spec.has_weight_suffix:
                _add(full_llm_r + "(?:\\.weight)?$", gguf_tmpl)
            else:
                _add(full_llm_r + "$", gguf_tmpl)

    # 5. Vision projector entries
    for llm_key, spec in arch_entry.get("vision_projector", {}).items():
        llm_regex = re.escape(llm_key)
        if spec.has_bias and spec.gguf_bias_name:
            _add(llm_regex + "\\.bias$", spec.gguf_bias_name)
        if spec.has_weight_suffix:
            _add(llm_regex + "(?:\\.weight)?$", spec.gguf_name)
        else:
            _add(llm_regex + "$", spec.gguf_name)

    return patterns


class GGUFQuantLoad:
    """GGUF loader that preserves quantization information."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.reader = GGUFReader(model_path)
        self.mmproj_reader = None
        self.tensor_cache = {}
        self._cache_lock = threading.Lock()
        self.quant_info_cache = {}
        self.metadata = self._extract_metadata()

        self.tensor_map = self._build_tensor_map()

        self.arch = self.metadata.get("architecture", "")
        arch_entry = ARCH_NAME_MAPS.get(self.arch)
        if arch_entry is None:
            raise ValueError(f"GGUF architecture '{self.arch}' not supported. "
                             f"Add a mapping entry to ARCH_NAME_MAPS. "
                             f"Supported: {sorted(ARCH_NAME_MAPS.keys())}")
        self._regex_map = _build_regex_map(arch_entry)
        self._has_deepstack = arch_entry.get("vision_deepstack", False)

    def load_mmproj(self, mmproj_path: str):
        """Load mmproj GGUF file for vision models.

        Merges vision/mmproj tensors into the main tensor map and
        extracts vision config metadata for VLM conversion.
        """
        self.mmproj_reader = GGUFReader(mmproj_path)
        logger.info("Loaded mmproj GGUF: %s", mmproj_path)

        mmproj_arch = None
        arch_field = self.mmproj_reader.get_field("general.architecture")
        if arch_field:
            mmproj_arch = arch_field.contents()
        logger.info("mmproj architecture: %s", mmproj_arch)

        for tensor in self.mmproj_reader.tensors:
            self.tensor_map[tensor.name] = tensor

        mmproj_metadata = {}
        for field_name, field in self.mmproj_reader.fields.items():
            try:
                mmproj_metadata[field_name] = field.contents()
            except Exception:
                pass

        self.metadata["mmproj"] = mmproj_metadata
        logger.info("Merged %d mmproj tensors into tensor map", len(self.mmproj_reader.tensors))

        mmproj_entry = ARCH_NAME_MAPS.get(mmproj_arch) if mmproj_arch else None
        if mmproj_entry:
            vision_patterns = _build_regex_map(mmproj_entry)
            self._regex_map.extend(vision_patterns)
            if mmproj_entry.get("vision_deepstack", False):
                self._has_deepstack = True
            logger.info("Added %d vision patterns from mmproj arch '%s'", len(vision_patterns),
                        mmproj_arch)
        else:
            if mmproj_arch:
                logger.warning(
                    "mmproj architecture '%s' not in ARCH_NAME_MAPS, "
                    "vision key mapping may not work", mmproj_arch)

        # Auto-detect VLM type from mmproj tensor names when arch is "clip"
        if mmproj_arch == "clip":
            has_attn_q = any(
                t.startswith("v.blk") and "attn_q.weight" in t for t in self.tensor_map)
            if not has_attn_q:
                for vlm_arch in ["qwen3vl", "qwen35"]:
                    vlm_entry = ARCH_NAME_MAPS.get(vlm_arch)
                    if vlm_entry:
                        extra = _build_regex_map(vlm_entry)
                        self._regex_map.extend(extra)
                        if vlm_entry.get("vision_deepstack", False):
                            self._has_deepstack = True
                        logger.info("Added %d vision patterns from '%s' for clip mmproj",
                                    len(extra), vlm_arch)

    def _extract_metadata(self) -> Dict:
        """Extract model metadata from GGUF file.

        Extracts all GGUF metadata fields and maps key model parameters
        for compatibility with existing code.
        """
        metadata = {}

        for field_name, field in self.reader.fields.items():
            try:
                metadata[field_name] = field.contents()
            except Exception:
                pass

        arch_field = self.reader.get_field("general.architecture")
        if arch_field:
            metadata["architecture"] = arch_field.contents()

        arch = metadata.get("architecture", "")

        def get_arch_key(key_template: str) -> Optional[str]:
            if arch:
                return key_template.format(arch=arch)
            return None

        params = [
            ("hidden_size", get_arch_key("{arch}.embedding_length")),
            ("num_hidden_layers", get_arch_key("{arch}.block_count")),
            ("num_attention_heads", get_arch_key("{arch}.attention.head_count")),
            ("num_key_value_heads", get_arch_key("{arch}.attention.head_count_kv")),
            ("vocab_size", get_arch_key("{arch}.vocab_size")),
            ("intermediate_size", get_arch_key("{arch}.feed_forward_length")),
            ("max_position_embeddings", get_arch_key("{arch}.context_length")),
            ("rms_norm_eps", get_arch_key("{arch}.attention.layer_norm_rms_epsilon")),
            ("rope_theta", get_arch_key("{arch}.rope.freq_base")),
        ]

        for param_name, gguf_key in params:
            if gguf_key:
                field = self.reader.get_field(gguf_key)
                if field:
                    metadata[param_name] = field.contents()

        return metadata

    def _build_tensor_map(self) -> Dict[str, ReaderTensor]:
        """Build mapping from tensor names to ReaderTensor objects."""
        tensor_map = {}
        for tensor in self.reader.tensors:
            tensor_map[tensor.name] = tensor

        return tensor_map

    def _get_deepstack_visual_indexes(self):
        """Extract deepstack visual block indexes from mmproj metadata.

        Returns a list of block indices where deepstack layers exist
        (e.g., [5, 11, 17] for Qwen3-VL-2B).
        """
        if not self.mmproj_reader:
            return []
        is_deepstack_field = self.mmproj_reader.get_field("clip.vision.is_deepstack_layers")
        if is_deepstack_field:
            try:
                is_deepstack = is_deepstack_field.contents()
                return [i for i, v in enumerate(is_deepstack) if v]
            except Exception:
                pass
        return []

    def _match_deepstack(self, key: str) -> Optional[str]:
        """Match deepstack merger patterns (requires runtime mmproj lookup)."""
        if not self._has_deepstack:
            return None

        deepstack_pattern = re.match(
            r"model\.visual\.deepstack_merger_list\.(\d+)\.(\w+)(?:\.weight)?$", key)
        if deepstack_pattern:
            list_idx = int(deepstack_pattern.group(1))
            sub_name = deepstack_pattern.group(2)
            if self.mmproj_reader:
                deepstack_layers = self._get_deepstack_visual_indexes()
                if list_idx < len(deepstack_layers):
                    block_idx = deepstack_layers[list_idx]
                    field_map = {"norm": "norm", "linear_fc1": "fc1", "linear_fc2": "fc2"}
                    gguf_sub = field_map.get(sub_name, sub_name)
                    mapped = f"v.deepstack.{block_idx}.{gguf_sub}.weight"
                    if mapped in self.tensor_map:
                        return mapped
            return None

        deepstack_bias_pattern = re.match(
            r"model\.visual\.deepstack_merger_list\.(\d+)\.(\w+)\.bias$", key)
        if deepstack_bias_pattern:
            list_idx = int(deepstack_bias_pattern.group(1))
            sub_name = deepstack_bias_pattern.group(2)
            if self.mmproj_reader:
                deepstack_layers = self._get_deepstack_visual_indexes()
                if list_idx < len(deepstack_layers):
                    block_idx = deepstack_layers[list_idx]
                    field_map = {"norm": "norm", "linear_fc1": "fc1", "linear_fc2": "fc2"}
                    gguf_sub = field_map.get(sub_name, sub_name)
                    mapped = f"v.deepstack.{block_idx}.{gguf_sub}.bias"
                    if mapped in self.tensor_map:
                        return mapped
            return None

        return None

    def _map_key_to_gguf(self, key: str) -> Optional[str]:
        """Map LlmConverter key names to GGUF tensor names."""
        for compiled_regex, gguf_template in self._regex_map:
            match = compiled_regex.match(key)
            if match:
                groups = match.groups()
                if groups:
                    layer_idx = groups[0]
                    mapped = gguf_template.replace("{N}", layer_idx)
                else:
                    mapped = gguf_template
                logger.debug("_map_key_to_gguf: '%s' -> '%s'", key, mapped)
                return mapped

        mapped = self._match_deepstack(key)
        if mapped is not None:
            return mapped

        if key in self.tensor_map:
            return key

        if not key.endswith('.bias'):
            if key + ".weight" in self.tensor_map:
                return key + ".weight"

        return None

    def get_tensor_info(self, key: str) -> Optional[Dict]:
        """Get quantization information for a tensor."""
        gguf_name = self._map_key_to_gguf(key)
        if gguf_name == "__POS_EMBD_TRANSPOSE__":
            if "v.position_embd.weight" in self.tensor_map:
                tensor = self.tensor_map["v.position_embd.weight"]
                shape = tuple(tensor.shape)
                is_quant = tensor.tensor_type not in [
                    GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
                ]
                return {
                    "name": "v.position_embd.weight",
                    "shape": tuple(tensor.data.shape),
                    "quant_type": tensor.tensor_type,
                    "is_quantized": is_quant,
                }
            return None
        if gguf_name and gguf_name.startswith("__QKV_MERGE__:"):
            layer_idx = gguf_name.split(":", 1)[1]
            q_key = f"v.blk.{layer_idx}.attn_q.weight"
            if q_key in self.tensor_map:
                tensor = self.tensor_map[q_key]
                q_shape = tuple(tensor.shape)
                is_quant = tensor.tensor_type not in [
                    GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
                ]
                return {
                    "name": gguf_name,
                    "shape": (q_shape[0], q_shape[1] * 3),
                    "quant_type": tensor.tensor_type,
                    "is_quantized": is_quant,
                }
            return None
        if gguf_name and gguf_name.startswith("__QKV_MERGE_BIAS__:"):
            return {
                "name": gguf_name,
                "shape": (0, ),
                "quant_type": GGMLQuantizationType.F32,
                "is_quantized": False,
            }

        if not gguf_name or gguf_name not in self.tensor_map:
            return None

        tensor = self.tensor_map[gguf_name]

        info = {
            "name":
            gguf_name,
            "shape":
            tuple(tensor.shape),
            "quant_type":
            tensor.tensor_type,
            "is_quantized":
            tensor.tensor_type not in [
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16,
            ],
        }

        return info

    def read(self, key: str) -> np.ndarray:
        """Read tensor data, preserving quantization where possible."""
        if key in self.tensor_cache:
            return self.tensor_cache[key]
        with self._cache_lock:
            if key in self.tensor_cache:
                return self.tensor_cache[key]

        gguf_name = self._map_key_to_gguf(key)

        if gguf_name == "__POS_EMBD_TRANSPOSE__":
            if "v.position_embd.weight" in self.tensor_map:
                data = self._dequantize_tensor("v.position_embd.weight")
                with self._cache_lock:
                    self.tensor_cache[key] = data
                return data
            else:
                raise RuntimeError("Position embedding tensor not found")

        if gguf_name and gguf_name.startswith("__QKV_MERGE__:"):
            layer_idx = gguf_name.split(":", 1)[1]
            q_key = f"v.blk.{layer_idx}.attn_q.weight"
            k_key = f"v.blk.{layer_idx}.attn_k.weight"
            v_key = f"v.blk.{layer_idx}.attn_v.weight"
            if q_key in self.tensor_map and k_key in self.tensor_map and v_key in self.tensor_map:
                q_data = np.ascontiguousarray(np.transpose(self._dequantize_tensor(q_key), (1, 0)))
                k_data = np.ascontiguousarray(np.transpose(self._dequantize_tensor(k_key), (1, 0)))
                v_data = np.ascontiguousarray(np.transpose(self._dequantize_tensor(v_key), (1, 0)))
                data = np.concatenate([q_data, k_data, v_data], axis=1)
                with self._cache_lock:
                    self.tensor_cache[key] = data
                return data
            else:
                raise RuntimeError(f"QKV tensors not found for layer {layer_idx}")

        if gguf_name and gguf_name.startswith("__QKV_MERGE_BIAS__:"):
            layer_idx = gguf_name.split(":", 1)[1]
            q_bias_key = f"v.blk.{layer_idx}.attn_q.bias"
            k_bias_key = f"v.blk.{layer_idx}.attn_k.bias"
            v_bias_key = f"v.blk.{layer_idx}.attn_v.bias"
            if q_bias_key in self.tensor_map and k_bias_key in self.tensor_map and v_bias_key in self.tensor_map:
                q_bias = self._dequantize_tensor(q_bias_key)
                k_bias = self._dequantize_tensor(k_bias_key)
                v_bias = self._dequantize_tensor(v_bias_key)
                data = np.concatenate([q_bias, k_bias, v_bias], axis=0)
                with self._cache_lock:
                    self.tensor_cache[key] = data
                return data
            else:
                raise RuntimeError(f"QKV bias tensors not found for layer {layer_idx}")

        if not gguf_name or gguf_name not in self.tensor_map:
            raise RuntimeError(f"Can't find key: {key} (mapped from: {key})")

        data = self._dequantize_tensor(gguf_name)

        split_suffixes = [".1", ".2", ".3", ".4", ".5", ".6", ".7"]
        existing_suffixes = [s for s in split_suffixes if gguf_name + s in self.tensor_map]
        if gguf_name == "v.patch_embd.weight" and existing_suffixes:
            parts = [data] + [self._dequantize_tensor(gguf_name + s) for s in existing_suffixes]
            data = self._interleave_temporal_patches(parts)
        elif existing_suffixes:
            for s in existing_suffixes:
                split_data = self._dequantize_tensor(gguf_name + s)
                data = np.concatenate([data, split_data], axis=0)

        with self._cache_lock:
            self.tensor_cache[key] = data
        return data

    def _dequantize_tensor(self, gguf_name: str) -> np.ndarray:
        """Dequantize a single GGUF tensor by name and return float32 data."""
        from gguf.quants import dequantize

        if gguf_name not in self.tensor_map:
            raise RuntimeError(f"GGUF tensor not found: {gguf_name}")

        tensor = self.tensor_map[gguf_name]

        k_quant_types = {
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_K,
        }

        if tensor.tensor_type in [
                GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        ]:
            data = tensor.data
            if tensor.tensor_type == GGMLQuantizationType.F16:
                data = data.view(np.float16).astype(np.float32)
            elif tensor.tensor_type == GGMLQuantizationType.BF16:
                data = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            if tensor.tensor_type in k_quant_types and tensor.data.shape[0] > 1000:
                rows = tensor.data.shape[0]
                chunk_size = 1000
                chunks = []
                logger.info("Dequantizing large %s tensor '%s' in chunks (%d rows)",
                            tensor.tensor_type.name, gguf_name, rows)
                for i in range(0, rows, chunk_size):
                    chunk_end = min(i + chunk_size, rows)
                    chunk_data = tensor.data[i:chunk_end, :]
                    chunk_dequant = dequantize(chunk_data, tensor.tensor_type)
                    chunks.append(chunk_dequant)
                    if i % 10000 == 0:
                        logger.debug("Progress: %d/%d rows", chunk_end, rows)
                data = np.concatenate(chunks, axis=0)
                logger.info("Finished dequantizing %d rows", rows)
            else:
                data = dequantize(tensor.data, tensor.tensor_type)

        return data

    def _interleave_temporal_patches(self, parts: List[np.ndarray]) -> np.ndarray:
        """Interleave temporal-patch components of a conv2d patch-embed weight.

        In GGUF mmproj files for vision models (e.g. Qwen-VL), the conv2d
        patch-embedding weight is split by temporal patch into separate tensors
        named v.patch_embd.weight (temporal=0), v.patch_embd.weight.1 (temporal=1),
        etc.  These are NOT standard GGUF split tensors — they are independent
        temporal components that must be interleaved within each output-channel
        row, not concatenated along the row axis.

        Each part may be 2D (embed_dim, part_cols) or 4D (embed_dim, in_channels,
        kernel_h, kernel_w).  We flatten to 2D first, then interleave as:
          [in0_t0, in0_t1, ..., in0_t{N-1}, in1_t0, ..., in{M-1}_t{N-1}]
        where N = num_parts and M = in_channels.
        """
        embed_dim = parts[0].shape[0]
        if parts[0].ndim >= 3:
            in_channels = parts[0].shape[1]
            parts = [p.reshape(embed_dim, -1) for p in parts]
        else:
            in_channels = 3
        part_cols = parts[0].shape[1]
        num_parts = len(parts)
        chunk_size = part_cols // in_channels
        if chunk_size * in_channels != part_cols:
            raise RuntimeError(f"Cannot interleave temporal patches: part_cols={part_cols} "
                               f"not divisible by in_channels={in_channels}")
        out_cols = part_cols * num_parts
        result = np.empty((embed_dim, out_cols), dtype=parts[0].dtype)
        for i in range(embed_dim):
            for c in range(in_channels):
                for t in range(num_parts):
                    si = c * chunk_size
                    so = c * num_parts * chunk_size + t * chunk_size
                    result[i, so:so + chunk_size] = parts[t][i, si:si + chunk_size]
        return result

    def read_quantized(self, key: str) -> Tuple[np.ndarray, Dict]:
        """Read tensor with quantization information preserved."""
        gguf_name = self._map_key_to_gguf(key)
        if gguf_name == "__POS_EMBD_TRANSPOSE__":
            data = self.read(key)
            return data, {"quant_type": GGMLQuantizationType.F32, "is_quantized": False}
        if gguf_name and (gguf_name.startswith("__QKV_MERGE__:")
                          or gguf_name.startswith("__QKV_MERGE_BIAS__:")):
            data = self.read(key)
            return data, {"quant_type": GGMLQuantizationType.F32, "is_quantized": False}
        if not gguf_name or gguf_name not in self.tensor_map:
            raise RuntimeError(f"Can't find key: {key}")

        tensor = self.tensor_map[gguf_name]

        if tensor.tensor_type in [
                GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        ]:
            data = self.read(key)
            return data, {"quant_type": tensor.tensor_type, "is_quantized": False}
        else:
            quant_info = {
                "quant_type": tensor.tensor_type,
                "is_quantized": True,
                "block_size": self._get_block_size(tensor.tensor_type),
                "data_shape": tensor.data.shape,
                "original_shape": tuple(tensor.shape),
            }
            return tensor.data.copy(), quant_info

    def _get_block_size(self, quant_type: GGMLQuantizationType) -> int:
        """Get block size for quantization type."""
        from gguf.constants import GGML_QUANT_SIZES
        if quant_type in GGML_QUANT_SIZES:
            return GGML_QUANT_SIZES[quant_type][0]
        return 1

    def is_exist(self, key: str) -> bool:
        """Check if key exists."""
        gguf_name = self._map_key_to_gguf(key)
        if gguf_name == "__POS_EMBD_TRANSPOSE__":
            return "v.position_embd.weight" in self.tensor_map
        if gguf_name and gguf_name.startswith("__QKV_MERGE__:"):
            layer_idx = gguf_name.split(":", 1)[1]
            q_key = f"v.blk.{layer_idx}.attn_q.weight"
            return q_key in self.tensor_map
        if gguf_name and gguf_name.startswith("__QKV_MERGE_BIAS__:"):
            layer_idx = gguf_name.split(":", 1)[1]
            q_key = f"v.blk.{layer_idx}.attn_q.bias"
            return q_key in self.tensor_map
        exists = gguf_name is not None and gguf_name in self.tensor_map
        if '.bias' in key:
            logger.debug("is_exist: bias key '%s' -> gguf_name '%s', exists %s", key, gguf_name,
                         exists)
        return exists

    def get_all_tensors(self) -> List[str]:
        """Get all tensor names."""
        return list(self.tensor_map.keys())

    def get_metadata(self) -> Dict:
        """Get model metadata."""
        return self.metadata.copy()
