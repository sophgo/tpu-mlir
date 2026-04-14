# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import threading
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Add gguf-py to path

from gguf import GGUFReader, GGMLQuantizationType, ReaderTensor


class GGUFQuantLoad:
    """GGUF loader that preserves quantization information."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.reader = GGUFReader(model_path)
        self.tensor_cache = {}  # Cache for dequantized tensors
        self._cache_lock = threading.Lock()  # Lock for thread-safe cache access
        self.quant_info_cache = {}  # Cache quantization info
        self.metadata = self._extract_metadata()

        # Build tensor name mapping
        self.tensor_map = self._build_tensor_map()

    def _extract_metadata(self) -> Dict:
        """Extract model metadata from GGUF file.

        Extracts all GGUF metadata fields and maps key model parameters
        for compatibility with existing code.
        """
        metadata = {}

        # First, extract all GGUF fields
        for field_name, field in self.reader.fields.items():
            try:
                metadata[field_name] = field.contents()
            except Exception:
                # Skip fields that cannot be read
                pass

        # Get architecture (also available via general.architecture)
        arch_field = self.reader.get_field("general.architecture")
        if arch_field:
            metadata["architecture"] = arch_field.contents()

        # Get model parameters (keep existing mappings for compatibility)
        arch = metadata.get("architecture", "")

        # Helper to get value with arch template
        def get_arch_key(key_template: str) -> Optional[str]:
            if arch:
                return key_template.format(arch=arch)
            return None

        # Extract key parameters
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
            # print(f'found gguf tensors {tensor.name} {tensor}')
        return tensor_map

    def _map_key_to_gguf(self, key: str) -> Optional[str]:
        """Map LlmConverter key names to GGUF tensor names."""
        import re

        # First, try to handle .weight suffix
        base_key = key
        has_weight_suffix = key.endswith(".weight")

        # Check for exact matches (with or without .weight suffix)
        mappings = [
            # Embedding and LM head
            # Note: Qwen3 uses tied weights - lm_head is the same as embedding
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.embed_tokens", "token_embd.weight"),
            ("lm_head.weight", "token_embd.weight"),
            ("lm_head", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("model.norm", "output_norm.weight"),
        ]

        for llm_key, gguf_key in mappings:
            if key == llm_key:
                return gguf_key

        # Check for layer patterns (LlmConverter may or may not include .weight suffix)
        layer_patterns = [
            # Bias patterns - must come before weight patterns to avoid partial matching
            (r"model\.layers\.(\d+)\.self_attn\.q_proj\.bias$", "blk.{}.attn_q.bias"),
            (r"model\.layers\.(\d+)\.self_attn\.k_proj\.bias$", "blk.{}.attn_k.bias"),
            (r"model\.layers\.(\d+)\.self_attn\.v_proj\.bias$", "blk.{}.attn_v.bias"),
            (r"model\.layers\.(\d+)\.self_attn\.o_proj\.bias$", "blk.{}.attn_output.bias"),
            (r"model\.layers\.(\d+)\.mlp\.gate_proj\.bias$", "blk.{}.ffn_gate.bias"),
            (r"model\.layers\.(\d+)\.mlp\.up_proj\.bias$", "blk.{}.ffn_up.bias"),
            (r"model\.layers\.(\d+)\.mlp\.down_proj\.bias$", "blk.{}.ffn_down.bias"),
            # Weight patterns with optional .weight suffix and end-of-string anchor
            (r"model\.layers\.(\d+)\.input_layernorm(?:\.weight)?$", "blk.{}.attn_norm.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.q_proj(?:\.weight)?$", "blk.{}.attn_q.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.k_proj(?:\.weight)?$", "blk.{}.attn_k.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.v_proj(?:\.weight)?$", "blk.{}.attn_v.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.o_proj(?:\.weight)?$", "blk.{}.attn_output.weight"),
            (r"model\.layers\.(\d+)\.post_attention_layernorm(?:\.weight)?$",
             "blk.{}.ffn_norm.weight"),
            (r"model\.layers\.(\d+)\.mlp\.gate_proj(?:\.weight)?$", "blk.{}.ffn_gate.weight"),
            (r"model\.layers\.(\d+)\.mlp\.up_proj(?:\.weight)?$", "blk.{}.ffn_up.weight"),
            (r"model\.layers\.(\d+)\.mlp\.down_proj(?:\.weight)?$", "blk.{}.ffn_down.weight"),
        ]

        for llm_pattern, gguf_pattern in layer_patterns:
            match = re.match(llm_pattern, key)
            if match:
                layer_idx = match.group(1)
                mapped = gguf_pattern.format(layer_idx)
                return mapped

        # Qwen3 specific patterns (with q_norm and k_norm)
        qwen3_patterns = [
            (r"model\.layers\.(\d+)\.self_attn\.q_norm(?:\.weight)?$", "blk.{}.attn_q_norm.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.k_norm(?:\.weight)?$", "blk.{}.attn_k_norm.weight"),
        ]

        for llm_pattern, gguf_pattern in qwen3_patterns:
            match = re.match(llm_pattern, key)
            if match:
                layer_idx = match.group(1)
                mapped = gguf_pattern.format(layer_idx)
                logger.debug("_map_key_to_gguf: pattern matched '%s' -> '%s'", key, mapped)
                return mapped

        # If no pattern matched, try direct match in tensor_map
        if key in self.tensor_map:
            return key

        # Try with .weight suffix (but not for bias keys)
        if not key.endswith('.bias'):
            if key + ".weight" in self.tensor_map:
                return key + ".weight"
        else:
            # For bias keys, try removing .bias and adding .weight? No, that doesn't make sense.
            # Bias keys should map to bias tensors, not weight tensors.
            pass

        return None

    def get_tensor_info(self, key: str) -> Optional[Dict]:
        """Get quantization information for a tensor."""
        gguf_name = self._map_key_to_gguf(key)
        # print(f'mapping key {key} to gguf tensor {gguf_name}')
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
        if not gguf_name or gguf_name not in self.tensor_map:
            raise RuntimeError(f"Can't find key: {key} (mapped from: {key})")

        tensor = self.tensor_map[gguf_name]

        # Debug: print tensor info

        # Handle different quantization types
        if tensor.tensor_type in [
                GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        ]:
            # Floating point types
            data = tensor.data
            if tensor.tensor_type == GGMLQuantizationType.F16:
                data = data.view(np.float16).astype(np.float32)
            elif tensor.tensor_type == GGMLQuantizationType.BF16:
                # BF16 not directly supported in numpy, convert via float32
                data = data.view(np.uint16)
                # Simple conversion (approximate)
                data = data.astype(np.float32) * (1.0 / 32768.0)
        else:
            # Quantized types - dequantize to float32
            from gguf.quants import dequantize

            # Check if it's a large K-quantization tensor (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
            # These can be memory intensive to dequantize all at once
            k_quant_types = {
                GGMLQuantizationType.Q2_K,
                GGMLQuantizationType.Q3_K,
                GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K,
                GGMLQuantizationType.Q8_K,
            }

            if tensor.tensor_type in k_quant_types and tensor.data.shape[0] > 1000:
                # Large K-quant tensor - dequantize in chunks to avoid memory issues
                rows = tensor.data.shape[0]
                chunk_size = 1000  # Dequantize 1000 rows at a time
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

                # Concatenate all chunks
                data = np.concatenate(chunks, axis=0)
                logger.info("Finished dequantizing %d rows", rows)
            else:
                # Small tensor or non-K quantization - dequantize all at once
                data = dequantize(tensor.data, tensor.tensor_type)

        with self._cache_lock:
            self.tensor_cache[key] = data
        return data

    def read_quantized(self, key: str) -> Tuple[np.ndarray, Dict]:
        """Read tensor with quantization information preserved."""
        gguf_name = self._map_key_to_gguf(key)
        if not gguf_name or gguf_name not in self.tensor_map:
            raise RuntimeError(f"Can't find key: {key}")

        tensor = self.tensor_map[gguf_name]

        if tensor.tensor_type in [
                GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        ]:
            # Not quantized
            data = self.read(key)
            return data, {"quant_type": tensor.tensor_type, "is_quantized": False}
        else:
            # Return raw quantized data + quantization info
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
