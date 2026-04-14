# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import os

# Add gguf-py to path

# Import GGUF types
from gguf import GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES, GGUF_DEFAULT_ALIGNMENT


class QuantFormat(Enum):
    """Quantization format enumeration."""
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q4_1 = "gguf_q4_1"
    GGUF_Q8_0 = "gguf_q8_0"
    GGUF_Q8_1 = "gguf_q8_1"
    LLMCONV_4BIT = "llmconv_4bit"
    LLMCONV_8BIT = "llmconv_8bit"


def get_quant_type_group_size(quant_type):
    """Get the natural sub-block (group) size for a GGML quantization type.

    Q6_K has 16-element sub-blocks; all other common types have 32.
    """
    if quant_type == GGMLQuantizationType.Q6_K:
        return 16
    elif quant_type in (
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q8_K,
    ):
        return 32
    else:
        return 32


class QuantConverter:
    """Convert between GGUF quantization and LlmConverter quantization formats."""

    def __init__(self, group_size: int = 64, scale_dtype: np.dtype = np.float32):
        self.group_size = group_size
        self.scale_dtype = scale_dtype

    def _reshape_gguf_data(self, gguf_data: np.ndarray, quant_type: GGMLQuantizationType,
                           original_shape: Tuple[int, ...]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Reshape GGUF quantized data to [n_blocks, type_size].

        Handles two cases:
        1. gguf_data is raw 1D uint8 array (with possible alignment padding)
        2. gguf_data is already shaped as (rows, bytes_per_row) from GGUFReader

        Args:
            gguf_data: GGUF tensor data as 1D uint8 array or 2D shaped array
            quant_type: GGUF quantization type
            original_shape: Original tensor shape (rows, cols) or (elements,)

        Returns:
            Tuple of (reshaped_data, (out_dim, in_dim))
            reshaped_data shape: [n_blocks, type_size] where n_blocks = rows * (cols // block_size)
        """
        if quant_type not in GGML_QUANT_SIZES:
            raise ValueError(f"Unknown quantization type: {quant_type}")

        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        block_size = int(block_size)
        type_size = int(type_size)
        alignment = GGUF_DEFAULT_ALIGNMENT  # 32 bytes

        if len(original_shape) == 1:
            # 1D tensor (e.g., bias)
            elements = int(original_shape[0])  # ensure Python int
            if elements % block_size != 0:
                raise ValueError(
                    f"1D tensor size {elements} not divisible by block size {block_size}")
            n_blocks = elements // block_size
            unpadded_bytes = n_blocks * type_size

            # If gguf_data is already shaped (shouldn't happen for 1D)
            if gguf_data.ndim == 2:
                # Reshape to (n_blocks, type_size) directly
                if gguf_data.shape[1] != type_size:
                    raise ValueError(f"Unexpected 2D shape for 1D tensor: {gguf_data.shape}")
                reshaped = gguf_data.reshape(n_blocks, type_size)
                return reshaped, (elements, 1)

            # Raw 1D array - handle padding
            padded_bytes = ((unpadded_bytes + alignment - 1) // alignment) * alignment
            if gguf_data.size != padded_bytes:
                raise ValueError(
                    f"GGUF data size {gguf_data.size} != expected padded size {padded_bytes}")
            # Slice off trailing padding
            unpadded_data = gguf_data[:unpadded_bytes]
            reshaped = unpadded_data.reshape(n_blocks, type_size)
            return reshaped, (elements, 1)  # treat as column vector

        elif len(original_shape) == 2:
            rows, cols = map(int, original_shape)  # ensure Python ints
            if cols % block_size != 0:
                raise ValueError(
                    f"Column dimension {cols} not divisible by block size {block_size}")
            blocks_per_row = cols // block_size
            unpadded_bytes_per_row = blocks_per_row * type_size

            # Check if gguf_data is already correctly shaped
            # Compute per-row padding (GGUF may pad each row to alignment)
            padded_bytes_per_row = (
                (unpadded_bytes_per_row + alignment - 1) // alignment) * alignment

            # Verify total size matches rows * padded_bytes_per_row
            expected_total = rows * padded_bytes_per_row
            if gguf_data.size != expected_total:
                # Try to see if there's additional global padding (no per-row padding)
                total_unpadded = rows * unpadded_bytes_per_row
                total_padded = ((total_unpadded + alignment - 1) // alignment) * alignment
                if gguf_data.size == total_padded:
                    # Global padding only, not per-row
                    unpadded_data = gguf_data[:total_unpadded]
                    n_blocks = rows * blocks_per_row
                    reshaped = unpadded_data.reshape(n_blocks, type_size)
                    return reshaped, (rows, cols)
                else:
                    raise ValueError(
                        f"GGUF data size {gguf_data.size} != expected per-row padded size {expected_total} or global padded size {total_padded}"
                    )

            # Reshape to rows × padded_bytes_per_row, slice unpadded columns per row
            padded_2d = gguf_data.reshape(rows, padded_bytes_per_row)
            unpadded_2d = padded_2d[:, :unpadded_bytes_per_row]
            # Now reshape to (rows * blocks_per_row, type_size)
            n_blocks = rows * blocks_per_row
            reshaped = unpadded_2d.reshape(n_blocks, type_size)
            return reshaped, (rows, cols)

        else:
            raise NotImplementedError(f"Tensor with {len(original_shape)} dimensions not supported")

    def convert_from_gguf(self,
                          gguf_data: np.ndarray,
                          quant_type: GGMLQuantizationType,
                          original_shape: Tuple[int, ...],
                          group_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Convert GGUF quantized data to LlmConverter format.

        Args:
            gguf_data: Raw GGUF tensor data
            quant_type: GGUF quantization type
            original_shape: Original tensor shape (before quantization)
            group_size: Override group_size for this conversion. If None, uses
                the natural sub-block size for the quant type (16 for Q6_K, 32 for others).

        Returns:
            Dictionary with keys:
            - 'qweight': Packed weight tensor
            - 'scales': Scale tensor
            - 'qzeros': Zero point tensor (if applicable)
            - 'bits': Quantization bits (4 or 8)
            - 'group_size': Group size used
        """

        reshaped_data, processed_shape = self._reshape_gguf_data(gguf_data, quant_type,
                                                                 original_shape)

        if len(processed_shape) == 2 and processed_shape[1] == 1:
            raise NotImplementedError(
                f"1D quantized tensor conversion not supported (shape: {original_shape})")

        if group_size is None:
            group_size = get_quant_type_group_size(quant_type)

        # Dispatch to appropriate conversion method
        if quant_type == GGMLQuantizationType.Q4_0:
            return self._convert_q4_0(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q8_0:
            return self._convert_q8_0(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q4_1:
            return self._convert_q4_1(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q8_1:
            return self._convert_q8_1(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q5_0:
            return self._convert_q5_0(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q5_1:
            return self._convert_q5_1(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q4_K:
            return self._convert_q4_k(reshaped_data, processed_shape, group_size)
        elif quant_type == GGMLQuantizationType.Q6_K:
            return self._convert_q6_k(reshaped_data, processed_shape, group_size)
        elif quant_type in [
                GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K, GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q8_K, GGMLQuantizationType.IQ2_XXS,
                GGMLQuantizationType.IQ2_XS, GGMLQuantizationType.IQ3_XXS,
                GGMLQuantizationType.IQ1_S, GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.IQ3_S,
                GGMLQuantizationType.IQ2_S, GGMLQuantizationType.IQ4_XS, GGMLQuantizationType.IQ1_M,
                GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0, GGMLQuantizationType.MXFP4
        ]:
            raise NotImplementedError(
                f"Conversion for {quant_type} not yet implemented. K-quantization and IQ types require additional implementation."
            )
        else:
            raise NotImplementedError(f"Conversion for {quant_type} not implemented")

    def _convert_q4_0(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q4_0 to LlmConverter 4-bit format.

        GGUF Q4_0 format:
        - Block size: 32 elements
        - Bytes per block: 2 (fp16 scale) + 16 (packed 4-bit weights)
        - Quantization: value = scale * (q - 8) where q ∈ [0, 15]
        """
        # GGUF data shape: [n_blocks, 18]
        n_blocks = gguf_data.shape[0]

        # Extract scales (fp16) and quantized data
        scales_fp16 = gguf_data[:, :2].view(np.float16)  # [n_blocks, 1]
        qdata = gguf_data[:, 2:]  # [n_blocks, 16] packed 4-bit

        # Convert scales to target dtype
        scales = scales_fp16.astype(self.scale_dtype)

        # Unpack 4-bit weights (vectorized)
        # GGUF Q4_0 packing: byte j contains x[j] (lower 4 bits) and x[j+16] (upper 4 bits)
        low_bits = (qdata & 0x0F).astype(np.uint8)  # [n_blocks, 16]
        high_bits = ((qdata >> 4) & 0x0F).astype(np.uint8)  # [n_blocks, 16]
        weights_4bit = np.concatenate([low_bits, high_bits], axis=1)  # [n_blocks, 32]

        # GGUF Q4_0 has implicit zero point = 8
        # Convert to LlmConverter format with explicit zero points
        zero_points = np.full((n_blocks, 1), 8, dtype=np.uint8)

        # Reshape to original tensor dimensions
        # Original shape is typically [out_dim, in_dim]
        # GGUF stores blocks along in_dim dimension
        out_dim, in_dim = original_shape

        # Calculate blocks per row (in_dim dimension)
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = weights_4bit.reshape(out_dim, in_dim)

        # Reshape scales and zero_points from [n_blocks, 1] to [out_dim, blocks_per_row]
        scales = scales.reshape(out_dim, blocks_per_row)
        zero_points = zero_points.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size
        return self._regroup_to_groups(weights_reshaped,
                                       scales,
                                       zero_points,
                                       original_shape,
                                       bits=4)

    def _convert_q8_0(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q8_0 to LlmConverter 8-bit format.

        GGUF Q8_0 format:
        - Block size: 32 elements
        - Bytes per block: 2 (fp16 scale) + 32 (8-bit weights)
        - Quantization: value = scale * q where q ∈ [-127, 127]
        """
        # GGUF data shape: [n_blocks, 34]
        n_blocks = gguf_data.shape[0]

        # Extract scales (fp16) and quantized data
        scales_fp16 = gguf_data[:, :2].view(np.float16)  # [n_blocks, 1]
        qdata = gguf_data[:, 2:].view(np.int8)  # [n_blocks, 32] int8

        # Convert scales to target dtype
        scales = scales_fp16.astype(self.scale_dtype)

        # Q8_0 is symmetric quantization, no zero points
        zero_points = None

        # Reshape to original tensor dimensions
        out_dim, in_dim = original_shape

        # Calculate blocks per row
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = qdata.reshape(out_dim, in_dim)

        # Reshape scales from [n_blocks, 1] to [out_dim, blocks_per_row]
        scales = scales.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size
        return self._regroup_to_groups(weights_reshaped,
                                       scales,
                                       zero_points,
                                       original_shape,
                                       bits=8)

    def _convert_q4_1(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q4_1 to LlmConverter 4-bit format.

        GGUF Q4_1 format:
        - Block size: 32 elements
        - Bytes per block: 2 (fp16 delta) + 2 (fp16 min) + 16 (packed 4-bit)
        - Quantization: value = delta * q + min where q ∈ [0, 15]
        """
        # GGUF data shape: [n_blocks, 20]
        n_blocks = gguf_data.shape[0]

        # Extract delta (scale), min, and quantized data
        delta_fp16 = gguf_data[:, :2].view(np.float16)  # [n_blocks, 1]
        min_fp16 = gguf_data[:, 2:4].view(np.float16)  # [n_blocks, 1]
        qdata = gguf_data[:, 4:]  # [n_blocks, 16] packed 4-bit

        # Convert to target dtype
        delta = delta_fp16.astype(self.scale_dtype)
        min_val = min_fp16.astype(self.scale_dtype)

        # Unpack 4-bit weights (vectorized)
        # GGUF Q4_1 packing: byte j contains x[j] (lower 4 bits) and x[j+16] (upper 4 bits)
        low_bits = (qdata & 0x0F).astype(np.uint8)  # [n_blocks, 16]
        high_bits = ((qdata >> 4) & 0x0F).astype(np.uint8)  # [n_blocks, 16]
        weights_4bit = np.concatenate([low_bits, high_bits], axis=1)  # [n_blocks, 32]

        # Q4_1 has explicit min value, convert to zero point representation
        # value = delta * q + min = delta * (q - zero_point)
        # => zero_point = -min / delta
        # Avoid division by zero
        epsilon = 1e-8
        delta_safe = np.where(np.abs(delta) > epsilon, delta, epsilon * np.sign(delta) + epsilon)
        zero_points_float = -min_val / delta_safe
        zero_points_int = np.round(zero_points_float).astype(np.int32)
        # Clamp to 0-15 range for 4-bit zero points
        zero_points_int = np.clip(zero_points_int, 0, 15)
        zero_points = zero_points_int.astype(np.uint8)

        # Reshape to original tensor dimensions
        out_dim, in_dim = original_shape
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = weights_4bit.reshape(out_dim, in_dim)

        # Reshape scales (delta) and zero_points from [n_blocks, 1] to [out_dim, blocks_per_row]
        delta = delta.reshape(out_dim, blocks_per_row)
        zero_points = zero_points.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size
        return self._regroup_to_groups(weights_reshaped,
                                       delta,
                                       zero_points,
                                       original_shape,
                                       bits=4,
                                       group_size=group_size)

    def _convert_q8_1(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q8_1 to LlmConverter 8-bit format.

        GGUF Q8_1 format:
        - Block size: 32 elements
        - Bytes per block: 4 (fp32 delta) + 4 (fp32 min) + 32 (8-bit weights)
        - Quantization: value = delta * q + min where q ∈ [-127, 127]
        """
        # GGUF data shape: [n_blocks, 40]
        n_blocks = gguf_data.shape[0]

        # Extract delta (fp32), min (fp32), and quantized data
        delta_fp32 = gguf_data[:, :4].view(np.float32)  # [n_blocks, 1]
        min_fp32 = gguf_data[:, 4:8].view(np.float32)  # [n_blocks, 1]
        qdata = gguf_data[:, 8:].view(np.int8)  # [n_blocks, 32] int8

        # Convert to target dtype
        delta = delta_fp32.astype(self.scale_dtype)
        min_val = min_fp32.astype(self.scale_dtype)

        # Compute zero point = -min / delta
        epsilon = 1e-8
        delta_safe = np.where(np.abs(delta) > epsilon, delta, epsilon * np.sign(delta) + epsilon)
        zero_points_float = -min_val / delta_safe
        zero_points_int = np.round(zero_points_float).astype(np.int32)
        # Clamp to int8 range (-128..127) but zero point is added to q which is already int8
        # Zero point outside this range would cause overflow; we clamp zero point to reasonable range
        zero_points_int = np.clip(zero_points_int, -128, 127)

        # Adjust weights: q' = q - zero_point (int8 with possible overflow)
        # We'll keep as int16 to avoid overflow, then clamp to int8 range
        q_adjusted = qdata.astype(np.int16) - zero_points_int.astype(np.int16)
        q_adjusted = np.clip(q_adjusted, -127, 127).astype(np.int8)

        # Reshape to original tensor dimensions
        out_dim, in_dim = original_shape
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = q_adjusted.reshape(out_dim, in_dim)

        # Reshape scales (delta) from [n_blocks, 1] to [out_dim, blocks_per_row]
        delta = delta.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size (8-bit, no zero points)
        return self._regroup_to_groups(weights_reshaped,
                                       delta,
                                       None,
                                       original_shape,
                                       bits=8,
                                       group_size=group_size)

    def _unpack_q4_k_scales(self, scales_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack Q4_K scales from 12 bytes to 8 scale values and 8 min values.

        Directly matches llama.cpp get_scale_min_k4(j, q, &d, &m):
        - For j < 4:  sc = q[j] & 63,      m = q[j+4] & 63
        - For j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4),
                       m = (q[j+4] >> 4)   | ((q[j] >> 6) << 4)
        """
        n_blocks = scales_raw.shape[0]
        q = scales_raw.view(np.uint8).reshape(n_blocks, 12)

        sc = np.zeros((n_blocks, 8), dtype=np.float32)
        m = np.zeros((n_blocks, 8), dtype=np.float32)

        for j in range(8):
            if j < 4:
                sc[:, j] = (q[:, j] & 63).astype(np.float32)
                m[:, j] = (q[:, j + 4] & 63).astype(np.float32)
            else:
                sc[:, j] = ((q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)).astype(np.float32)
                m[:, j] = ((q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)).astype(np.float32)

        return sc, m

    def _convert_q4_k(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q4_K to LlmConverter 8-bit format with float16 scale.

        Q4_K format:
        - Block size: 256 elements (8 sub-blocks of 32)
        - Bytes per block: 2 (fp16 d) + 2 (fp16 dmin) + 12 (scales) + 128 (packed 4-bit) = 144

        Dequant formula: value = d * sc * q4 - dmin * m
        This is asymmetric with float zero points.

        Since Q4_K's zero points are non-integer (dmin*m / d*sc), we cannot
        directly store them as int8. Instead, we fully dequantize to float32
        then re-quantize to int8 with proper integer scales and zero points.

        Output format (w8f16/w8bf16):
        - scale: float32 per group (32 elements)
        - qweight: uint8 (re-quantized 8-bit)
        - qzeros: uint8 (integer zero point)
        - Formula: scale * (qweight_uint8 - qzeros_uint8)
        """
        # GGUF data shape: [n_blocks, 144]
        n_blocks = gguf_data.shape[0]
        out_dim, in_dim = original_shape

        # 1. Extract super-scales
        d = gguf_data[:, :2].view(np.float16).astype(np.float32)
        dmin = gguf_data[:, 2:4].view(np.float16).astype(np.float32)
        scales_raw = gguf_data[:, 4:16]
        qdata = gguf_data[:, 16:]

        # 2. Unpack scales: 8 scale values + 8 min values (6-bit each)
        sc, m = self._unpack_q4_k_scales(scales_raw)

        # 3. Unpack 4-bit weights (vectorized)
        # Q4_K packs 4 4-bit values per byte across 4 sub-blocks of 64 elements
        # Each 64-element chunk has 32 bytes, with byte[l] holding lower4 for position l
        # and upper4 for position l+32 within that chunk.
        weights_4bit = np.zeros((n_blocks, 256), dtype=np.uint8)
        for j in range(4):  # 4 chunks of 64 elements
            chunk_start = j * 64
            q_offset = j * 32
            chunk_qdata = qdata[:, q_offset:q_offset + 32]  # [n_blocks, 32]
            low_bits = (chunk_qdata & 0x0F)  # [n_blocks, 32]
            high_bits = ((chunk_qdata >> 4) & 0x0F)  # [n_blocks, 32]
            weights_4bit[:, chunk_start:chunk_start + 32] = low_bits
            weights_4bit[:, chunk_start + 32:chunk_start + 64] = high_bits

        # 4. Fully dequantize to float32: value = d * sc * q4 - dmin * m (vectorized)
        sub_block_idx = np.arange(256) // 32  # [256], maps each element to its sub-block (0-7)
        sc_per_element = sc[:, sub_block_idx]  # [n_blocks, 256]
        m_per_element = m[:, sub_block_idx]  # [n_blocks, 256]
        dequant = d * sc_per_element * weights_4bit.astype(np.float32) - dmin * m_per_element

        # 5. Re-quantize to int8 per 32-element group (vectorized)
        dequant = dequant.reshape(out_dim, in_dim)
        groups_per_row = in_dim // 32
        reshaped = dequant.reshape(out_dim, groups_per_row, 32)
        vmin = reshaped.min(axis=2)  # [out_dim, groups_per_row]
        vmax = reshaped.max(axis=2)  # [out_dim, groups_per_row]

        scale_val = np.where(vmax == vmin, 1.0, (vmax - vmin) / 255.0)
        zp_val = np.where(vmax == vmin, 128.0, np.round(-vmin / scale_val))
        zp_val = np.clip(zp_val, 0, 255)

        scales = scale_val.astype(self.scale_dtype)
        qzeros = zp_val.astype(np.uint8)

        qweight = np.round(reshaped / scale_val[:, :, np.newaxis]) + zp_val[:, :, np.newaxis]
        qweight = np.clip(qweight, 0, 255).astype(np.uint8).reshape(out_dim, in_dim)

        # 6. Regroup to group_size
        return self._regroup_to_groups((qweight.astype(np.int16) - 128).astype(np.int8),
                                       scales, (qzeros.astype(np.int16) - 128).astype(np.int8),
                                       original_shape,
                                       bits=8,
                                       group_size=32)

    def _convert_q6_k(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 16) -> Dict[str, np.ndarray]:
        """Convert GGUF Q6_K to LlmConverter 8-bit symmetric format.

        Q6_K format (210 bytes per 256 elements):
        - ql[128]: lower 4 bits of weights (interleaved packing)
        - qh[64]: upper 2 bits of weights (interleaved packing)
        - scales[16]: 16 per-sub-block scales (int8)
        - d: super-block scale (fp16)

        Dequant formula: value = d * scales[j] * ((lower4 | (upper2 << 4)) - 32)
        Symmetric: zero point = 0

        group_size defaults to 16 (Q6_K sub-block size) to preserve
        per-sub-block scale accuracy. Using group_size=32 for Q6_K
        averages two 16-element sub-block scales into one, losing accuracy.
        """
        n_blocks = gguf_data.shape[0]
        out_dim, in_dim = original_shape

        # 1. Extract fields from block
        ql = gguf_data[:, :128].copy()  # [n_blocks, 128] uint8
        qh = gguf_data[:, 128:192].copy()  # [n_blocks, 64] uint8
        scales_int8 = gguf_data[:, 192:208].view(np.int8).astype(np.float32)  # [n_blocks, 16]
        d = gguf_data[:, 208:210].view(np.float16).astype(np.float32)  # [n_blocks, 1]

        # 2. Unpack 6-bit weights from interleaved ql/qh format (vectorized)
        # Each block of 256 elements is split into 2 chunks of 128
        # Each chunk: 4 rows of 32, ql holds lower 4 bits, qh holds upper 2 bits
        weights = np.zeros((n_blocks, 256), dtype=np.int8)

        for n in range(2):  # 2 chunks per block
            chunk_off = n * 128
            ql_off = n * 64
            qh_off = n * 32

            ql_chunk = ql[:, ql_off:ql_off + 64]  # [n_blocks, 64]
            qh_chunk = qh[:, qh_off:qh_off + 32]  # [n_blocks, 32]

            # Row 0: positions [chunk_off+0..31]
            # q1 = (ql[l] & 0xF) | ((qh[l] & 0x3) << 4), for l in 0..31
            q1 = (ql_chunk[:, :32] & 0x0F) | ((qh_chunk & 0x03) << 4)

            # Row 1: positions [chunk_off+32..63]
            # q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 0x3) << 4), for l in 0..31
            q2 = (ql_chunk[:, 32:64] & 0x0F) | (((qh_chunk >> 2) & 0x03) << 4)

            # Row 2: positions [chunk_off+64..95]
            # q3 = (ql[l] >> 4) | ((qh[l] >> 4) & 0x3) << 4), for l in 0..31
            q3 = (ql_chunk[:, :32] >> 4) | (((qh_chunk >> 4) & 0x03) << 4)

            # Row 3: positions [chunk_off+96..127]
            # q4 = (ql[l+32] >> 4) | ((qh[l] >> 6) & 0x3) << 4), for l in 0..31
            q4 = (ql_chunk[:, 32:64] >> 4) | (((qh_chunk >> 6) & 0x03) << 4)

            # Subtract 32 to get signed range [-32, 31]
            weights[:, chunk_off:chunk_off + 32] = q1.astype(np.int8) - 32
            weights[:, chunk_off + 32:chunk_off + 64] = q2.astype(np.int8) - 32
            weights[:, chunk_off + 64:chunk_off + 96] = q3.astype(np.int8) - 32
            weights[:, chunk_off + 96:chunk_off + 128] = q4.astype(np.int8) - 32

        # 3. Compute per-sub-block effective scales
        # 16 sub-blocks of 16 elements each per 256-element block
        # value = d * scales[j] * q, so effective scale = d * scales[j]
        scale = d.reshape(n_blocks, 1) * scales_int8  # [n_blocks, 16]
        scale = scale.astype(self.scale_dtype)

        # 4. Reshape to output format
        # 16 sub-blocks per row, each of 16 elements
        # groups_per_row = in_dim // 16
        groups_per_row = in_dim // 16
        weights = weights.reshape(out_dim, in_dim)
        scales = scale.reshape(out_dim, groups_per_row)

        # 5. Symmetric: no zero points (None)
        return self._regroup_to_groups(weights,
                                       scales,
                                       None,
                                       original_shape,
                                       bits=8,
                                       group_size=group_size)

    def _convert_q5_0(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q5_0 to LlmConverter 8-bit format.

        GGUF Q5_0 format:
        - Block size: 32 elements
        - Bytes per block: 2 (fp16 scale) + 4 (high bits) + 16 (low bits)
        - Quantization: value = scale * (q - 16) where q ∈ [0, 31]
        """
        # GGUF data shape: [n_blocks, 22]
        n_blocks = gguf_data.shape[0]

        # Extract scales (fp16) and quantized data
        scales_fp16 = gguf_data[:, :2].view(np.float16)  # [n_blocks, 1]
        qh = gguf_data[:, 2:6]  # [n_blocks, 4] high bits
        qs = gguf_data[:, 6:]  # [n_blocks, 16] low bits

        # Convert scales to target dtype
        scales = scales_fp16.astype(self.scale_dtype)

        # Unpack low bits: each byte contains two 4-bit weights
        qs_uint8 = qs.view(np.uint8).reshape(n_blocks, 16)
        ql_low = qs_uint8 & 0x0F
        ql_high = (qs_uint8 >> 4) & 0x0F
        ql = np.zeros((n_blocks, 32), dtype=np.uint8)
        ql[:, 0::2] = ql_low
        ql[:, 1::2] = ql_high

        # Unpack high bits: 4 bytes per block, each bit corresponds to one weight
        qh_uint32 = qh.view(np.uint32).reshape(n_blocks)  # [n_blocks]
        element_idx = np.arange(32, dtype=np.uint32)
        qh_bits = ((qh_uint32[:, np.newaxis] >> element_idx) & 1).astype(np.uint8)  # [n_blocks, 32]

        # Combine low and high bits to get 5-bit weights (0-31)
        q = ql | (qh_bits << 4)

        # Convert to signed int8 with bias -16
        weights_int8 = q.astype(np.int8) - 16

        # Reshape to original tensor dimensions
        out_dim, in_dim = original_shape
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = weights_int8.reshape(out_dim, in_dim)

        # Reshape scales from [n_blocks, 1] to [out_dim, blocks_per_row]
        scales = scales.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size (8-bit, no zero points)
        return self._regroup_to_groups(weights_reshaped,
                                       scales,
                                       None,
                                       original_shape,
                                       bits=8,
                                       group_size=group_size)

    def _convert_q5_1(self,
                      gguf_data: np.ndarray,
                      original_shape: Tuple[int, ...],
                      group_size: int = 32) -> Dict[str, np.ndarray]:
        """Convert GGUF Q5_1 to LlmConverter 8-bit format.

        GGUF Q5_1 format:
        - Block size: 32 elements
        - Bytes per block: 2 (fp16 delta) + 2 (fp16 min) + 4 (high bits) + 16 (low bits)
        - Quantization: value = delta * q + min where q ∈ [0, 31]
        """
        # GGUF data shape: [n_blocks, 24]
        n_blocks = gguf_data.shape[0]

        # Extract delta (fp16), min (fp16), and quantized data
        delta_fp16 = gguf_data[:, :2].view(np.float16)  # [n_blocks, 1]
        min_fp16 = gguf_data[:, 2:4].view(np.float16)  # [n_blocks, 1]
        qh = gguf_data[:, 4:8]  # [n_blocks, 4] high bits
        qs = gguf_data[:, 8:]  # [n_blocks, 16] low bits

        # Convert to target dtype
        delta = delta_fp16.astype(self.scale_dtype)
        min_val = min_fp16.astype(self.scale_dtype)

        # Unpack low bits: each byte contains two 4-bit weights
        qs_uint8 = qs.view(np.uint8).reshape(n_blocks, 16)
        ql_low = qs_uint8 & 0x0F
        ql_high = (qs_uint8 >> 4) & 0x0F
        ql = np.zeros((n_blocks, 32), dtype=np.uint8)
        ql[:, 0::2] = ql_low
        ql[:, 1::2] = ql_high

        # Unpack high bits: 4 bytes per block, each bit corresponds to one weight
        qh_uint32 = qh.view(np.uint32).reshape(n_blocks)  # [n_blocks]
        element_idx = np.arange(32, dtype=np.uint32)
        qh_bits = ((qh_uint32[:, np.newaxis] >> element_idx) & 1).astype(np.uint8)  # [n_blocks, 32]

        # Combine low and high bits to get 5-bit weights (0-31)
        q = ql | (qh_bits << 4)

        # Compute zero point = -min / delta
        epsilon = 1e-8
        delta_safe = np.where(np.abs(delta) > epsilon, delta, epsilon * np.sign(delta) + epsilon)
        zero_points_float = -min_val / delta_safe
        zero_points_int = np.round(zero_points_float).astype(np.int32)
        # Clamp to reasonable range (0-31) for 5-bit weights
        zero_points_int = np.clip(zero_points_int, 0, 31)

        # Adjust weights: q' = q - zero_point (int8 with possible overflow)
        q_adjusted = q.astype(np.int16) - zero_points_int.astype(np.int16)
        # Clamp to int8 range (-127, 127) but q' should be within -31..31
        q_adjusted = np.clip(q_adjusted, -127, 127).astype(np.int8)

        # Reshape to original tensor dimensions
        out_dim, in_dim = original_shape
        blocks_per_row = in_dim // 32
        if in_dim % 32 != 0:
            raise ValueError(f"Input dimension {in_dim} not divisible by GGUF block size 32")

        # Reshape weights to [out_dim, in_dim]
        weights_reshaped = q_adjusted.reshape(out_dim, in_dim)

        # Reshape scales (delta) from [n_blocks, 1] to [out_dim, blocks_per_row]
        delta = delta.reshape(out_dim, blocks_per_row)

        # Regroup to LlmConverter group size (8-bit, no zero points)
        return self._regroup_to_groups(weights_reshaped,
                                       delta,
                                       None,
                                       original_shape,
                                       bits=8,
                                       group_size=group_size)

    def _regroup_to_groups(self,
                           weights: np.ndarray,
                           block_scales: np.ndarray,
                           block_zeros: Optional[np.ndarray],
                           original_shape: Tuple[int, ...],
                           bits: int,
                           group_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Regroup from GGUF blocks to LlmConverter groups.

        group_size: effective group size for this tensor. For Q6_K this must
        be 16; for Q4_K and other types it is 32. Using a mismatched
        group_size (e.g., 32 for Q6_K) averages sub-block scales and
        loses accuracy.
        """
        if group_size is None:
            group_size = self.group_size
        out_dim, in_dim = original_shape

        # Calculate number of groups
        if in_dim % group_size != 0:
            # Pad if necessary
            padded_in_dim = ((in_dim + group_size - 1) // group_size) * group_size
            pad_width = ((0, 0), (0, padded_in_dim - in_dim))
            weights = np.pad(weights, pad_width, mode='constant')
            # Pad block_scales and block_zeros to match padded dimension
            blocks_per_row = block_scales.shape[1]
            sub_block_size = in_dim // blocks_per_row if blocks_per_row > 0 else 32
            original_blocks_per_row = in_dim // sub_block_size
            padded_blocks_per_row = padded_in_dim // sub_block_size
            if padded_blocks_per_row > original_blocks_per_row:
                # Pad block_scales with 1.0 (neutral scale)
                pad_blocks = padded_blocks_per_row - original_blocks_per_row
                block_scales = np.pad(block_scales, ((0, 0), (0, pad_blocks)),
                                      mode='constant',
                                      constant_values=1.0)
                if block_zeros is not None:
                    # Pad block_zeros with 0 (neutral zero point)
                    block_zeros = np.pad(block_zeros, ((0, 0), (0, pad_blocks)),
                                         mode='constant',
                                         constant_values=0)
            in_dim = padded_in_dim

        n_groups = in_dim // group_size

        # Initialize output arrays
        if bits == 4:
            # Pack 2 weights per byte (vectorized)
            even = weights[:, 0::2] & 0x0F
            odd = weights[:, 1::2] & 0x0F
            qweight = (even | (odd << 4)).astype(np.uint8)
        else:  # 8-bit
            # Convert signed int8 to unsigned uint8 by adding 128
            # GGUF stores signed int8 values in range [-127, 127]
            # MLIR expects uint8 with zero point 128
            if weights.dtype == np.int8:
                # Convert to uint8: (weights + 128) mod 256
                weights_uint8 = ((weights.astype(np.int16) + 128) & 0xFF).astype(np.uint8)
            else:
                # Already uint8 (should not happen for GGUF 8-bit)
                weights_uint8 = weights.astype(np.uint8)

            # Store unpacked uint8 weights
            qweight = weights_uint8

        # Handle scales
        # Determine sub-block size from block_scales shape
        # block_scales shape: (out_dim, blocks_per_row)
        blocks_per_row = block_scales.shape[1]
        sub_block_size = in_dim // blocks_per_row  # 32 for Q4_K, 16 for Q6_K
        blocks_per_group = group_size // sub_block_size
        if group_size % sub_block_size != 0:
            raise ValueError(
                f"Group size {group_size} must be multiple of sub-block size {sub_block_size}")

        # block_scales shape: (out_dim, blocks_per_row)
        # scales shape: (out_dim, n_groups)
        # Reshape block_scales to (out_dim, n_groups, blocks_per_group) and average
        scales = block_scales.reshape(out_dim, n_groups,
                                      blocks_per_group).mean(axis=2).astype(self.scale_dtype)

        # Handle zero points - ALWAYS create qzeros for quantized tensors
        if bits == 8:
            # For 8-bit quantization
            if block_zeros is not None:
                # Asymmetric: average zero points from blocks per group (vectorized)
                zp_float = block_zeros.reshape(out_dim, n_groups,
                                               blocks_per_group).astype(np.float32).mean(axis=2)
                zp_int = np.round(zp_float).astype(np.int32)
                zp_int = np.clip(zp_int, -128, 127)
                qzeros = ((zp_int + 128) & 0xFF).astype(np.uint8)
            else:
                # Symmetric: zero point is 128 for uint8 representation
                qzeros = np.full((out_dim, n_groups), 128, dtype=np.uint8)
        elif bits == 4:
            # For 4-bit quantization, create unpacked zero points per group
            qzeros = np.zeros((out_dim, n_groups), dtype=np.uint8)

            if block_zeros is not None:
                # For 4-bit with zero points (e.g., Q4_1) (vectorized)
                zp_float = block_zeros.reshape(out_dim, n_groups,
                                               blocks_per_group).astype(np.float32).mean(axis=2)
                zp_int = np.round(zp_float).astype(np.int32)
                zp_int = np.clip(zp_int, 0, 15)
                qzeros = zp_int.astype(np.uint8)
            else:
                # For symmetric 4-bit (Q4_0), zero point is 8
                qzeros.fill(8)
        else:
            # Should not happen (bits should be 4 or 8)
            raise ValueError(f"Unsupported bits: {bits}")

        result = {
            'qweight': qweight,
            'scales': scales,
            'qzeros': qzeros,
            'bits': bits,
            'group_size': group_size,
        }

        return result

    def convert_to_llmconv_format(self,
                                  gguf_loader,
                                  key: str,
                                  transpose: bool = True) -> Dict[str, np.ndarray]:
        """High-level conversion from GGUF to LlmConverter format.

        Args:
            gguf_loader: GGUFQuantLoad instance
            key: Tensor key name
            transpose: Whether to transpose weight matrix (typical for linear layers)

        Returns:
            Dictionary with converted tensors
        """
        # Get tensor info
        tensor_info = gguf_loader.get_tensor_info(key)
        # print(f' tensor info of {key} is {tensor_info}')
        if not tensor_info:
            raise RuntimeError(f"Tensor {key} not found")

        # import pdb;pdb.set_trace()
        # Read quantized data
        gguf_data, quant_info = gguf_loader.read_quantized(key)

        if not quant_info['is_quantized']:
            # Not quantized, return as-is
            data = gguf_loader.read(key)
            if transpose:
                data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            return {'weight': data, 'is_quantized': False}

        # Get original shape
        original_shape = quant_info['original_shape']

        # Convert GGML dimension order to numpy dimension order
        # GGUF stores dims as [outermost, innermost] but data is in numpy order [rows, cols]
        # The transpose here is needed for _reshape_gguf_data to work correctly
        if len(original_shape) == 2:
            original_shape = (original_shape[1], original_shape[0])

        # Use per-quant-type group_size (16 for Q6_K, 32 for Q4_K and others)
        quant_type = quant_info['quant_type']
        effective_group_size = get_quant_type_group_size(quant_type)

        # Convert quantization format
        try:
            converted = self.convert_from_gguf(gguf_data,
                                               quant_type,
                                               original_shape,
                                               group_size=effective_group_size)
            converted['is_quantized'] = True
            return converted
        except (NotImplementedError, ValueError) as e:
            # Unsupported quantization type or shape issue
            logger.warning("Cannot convert %s with quantization type %s: %s", key,
                           quant_info['quant_type'], e)
            logger.debug("  original_shape: %s", quant_info['original_shape'])
            logger.debug("  gguf_data shape: %s, size: %s", gguf_data.shape, gguf_data.size)
            logger.debug("  block_size: %s", quant_info.get('block_size', 'N/A'))
            logger.info("Falling back to dequantized float32 for %s", key)
            data = gguf_loader.read(key)  # This dequantizes to float32
            if transpose:
                data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            return {'weight': data, 'is_quantized': False}
