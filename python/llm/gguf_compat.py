# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Pure-Python replacements for the small ``gguf`` constants used across the
LLM conversion pipeline.

The full ``gguf`` package is only needed on the GGUF code path (it provides
``GGUFReader`` to parse ``.gguf`` binaries and ``gguf.quants.dequantize`` to
decode ~24 quantization formats).  The handful of *constants* every converter
imports, however, are trivial: an :class:`enum.IntEnum` of quantization type
ids, a ``{(block_size, type_size)}`` table, and the default alignment.

Re-implementing them here lets the common (safetensors / HuggingFace) path be
imported without ``gguf`` being installed.  The heavy reader/dequantize code
remains a lazy import inside :mod:`python.llm.GGUFQuantLoad`.

The integer values intentionally mirror ``gguf.GGMLQuantizationType`` exactly.
Because :class:`enum.IntEnum` members compare equal (and hash equal) to plain
ints of the same value, a ``gguf`` enum returned by ``GGUFReader`` on
``tensor.tensor_type`` compares equal to the corresponding member here, so
``tensor.tensor_type in [GGMLQuantizationType.Q4_K, ...]`` and
``GGML_QUANT_SIZES[tensor.tensor_type]`` keep working whether the enum came
from ``gguf`` or from this module.
"""

from enum import IntEnum


class GGMLQuantizationType(IntEnum):
    """Quantization type ids (mirrors ``gguf.GGMLQuantizationType``)."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41


# {type: (block_size, type_size)} in bytes, mirrors gguf.constants.GGML_QUANT_SIZES.
# block_size = number of values per block; type_size = bytes stored per block.
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.Q4_0: (32, 18),
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q5_0: (32, 22),
    GGMLQuantizationType.Q5_1: (32, 24),
    GGMLQuantizationType.Q8_0: (32, 34),
    GGMLQuantizationType.Q8_1: (32, 40),
    GGMLQuantizationType.Q2_K: (256, 84),
    GGMLQuantizationType.Q3_K: (256, 110),
    GGMLQuantizationType.Q4_K: (256, 144),
    GGMLQuantizationType.Q5_K: (256, 176),
    GGMLQuantizationType.Q6_K: (256, 210),
    GGMLQuantizationType.Q8_K: (256, 292),
    GGMLQuantizationType.IQ2_XXS: (256, 66),
    GGMLQuantizationType.IQ2_XS: (256, 74),
    GGMLQuantizationType.IQ3_XXS: (256, 98),
    GGMLQuantizationType.IQ1_S: (256, 50),
    GGMLQuantizationType.IQ4_NL: (32, 18),
    GGMLQuantizationType.IQ3_S: (256, 110),
    GGMLQuantizationType.IQ2_S: (256, 82),
    GGMLQuantizationType.IQ4_XS: (256, 136),
    GGMLQuantizationType.I8: (1, 1),
    GGMLQuantizationType.I16: (1, 2),
    GGMLQuantizationType.I32: (1, 4),
    GGMLQuantizationType.I64: (1, 8),
    GGMLQuantizationType.F64: (1, 8),
    GGMLQuantizationType.IQ1_M: (256, 56),
    GGMLQuantizationType.BF16: (1, 2),
    GGMLQuantizationType.TQ1_0: (256, 54),
    GGMLQuantizationType.TQ2_0: (256, 66),
    GGMLQuantizationType.MXFP4: (32, 17),
    GGMLQuantizationType.NVFP4: (64, 36),
    GGMLQuantizationType.Q1_0: (128, 18),
}

# GGUF binary files align each tensor's data to this many bytes.
GGUF_DEFAULT_ALIGNMENT = 32
