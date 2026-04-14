# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# LLM Converter Modules

# Import key classes for easier access
from .LlmConverter import LlmConverter
from .LlmInfo import ModelConfig, LlmType
from .LlmLoad import LlmLoad


# Define LlmInfo function since it's not defined in LlmInfo.py
def LlmInfo(llm_type: str):
    """Get ModelInfo for the given LLM type.
    
    Args:
        llm_type: One of the LlmType values
        
    Returns:
        ModelInfo instance for the given LLM type
    """
    # Import here to avoid circular imports
    from .LlmInfo import (COMMON_INFO, MLLAMA_INFO, CHATGLM3_INFO, GEMMA3_INFO, GLM4V_INFO,
                          MINICPMV_INFO)

    # Map LlmType values to INFO constants
    if llm_type == LlmType.MLLAMA:
        return MLLAMA_INFO
    elif llm_type == LlmType.CHATGLM3:
        return CHATGLM3_INFO
    elif llm_type == LlmType.GEMMA3:
        return GEMMA3_INFO
    elif llm_type == LlmType.GLM4V:
        return GLM4V_INFO
    elif llm_type == LlmType.MINICPM4:
        return MINICPMV_INFO
    elif llm_type in (LlmType.QWEN3, LlmType.QWEN2, LlmType.LLAMA):
        # Qwen3, Qwen2, and Llama use COMMON_INFO
        return COMMON_INFO
    else:
        # Default to COMMON_INFO for unknown types
        return COMMON_INFO


# Monkey-patch LlmInfo into the LlmInfo module so that
# 'from .LlmInfo import *' in LlmConverter.py imports LlmInfo
import sys
# Get the LlmInfo module that was already imported
if 'llm.LlmInfo' in sys.modules:
    LlmInfo_module = sys.modules['llm.LlmInfo']
    # Add LlmInfo function to the module
    setattr(LlmInfo_module, 'LlmInfo', LlmInfo)

# GGUF internals
from .GGUFQuantLoad import GGUFQuantLoad
from .QuantConverter import QuantConverter
from .ModelHandle import ModelHandle, SafetensorsModelHandle, GGUFModelHandle, create_gguf_config

# Other converters (for reference)
from .Qwen3_5Converter import Qwen3_5Converter
from .Qwen2VLConverter import Qwen2VLConverter
from .Qwen2_5VLConverter import Qwen2_5VLConverter
from .Qwen3VLConverter import Qwen3VLConverter
from .Qwen2_5OConverter import Qwen2_5OConverter

__all__ = [
    # Core
    'LlmConverter',
    'LlmInfo',
    'ModelConfig',
    'LlmType',
    'LlmLoad',

    # Model Handles
    'ModelHandle',
    'SafetensorsModelHandle',
    'GGUFModelHandle',
    'create_gguf_config',

    # GGUF internals
    'GGUFQuantLoad',
    'QuantConverter',

    # Other converters
    'Qwen3_5Converter',
    'Qwen2VLConverter',
    'Qwen2_5VLConverter',
    'Qwen3VLConverter',
    'Qwen2_5OConverter',
]
