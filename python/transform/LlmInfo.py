# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List
from enum import Enum


class ModelConfig:

    def __init__(self,
                 num_attention_heads: str = 'num_attention_heads',
                 num_hidden_layers: str = 'num_hidden_layers',
                 num_key_value_heads: str = 'num_key_value_heads',
                 hidden_size: str = 'hidden_size',
                 vocab_size: str = 'vocab_size',
                 intermediate_size: str = 'intermediate_size',
                 rope_theta: str = "rope_theta",
                 rms_norm_eps: str = "rms_norm_eps"):
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps


class LlmList:
    LAYERS = "LAYERS"
    EMBEDING = "EMBEDING"
    # =========== in layers ==========
    INPUT_LN = "INPUT_LN"
    Q_PROJ = "Q_PROJ"
    K_PROJ = "K_PROJ"
    V_PROJ = "V_PROJ"
    O_PROJ = "O_PROJ"
    POST_LN = "POST_LN"
    MLP_GATE = "MLP_GATE"
    MLP_UP = "MLP_UP"
    MLP_DOWN = "MLP_DOWN"
    # ===============================
    NORM = "NORM"
    LMHEAD = "LMHEAD"


class ModelInfo:

    def __init__(self, config: ModelConfig, weights: map):
        self.config = config
        self.weights = weights

# qwen2/llama
COMMON_INFO = ModelInfo(
    ModelConfig(),
    weights={
        LlmList.LAYERS: "model.layers",
        LlmList.EMBEDING: "model.embed_tokens",
        # ========= in layers =============
        LlmList.INPUT_LN: "input_layernorm",
        LlmList.Q_PROJ: "self_attn.q_proj",
        LlmList.K_PROJ: "self_attn.k_proj",
        LlmList.V_PROJ: "self_attn.v_proj",
        LlmList.O_PROJ: "self_attn.o_proj",
        LlmList.POST_LN: "post_attention_layernorm",
        LlmList.MLP_GATE: "mlp.gate_proj",
        LlmList.MLP_UP: "mlp.up_proj",
        LlmList.MLP_DOWN: "mlp.down_proj",
        # ================================
        LlmList.NORM: "model.norm",
        LlmList.LMHEAD: "lm_head",
    })
