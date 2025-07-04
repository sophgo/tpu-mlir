# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


class ModelConfig:

    def __init__(self,
                 num_attention_heads: str = 'num_attention_heads',
                 num_hidden_layers: str = 'num_hidden_layers',
                 num_key_value_heads: str = 'num_key_value_heads',
                 hidden_size: str = 'hidden_size',
                 vocab_size: str = 'vocab_size',
                 intermediate_size: str = 'intermediate_size',
                 rope_theta: str = "rope_theta",
                 rms_norm_eps: str = "rms_norm_eps",
                 hidden_act: str = "hidden_act",
                 quantization_config: str = "quantization_config"):
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.quantization_config = quantization_config


# only for llm, not for vlm
class LlmType:
    QWEN2 = "qwen2"
    LLAMA = "llama"
    QWEN3 = "qwen3"
    CHATGLM3 = "chatglm"
    GEMMA3 = "gemma3_text"
    MINICPM4 = "minicpm"


class ActType:
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"
    TANH = "tanh"
    QUICK_GELU = "quick_gelu"
    GELU_PYTORCH_TANH = "gelu_pytorch_tanh"


class WeightType:
    NORMAL = "Normal"
    RMS_NORM = "RMSNorm"
    LAYER_NORM = "LayerNorm"


class LlmList:
    LAYERS = "LAYERS"
    EMBEDING = "EMBEDING"
    # =========== in layers ==========
    INPUT_LN = "INPUT_LN"
    Q_PROJ = "Q_PROJ"
    Q_NORM = "Q_NORM"  # qwen3
    K_PROJ = "K_PROJ"
    K_NORM = "K_NORM"  # qwen3
    V_PROJ = "V_PROJ"
    O_PROJ = "O_PROJ"
    QKV_WB = "QKV_WB"  # chatglm
    ATT_D = "ATT_D"  # chatglm
    POST_ATTN_LN = "POST_ATTN_LN"
    # MLP
    PRE_MLP_LN = "PRE_MLP_LN"  # gemma3
    POST_MLP_LN = "POST_MLP_LN"  # gemma3
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


# qwen3/qwen2/llama
COMMON_INFO = ModelInfo(
    ModelConfig(),
    weights={
        LlmList.LAYERS: "model.layers",
        LlmList.EMBEDING: "model.embed_tokens",
        # ========= in layers =============
        LlmList.INPUT_LN: "input_layernorm",
        LlmList.Q_PROJ: "self_attn.q_proj",
        LlmList.Q_NORM: "self_attn.q_norm",  # qwen3
        LlmList.K_PROJ: "self_attn.k_proj",
        LlmList.K_NORM: "self_attn.k_norm",  # qwen3
        LlmList.V_PROJ: "self_attn.v_proj",
        LlmList.O_PROJ: "self_attn.o_proj",
        LlmList.POST_ATTN_LN: "post_attention_layernorm",
        LlmList.MLP_GATE: "mlp.gate_proj",
        LlmList.MLP_UP: "mlp.up_proj",
        LlmList.MLP_DOWN: "mlp.down_proj",
        # ================================
        LlmList.NORM: "model.norm",
        LlmList.LMHEAD: "lm_head",
    })

# Phi3
PHI3_INFO = ModelInfo(
    ModelConfig(),
    weights={
        LlmList.LAYERS: "model.layers",
        LlmList.EMBEDING: "model.embed_tokens",
        # ========= in layers =============
        LlmList.INPUT_LN: "input_layernorm",
        LlmList.QKV_WB: "self_attn.qkv_proj",
        LlmList.ATT_D: "self_attn.o_proj",
        LlmList.POST_ATTN_LN: "post_attention_layernorm",
        LlmList.MLP_UP: "mlp.gate_up_proj",
        LlmList.MLP_DOWN: "mlp.down_proj",
        # ================================
        LlmList.NORM: "model.norm",
        LlmList.LMHEAD: "lm_head",
    })

# chatglm3
CHATGLM3_INFO = ModelInfo(
    ModelConfig(intermediate_size="ffn_hidden_size",
                rms_norm_eps="layernorm_epsilon",
                num_key_value_heads="multi_query_group_num",
                num_hidden_layers="num_layers"),
    weights={
        LlmList.LAYERS: "transformer.encoder.layers",
        LlmList.EMBEDING: "transformer.embedding.word_embeddings",
        # ========= in layers =============
        LlmList.INPUT_LN: "input_layernorm",
        LlmList.QKV_WB: "self_attention.query_key_value",
        LlmList.ATT_D: "self_attention.dense",
        LlmList.POST_ATTN_LN: "post_attention_layernorm",
        LlmList.MLP_UP: "mlp.dense_h_to_4h",
        LlmList.MLP_DOWN: "mlp.dense_4h_to_h",
        # ================================
        LlmList.NORM: "transformer.encoder.final_layernorm",
        LlmList.LMHEAD: "transformer.output_layer",
    })

# gemma3
GEMMA3_INFO = ModelInfo(
    ModelConfig(hidden_act="hidden_activation", ),
    weights={
        LlmList.LAYERS: "language_model.model.layers",
        LlmList.EMBEDING: "language_model.model.embed_tokens",
        # ========= in layers =============
        LlmList.INPUT_LN: "input_layernorm",
        LlmList.Q_PROJ: "self_attn.q_proj",
        LlmList.Q_NORM: "self_attn.q_norm",
        LlmList.K_PROJ: "self_attn.k_proj",
        LlmList.K_NORM: "self_attn.k_norm",
        LlmList.V_PROJ: "self_attn.v_proj",
        LlmList.O_PROJ: "self_attn.o_proj",
        LlmList.POST_ATTN_LN: "post_attention_layernorm",
        LlmList.PRE_MLP_LN: "pre_feedforward_layernorm",
        LlmList.POST_MLP_LN: "post_feedforward_layernorm",
        LlmList.MLP_GATE: "mlp.gate_proj",
        LlmList.MLP_UP: "mlp.up_proj",
        LlmList.MLP_DOWN: "mlp.down_proj",
        # ================================
        LlmList.NORM: "language_model.model.norm",
        LlmList.LMHEAD: "language_model.model.lm_head",
    })
