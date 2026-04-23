# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
import os
import re

from .LlmInfo import WeightType
from transform.MLIRImporter import Platform
from .LlmInfo import LlmType
import logging

logger = logging.getLogger(__name__)


class ModelHandle(ABC):

    @abstractmethod
    def read(self, key: str) -> np.ndarray:
        pass

    @abstractmethod
    def is_exist(self, key: str) -> bool:
        pass

    def get_tensor_info(self, key: str):
        return None

    def read_quantized(self, key: str):
        return self.read(key), {"is_quantized": False}

    def init_quantization(self, conv):
        pass

    def gen_config(self, conv):
        pass

    def set_linear_weight(self, conv, path: str, weight_dict: dict, do_lora: bool = False):
        pass

    def set_common_weight(self, conv, path: str, weight_dict: dict, type=None):
        pass

    def compile_block_args(self, conv, layer_id, is_cache=False):
        return None


from .LlmLoad import LlmLoad
from .LlmInfo import LlmList
import torch


class SafetensorsModelHandle(ModelHandle):

    def __init__(self, model_path: str):
        self.model = LlmLoad(model_path)
        self.model_path = model_path

    def read(self, key: str) -> np.ndarray:
        return self.model.read(key)

    def is_exist(self, key: str) -> bool:
        return self.model.is_exist(key)

    def init_quantization(self, conv):
        c = conv.model_info.config
        conv.quantization_config = getattr(conv.llm_config, c.quantization_config, None)
        dtype = conv.get_dtype()
        if conv.quantization_config is None:
            conv.quantization_config = getattr(conv.config, c.quantization_config, None)
        real_quantize = None
        if conv.quantization_config is None:
            if conv.quantize == "auto":
                raise RuntimeError("No quantization config found, please set quantize type")
            real_quantize = conv.get_qtype(dtype, 16)
            if real_quantize is None:
                real_quantize = conv.quantize
            conv.half_precision_quantize = "bf16" if "bf16" in real_quantize else "f16"
            if conv.half_precision_quantize not in conv.quantize:
                raise RuntimeError(f"Quantize {conv.quantize} mismatch with model dtype :{dtype}")
        else:
            conv.quant_mode = conv.quantization_config["quant_method"]
            conv.platform = Platform.LLM_QUANTIZED
            if conv.quant_mode not in ["gptq", "awq", "compressed-tensors", "auto-round"]:
                raise NotImplementedError(f"Not support quantization method: {conv.quant_mode}")
            if conv.quant_mode != "compressed-tensors":
                conv.q_group_size = conv.quantization_config["group_size"]
                conv.quant_bits = conv.quantization_config["bits"]
            if conv.quant_mode == "auto-round":
                packing_format = conv.quantization_config.get("packing_format",
                                                              "auto_round:auto_gptq")
                if packing_format == "auto_round:auto_gptq":
                    conv.quant_mode = "gptq"
                elif packing_format == "auto_round:auto_awq":
                    conv.quant_mode = "awq"
                else:
                    raise NotImplementedError(f"Not support packing_format: {packing_format}")
            if conv.quant_mode == "awq":
                assert conv.quantization_config["version"] == "gemm", (
                    "AWQ only support gemm version for now")
                assert conv.quant_bits == 4, ("AWQ only support quant bits == 4 for now")
                if conv.quantize != "w4f16" and conv.quantize != "auto":
                    logger.warning("AWQ only support w4f16 quantize, change quantize to w4f16")
                real_quantize = "w4f16"
            elif conv.quant_mode == "compressed-tensors":
                format = conv.quantization_config.get("format", "pack-quantized")
                quantization_status = conv.quantization_config.get("quantization_status",
                                                                   "compressed")
                if format != "pack-quantized" and quantization_status != "compressed":
                    raise NotImplementedError("Only support compressed pack-quantized now")
                config_groups = conv.quantization_config.get("config_groups", {})
                assert len(config_groups) == 1, "Only support one group config now"
                group_0 = config_groups.get("group_0", {})
                weights_config = group_0.get("weights", {})
                conv.quant_bits = weights_config.get("num_bits")
                conv.q_group_size = weights_config.get("group_size")
                conv.compressed_with_zp = weights_config.get("symmetric", True) is False
                weight_type = weights_config.get("type")
                assert (weight_type == "int")
                real_quantize = conv.get_qtype(dtype, conv.quant_bits)
            elif conv.quant_mode == "gptq":
                real_quantize = conv.get_qtype(dtype, conv.quant_bits)
            if conv.quantize != "auto" and conv.quantize != real_quantize:
                logger.warning("%s is different from quantization config %s. Force to %s",
                               conv.quantize, real_quantize, real_quantize)
            conv.quantize = real_quantize
            conv.half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"
        if conv.q_group_size < 0:
            conv.q_group_size = 0

    def gen_config(self, conv):
        import shutil
        if conv.config_dir.startswith(os.path.abspath(conv.model_path)):
            os.rmdir(conv.bmodel_dir)
            os.rmdir(conv.out_dir)
            raise RuntimeError("Can't run under original model path!")
        shutil.copytree(conv.model_path,
                        conv.config_dir,
                        ignore=shutil.ignore_patterns("*.safetensors", ".*", "*.pth", "*.pt",
                                                      "*.py", "*.bin", "*.bin.index.json",
                                                      "model.safetensors.index.json"),
                        dirs_exist_ok=True)

    def unpack_weights(self, conv, qweight, qzeros, bits, quant_mode, path):
        dtype = np.int32
        compress_ratio = 32 // bits
        mask = 0xF if bits == 4 else 0xFF
        K, N = qweight.shape
        Kz, Nz = qzeros.shape
        unpacked_zeros = np.zeros((Kz, Nz * compress_ratio), dtype=np.uint8)
        need_int8_zeros = False
        if conv.fused_mlp:
            if conv.model_info.weights[LlmList.MLP_GATE] in path or conv.model_info.weights[
                    LlmList.MLP_UP] in path or conv.model_info.weights[LlmList.MLP_DOWN] in path:
                need_int8_zeros = True
            if conv.model_info.weights[
                    LlmList.SHARED_EXPERT_GATE] in path or conv.model_info.weights[
                        LlmList.SHARED_EXPERT_UP] in path or conv.model_info.weights[
                            LlmList.SHARED_EXPERT_DOWN] in path:
                need_int8_zeros = True
        if conv.llm_type in [LlmType.QWEN3_5_MOE, LlmType.QWEN2_MOE]:
            if conv.check_experts_gate_up(path) or conv.check_experts_down(path):
                need_int8_zeros = True

        if quant_mode == "gptq":
            unpacked_weights = np.zeros((K * compress_ratio, N), dtype=dtype)
            pack_int8_weights = np.zeros((K * compress_ratio // 2, N), dtype=np.uint8)
            order_map = [i for i in range(compress_ratio)]
            for row in range(unpacked_weights.shape[0]):
                i = order_map[row % compress_ratio]
                unpacked_weights[row, :] = (qweight[row // compress_ratio, :] >> (bits * i)) & mask
                if bits == 4:
                    if row % 2 == 0:
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]
        elif quant_mode == "awq":
            unpacked_weights = np.zeros((K, N * compress_ratio), dtype=dtype)
            pack_int8_weights = np.zeros((K // 2, N * compress_ratio), dtype=np.uint8)
            order_map = [0, 4, 1, 5, 2, 6, 3, 7]
            for col in range(unpacked_weights.shape[1]):
                i = order_map[col % compress_ratio]
                unpacked_weights[:, col] = (qweight[:, col // compress_ratio] >> (bits * i)) & mask
            if bits == 4:
                for row in range(unpacked_weights.shape[0]):
                    if row % 2 == 0:
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]
        else:
            raise NotImplementedError(f"Not support now: {quant_mode}")

        for col in range(unpacked_zeros.shape[1]):
            i = order_map[col % compress_ratio]
            unpacked_zeros[:, col] = (qzeros[:, col // compress_ratio] >> (bits * i)) & mask

        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")

        if need_int8_zeros:
            pack_int8_zeros = np.zeros((Kz // 2, Nz * compress_ratio), dtype=np.uint8)
            if quant_mode == "gptq":
                unpacked_zeros += 1
            if bits == 4:
                for row in range(unpacked_zeros.shape[0]):
                    if row % 2 == 0:
                        pack_int8_zeros[row // 2, :] = unpacked_zeros[row, :]
                    else:
                        pack_int8_zeros[row //
                                        2, :] = unpacked_zeros[row, :] << 4 | pack_int8_zeros[row //
                                                                                              2, :]
            return unpacked_weights, pack_int8_weights, pack_int8_zeros

        if quant_mode == "gptq":
            return unpacked_weights, pack_int8_weights, unpacked_zeros + 1
        else:
            return unpacked_weights, pack_int8_weights, unpacked_zeros

    def decompressed_weights(self, conv, weight_packed, weight_scale, qzeros):
        N, K = weight_packed.shape
        Ns, Ks = weight_scale.shape
        assert (N == Ns)
        bits = conv.quant_bits
        compress_ratio = 32 // bits
        mask = 0xF if bits == 4 else 0xFF
        unpacked_weights = np.zeros((N, K * compress_ratio), dtype=np.int32)
        pack_int8_weights = np.zeros((N, K * compress_ratio // 2), dtype=np.uint8)
        unpacked_zeros = np.zeros((Ns, Ks), dtype=np.uint8)
        order_map = [i for i in range(compress_ratio)]
        for row in range(unpacked_weights.shape[1]):
            i = order_map[row % compress_ratio]
            unpacked_weights[:, row] = (weight_packed[:, row // compress_ratio] >>
                                        (bits * i)) & mask
            if bits == 4:
                if row % 2 == 0:
                    pack_int8_weights[:, row // 2] = unpacked_weights[:, row]
                else:
                    pack_int8_weights[:, row //
                                      2] = unpacked_weights[:, row] << 4 | pack_int8_weights[:,
                                                                                             row //
                                                                                             2]
        if qzeros is not None:
            for col in range(unpacked_zeros.shape[0]):
                i = order_map[col % compress_ratio]
                unpacked_zeros[col, :] = (qzeros[col // compress_ratio, :] >> (bits * i)) & mask
        else:
            unpacked_zeros.fill((1 << (bits - 1)))

        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")
        return unpacked_weights, pack_int8_weights, unpacked_zeros

    def set_linear_weight(self, conv, path: str, weight_dict: dict, do_lora: bool = False):
        is_quant = False
        K, N = 0, 0
        if conv.quant_mode is not None:
            if self.model.is_exist(path + ".qweight") or self.model.is_exist(path +
                                                                             ".weight_packed"):
                is_quant = True
        if not is_quant:
            weight_path = path + ".weight"
            if self.model.is_exist(weight_path):
                data = self.model.read(weight_path)
                if conv.fused_mlp and (conv.model_info.weights[LlmList.MLP_GATE] in path
                                       or conv.model_info.weights[LlmList.MLP_UP] in path):
                    weight_dict[weight_path] = np.ascontiguousarray(data)
                else:
                    weight_dict[weight_path] = np.ascontiguousarray(np.transpose(data, (1, 0)))
                K = data.shape[1]
                N = data.shape[0]
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        elif conv.quant_mode in ["gptq", "awq"]:
            qweight_path = path + ".qweight"
            scale_path = path + ".scales"
            zp_path = path + ".qzeros"
            qweight_data = self.model.read(qweight_path)
            scale_data = self.model.read(scale_path)
            zp_data = self.model.read(zp_path)
            _, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                conv, qweight_data, zp_data, conv.quant_bits, conv.quant_mode, path)
            if conv.fused_mlp and (conv.model_info.weights[LlmList.MLP_DOWN] in path):
                weight_dict[qweight_path] = np.ascontiguousarray(
                    np.transpose(pack_int8_weights.reshape(-1, conv.q_group_size, conv.hidden_size),
                                 (0, 2, 1)).reshape(-1, conv.hidden_size * conv.q_group_size))
                weight_dict[scale_path] = np.ascontiguousarray(scale_data)
                weight_dict[zp_path] = np.ascontiguousarray(unpacked_zeros)
            else:
                weight_dict[qweight_path] = np.ascontiguousarray(
                    np.transpose(pack_int8_weights, (1, 0)))
                weight_dict[scale_path] = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))
                weight_dict[zp_path] = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
            K = pack_int8_weights.shape[0] * (8 // conv.quant_bits)
            N = pack_int8_weights.shape[1]
        elif conv.quant_mode == "compressed-tensors":
            qweight_path = path + ".weight_packed"
            scale_path = path + ".weight_scale"
            zp_path = path + ".weight_zero_point"
            qweight_data = self.model.read(qweight_path)
            scale_data = self.model.read(scale_path)
            if conv.compressed_with_zp:
                zp_data = self.model.read(zp_path)
            else:
                zp_data = None
            _, pack_int8_weights, unpacked_zeros = self.decompressed_weights(
                conv, qweight_data, scale_data, zp_data)
            weight_dict[path + ".qweight"] = pack_int8_weights
            weight_dict[path + ".scales"] = scale_data
            weight_dict[path + ".qzeros"] = unpacked_zeros
            K = pack_int8_weights.shape[1] * (8 // conv.quant_bits)
            N = pack_int8_weights.shape[0]

        bias_path = path + ".bias"
        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)
        if do_lora:
            conv.set_linear_lora_weight(weight_dict, path, K, N)

    def set_common_weight(self, conv, path: str, weight_dict: dict, type=None):
        weight_path = path + ".weight"
        bias_path = path + ".bias"
        has_weight = self.model.is_exist(weight_path)
        has_bias = self.model.is_exist(bias_path)
        has_path = self.model.is_exist(path)
        if not has_weight and not has_bias and not has_path:
            raise RuntimeError("Can't find key: {}".format(path))
        if has_weight:
            data = self.model.read(weight_path)
            if type == WeightType.ZEROCENTERED_RMSNORM:
                data = data + 1.0
            weight_dict[weight_path] = data
        if has_bias:
            weight_dict[bias_path] = self.model.read(bias_path)
        if has_path:
            weight_dict[path] = self.model.read(path)


from .GGUFQuantLoad import GGUFQuantLoad
from .QuantConverter import QuantConverter, get_quant_type_group_size
from gguf import GGMLQuantizationType


def get_gguf_group_size(gguf_reader):
    count_16 = 0
    count_32 = 0
    q6_k_type = GGMLQuantizationType.Q6_K
    float_types = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    }
    for tensor in gguf_reader.tensors:
        qt = tensor.tensor_type
        if qt in float_types:
            continue
        if qt == q6_k_type:
            count_16 += 1
        else:
            count_32 += 1
    if count_16 > count_32:
        return 16
    else:
        return 32


def create_gguf_config(gguf_reader, quantize='w8bf16', seq_length=4096, group_size=None):
    arch_field = gguf_reader.get_field("general.architecture")
    architecture = arch_field.contents() if arch_field else "qwen3"
    arch_lower = architecture.lower()

    def get_val(key):
        field = gguf_reader.get_field(key)
        return field.contents() if field else None

    emb = get_val(f"{architecture}.embedding_length")
    heads = get_val(f"{architecture}.attention.head_count")
    kv_heads = get_val(f"{architecture}.attention.head_count_kv")
    key_len = get_val(f"{architecture}.attention.key_length")
    ffn = get_val(f"{architecture}.feed_forward_length")
    rope = get_val(f"{architecture}.rope.freq_base")
    layers = get_val(f"{architecture}.block_count")
    eps = get_val(f"{architecture}.attention.layer_norm_epsilon")
    vocab = get_val(f"{architecture}.vocab_size")

    hidden_size = emb if emb else 1024
    num_attention_heads = heads if heads else 16
    num_key_value_heads = kv_heads if kv_heads else 8
    head_dim = key_len if key_len else (hidden_size //
                                        num_attention_heads if num_attention_heads else 64)
    intermediate_size = ffn if ffn else 3072
    num_hidden_layers = layers if layers else 28
    rms_norm_eps = eps if eps else 1e-6
    rope_theta = rope if rope else 1000000.0
    vocab_size = vocab if vocab else 151936

    if 'bf16' in quantize:
        dtype_str = "bfloat16"
    else:
        dtype_str = "float16"

    if group_size is None:
        group_size = get_gguf_group_size(gguf_reader)

    quantized_tensors = []
    quant_type_counts = {}

    for tensor in gguf_reader.tensors:
        if tensor.tensor_type not in [
                GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        ]:
            quantized_tensors.append(tensor.name)
            quant_type = tensor.tensor_type
            quant_type_counts[quant_type] = quant_type_counts.get(quant_type, 0) + 1

    quantization_config = None
    if quantized_tensors:
        bits = 8
        quant_method = "gptq"

        detected_bits = 8  # default
        for qt, count in quant_type_counts.items():
            if qt in [GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1]:
                detected_bits = 4
                break
            elif qt in [
                    GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1, GGMLQuantizationType.Q8_K,
                    GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q4_K
            ]:
                detected_bits = 8

        if 'w4' in quantize and detected_bits != 4:
            logger.error(
                "quantize=%s requires 4-bit weights, but GGUF model has %d-bit quantization (types: %s). Please use -q w8f16 or w8bf16 instead.",
                quantize, detected_bits, quant_type_counts)
            raise RuntimeError(
                f"quantize={quantize} mismatch with GGUF {detected_bits}-bit weights")
        elif 'w8' in quantize and detected_bits != 8:
            logger.error(
                "quantize=%s requires 8-bit weights, but GGUF model has %d-bit quantization (types: %s). Please use -q w4f16 or w4bf16 instead.",
                quantize, detected_bits, quant_type_counts)
            raise RuntimeError(
                f"quantize={quantize} mismatch with GGUF {detected_bits}-bit weights")
        if 'w4' in quantize:
            bits = 4
        elif 'w8' in quantize:
            bits = 8

        quantization_config = {
            "bits": bits,
            "group_size": group_size,
            "lm_head": False,
            "desc_act": False,
            "quant_method": quant_method,
            "sym": True,
            "pack_dtype": "int32",
            "checkpoint_format": "gguf",
            "meta": {
                "quantizer": ["llama.cpp:gguf-py"],
                "uri": "https://github.com/ggerganov/llama.cpp",
                "static_groups": False,
                "true_sequential": False,
                "mse": 0.0,
                "damp_percent": 0.01,
                "damp_auto_increment": 0.0025,
            }
        }

    try:
        from transformers import Qwen3Config, Qwen2Config, LlamaConfig, GemmaConfig
    except ImportError:
        config = object()
        config.vocab_size = vocab_size
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads
        config.head_dim = head_dim
        config.max_position_embeddings = seq_length
        config.rms_norm_eps = rms_norm_eps
        config.rope_theta = rope_theta
        config.hidden_act = "silu"
        config.dtype = dtype_str
        config.torch_dtype = None
        config.tie_word_embeddings = True
        config.model_type = architecture
        config.quantization_config = quantization_config
        return config

    config_class_map = {
        'qwen3': Qwen3Config,
        'qwen2': Qwen2Config,
        'llama': LlamaConfig,
        'llama3': LlamaConfig,
        'gemma': GemmaConfig,
        'gemma2': GemmaConfig,
    }

    if arch_lower not in config_class_map:
        config_class = Qwen3Config
    else:
        config_class = config_class_map[arch_lower]

    config = config_class(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_length,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act="silu",
        torch_dtype=dtype_str,
        tie_word_embeddings=True,
    )
    config.model_type = architecture
    if quantization_config:
        config.quantization_config = quantization_config

    return config


class GGUFModelHandle(ModelHandle):

    def __init__(self, model_path: str, args=None):
        self.model = GGUFQuantLoad(model_path)
        self.model_path = model_path
        self.args = args

        scale_dtype = np.float32
        group_size = args.q_group_size if hasattr(args, 'q_group_size') else 32
        self.quant_converter = QuantConverter(group_size=group_size, scale_dtype=scale_dtype)

        self.quantized_tensors = {}
        self._lmhead_float_fallback = False
        self._blocks_full_float_fallback = {}
        self._mixed_quant_fallback = False

    def read(self, key: str) -> np.ndarray:
        return self.model.read(key)

    def is_exist(self, key: str) -> bool:
        return self.model.is_exist(key)

    def get_tensor_info(self, key: str):
        return self.model.get_tensor_info(key)

    def read_quantized(self, key: str):
        return self.model.read_quantized(key)

    def init_quantization(self, conv):
        from gguf import GGMLQuantizationType

        quant_types_found = {}
        for tensor in self.model.reader.tensors:
            qt = tensor.tensor_type
            if qt in {
                    GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
            }:
                continue
            qt_name = qt.name if hasattr(qt, 'name') else str(qt)
            quant_types_found[qt_name] = quant_types_found.get(qt_name, 0) + 1

        has_q4_k = 'Q4_K' in quant_types_found
        has_q6_k = 'Q6_K' in quant_types_found
        self._mixed_quant_fallback = has_q4_k and has_q6_k
        if self._mixed_quant_fallback:
            logger.warning("Mixed quant types detected: %s", quant_types_found)
            logger.warning("  Q4_K needs group_size=32, Q6_K needs group_size=16")
            logger.warning(
                "  Blocks with group_size mismatch will have ALL linears fallback to float32.")

        test_tensors = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ]

        quantized_count = 0
        total_count = 0

        for tensor_name in test_tensors:
            if self.model.is_exist(tensor_name):
                total_count += 1
                tensor_info = self.model.get_tensor_info(tensor_name)
                if tensor_info and tensor_info['is_quantized']:
                    quantized_count += 1
                    self.quantized_tensors[tensor_name] = tensor_info

        if quantized_count > 0:
            conv.quant_mode = "gptq"
            conv.platform = Platform.LLM_QUANTIZED
            self._determine_quant_bits(conv)
            self._determine_symmetric(conv)
            if self.quantized_tensors:
                first_tensor = next(iter(self.quantized_tensors.values()))
                quant_type = first_tensor['quant_type']
                group_size = get_quant_type_group_size(quant_type)
            else:
                group_size = get_gguf_group_size(self.model.reader)
            self.quant_converter.group_size = group_size
            conv.q_group_size = group_size
            conv.half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"
            logger.info("Auto-detected GGUF group size: %d", group_size)
            if self._mixed_quant_fallback:
                logger.warning(
                    "  Note: Blocks with mismatched group_size will have ALL linears fallback to float"
                )
            self._detect_block_float_fallbacks(conv)
        else:
            conv.half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"

    def _determine_quant_bits(self, conv):
        if not self.quantized_tensors:
            return
        first_tensor = next(iter(self.quantized_tensors.values()))
        quant_type = first_tensor['quant_type']
        from gguf import GGMLQuantizationType

        # quant_bits reflects the OUTPUT format of the conversion, not the input.
        # K-quant types (Q4_K, Q5_K, Q6_K, Q8_K) all get re-quantized to 8-bit.
        # Simple quant types (Q4_0, Q4_1) output 4-bit packed format.
        if quant_type in [GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1]:
            conv.quant_bits = 4
        elif quant_type in [
                GGMLQuantizationType.Q8_0,
                GGMLQuantizationType.Q8_1,
                GGMLQuantizationType.Q5_0,
                GGMLQuantizationType.Q5_1,
                GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K,
                GGMLQuantizationType.Q8_K,
        ]:
            conv.quant_bits = 8
        elif quant_name.startswith('IQ'):
            conv.quant_bits = 4
        else:
            conv.quant_bits = 8

    def _determine_symmetric(self, conv):
        if not self.quantized_tensors:
            return
        first_tensor = next(iter(self.quantized_tensors.values()))
        quant_type = first_tensor['quant_type']
        from gguf import GGMLQuantizationType
        symmetric_types = {
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q5_0,
        }
        quant_name = quant_type.name if hasattr(quant_type, 'name') else str(quant_type)
        if quant_name.startswith('Q') and '_K' in quant_name:
            conv.symmetric = True
        elif quant_type in symmetric_types:
            conv.symmetric = True
        else:
            conv.symmetric = False
        logger.info("Auto-detected GGUF symmetric quantization: %s", conv.symmetric)

    def _detect_block_float_fallbacks(self, conv):
        num_layers = getattr(conv.llm_config, 'num_hidden_layers', 0)
        model_q_group_size = conv.q_group_size
        half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"

        for idx in range(num_layers):
            linear_paths = self._get_linear_paths_for_block(conv, idx)
            has_mismatch = False

            for path in linear_paths:
                tensor_info = self.model.get_tensor_info(path)
                if tensor_info and tensor_info.get('is_quantized', False):
                    quant_type = tensor_info.get('quant_type')
                    tensor_gs = get_quant_type_group_size(quant_type) if quant_type else 32
                    if tensor_gs != model_q_group_size:
                        has_mismatch = True
                        break

            if has_mismatch:
                self._blocks_full_float_fallback[idx] = True
                linear_names = [p.split('.')[-1] for p in linear_paths]
                logger.warning(
                    "  Block %d: group_size mismatch detected. ALL linears (%s) fallback to float. Will compile with %s instead of %s.",
                    idx, ', '.join(linear_names), half_precision_quantize, conv.quantize)
            else:
                self._blocks_full_float_fallback[idx] = False

        fallback_blocks = [idx for idx, v in self._blocks_full_float_fallback.items() if v]
        if fallback_blocks:
            logger.warning("  Total blocks with full float fallback: %d / %d", len(fallback_blocks),
                           num_layers)
            logger.warning("  Fallback block indices: %s", fallback_blocks)

    def _get_linear_paths_for_block(self, conv, idx: int):
        TOP_PATH = f'{conv.model_info.weights[LlmList.LAYERS]}.{idx}.'
        paths = []
        for key in [
                LlmList.Q_PROJ, LlmList.K_PROJ, LlmList.V_PROJ, LlmList.O_PROJ, LlmList.MLP_GATE,
                LlmList.MLP_UP, LlmList.MLP_DOWN
        ]:
            if key in conv.model_info.weights:
                paths.append(TOP_PATH + conv.model_info.weights[key])
        if LlmList.QKV_WB in conv.model_info.weights:
            paths.append(TOP_PATH + conv.model_info.weights[LlmList.QKV_WB])
        if LlmList.ATT_D in conv.model_info.weights:
            paths.append(TOP_PATH + conv.model_info.weights[LlmList.ATT_D])
        for key in [LlmList.C_Q_PROJ, LlmList.C_K_PROJ, LlmList.C_V_PROJ, LlmList.C_O_PROJ]:
            if key in conv.model_info.weights:
                paths.append(TOP_PATH + conv.model_info.weights[key])
        return paths

    def gen_config(self, conv):
        os.makedirs(conv.config_dir, exist_ok=True)
        os.makedirs(conv.bmodel_dir, exist_ok=True)
        gguf_reader = self.model.reader

        def get_gguf_val(key):
            field = gguf_reader.get_field(key)
            if field is None:
                return None
            try:
                return field.contents()
            except Exception:
                return None

        def decode_token(tok):
            if isinstance(tok, bytes):
                return tok.decode("utf-8", errors="replace")
            return tok

        arch = get_gguf_val("general.architecture") or "qwen3"
        tokenizer_model = get_gguf_val("tokenizer.ggml.model") or "gpt2"
        tokenizer_pre = get_gguf_val("tokenizer.ggml.pre") or "default"
        tokens_list = get_gguf_val("tokenizer.ggml.tokens") or []
        token_types = get_gguf_val("tokenizer.ggml.token_type") or []
        merges_list = get_gguf_val("tokenizer.ggml.merges") or []
        add_prefix_space = bool(get_gguf_val("tokenizer.ggml.add_space_prefix") or False)

        TOKEN_TYPE_NORMAL = 1
        TOKEN_TYPE_UNKNOWN = 2
        TOKEN_TYPE_CONTROL = 3
        TOKEN_TYPE_USER_DEFINED = 4
        TOKEN_TYPE_UNUSED = 5
        TOKEN_TYPE_BYTE = 6

        base_vocab_tokens = []
        added_tokens_entries = []
        for idx in range(len(tokens_list)):
            tok = decode_token(tokens_list[idx])
            tt = int(token_types[idx]) if idx < len(token_types) else TOKEN_TYPE_NORMAL
            if tt == TOKEN_TYPE_NORMAL:
                base_vocab_tokens.append((tok, idx))
            else:
                is_special = (tt == TOKEN_TYPE_CONTROL)
                added_tokens_entries.append({
                    "id": idx,
                    "content": tok,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": not is_special,
                    "special": is_special,
                })

        special_token_id_map = {}
        for key in [
                "bos_token_id", "eos_token_id", "unk_token_id", "sep_token_id", "pad_token_id",
                "eot_token_id", "eom_token_id", "mask_token_id"
        ]:
            gguf_key_map = {
                "bos_token_id": "tokenizer.ggml.bos_token_id",
                "eos_token_id": "tokenizer.ggml.eos_token_id",
                "unk_token_id": "tokenizer.ggml.unknown_token_id",
                "sep_token_id": "tokenizer.ggml.seperator_token_id",
                "pad_token_id": "tokenizer.ggml.padding_token_id",
                "eot_token_id": "tokenizer.ggml.eot_token_id",
                "eom_token_id": "tokenizer.ggml.eom_token_id",
                "mask_token_id": "tokenizer.ggml.mask_token_id",
            }
            val = get_gguf_val(gguf_key_map[key])
            if val is not None:
                special_token_id_map[key] = int(val)

        def get_special_token_str(token_id):
            if token_id is None or token_id < 0 or token_id >= len(tokens_list):
                return None
            return decode_token(tokens_list[token_id])

        arch_to_model_type = {
            "qwen3": "qwen3",
            "qwen2": "qwen2",
            "llama": "llama",
            "llama3": "llama",
            "gemma": "gemma",
            "gemma2": "gemma2",
        }
        model_type = arch_to_model_type.get(arch, arch)

        arch_to_tokenizer_class = {
            "qwen3": "Qwen2Tokenizer",
            "qwen2": "Qwen2Tokenizer",
            "llama": "LlamaTokenizer",
            "llama3": "LlamaTokenizer",
            "gemma": "GemmaTokenizer",
            "gemma2": "GemmaTokenizerFast",
        }
        tokenizer_class = arch_to_tokenizer_class.get(arch, "LlamaTokenizer")

        gguf_model_to_hf_model = {
            "gpt2": "BPE",
            "llama": "SentencePiece",
            "bert": "WordPiece",
            "t5": "Unigram",
        }
        hf_model_type = gguf_model_to_hf_model.get(tokenizer_model, tokenizer_model)

        PRE_TOKENIZER_CONFIGS = {
            "qwen2": {
                "type":
                "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {
                            "Regex":
                            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        },
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": add_prefix_space,
                        "trim_offsets": False,
                        "use_regex": False,
                    },
                ],
            },
            "default": {
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": True,
                "use_regex": True,
            },
            "llama3": {
                "type":
                "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {
                            "Regex":
                            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        },
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": add_prefix_space,
                        "trim_offsets": True,
                        "use_regex": False,
                    },
                ],
            },
        }

        DECODER_CONFIGS = {
            "qwen2": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True
            },
            "default": {
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": True,
                "use_regex": True
            },
            "llama3": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True
            },
        }

        POST_PROCESSOR_CONFIGS = {
            "qwen2": {
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": False,
                "use_regex": False
            },
            "default": {
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": True,
                "use_regex": True
            },
            "llama3": {
                "type": "ByteLevel",
                "add_prefix_space": add_prefix_space,
                "trim_offsets": False,
                "use_regex": False
            },
        }

        pre_tokenizer = PRE_TOKENIZER_CONFIGS.get(tokenizer_pre, PRE_TOKENIZER_CONFIGS["default"])
        decoder = DECODER_CONFIGS.get(tokenizer_pre, DECODER_CONFIGS["default"])
        post_processor = POST_PROCESSOR_CONFIGS.get(tokenizer_pre,
                                                    POST_PROCESSOR_CONFIGS["default"])

        config_dict = {
            "architectures": [f"{arch.title()}ForCausalLM"],
            "model_type": model_type,
        }

        gguf_config_keys = [
            (f"{arch}.embedding_length", "hidden_size"),
            (f"{arch}.block_count", "num_hidden_layers"),
            (f"{arch}.attention.head_count", "num_attention_heads"),
            (f"{arch}.attention.head_count_kv", "num_key_value_heads"),
            (f"{arch}.vocab_size", "vocab_size"),
            (f"{arch}.feed_forward_length", "intermediate_size"),
            (f"{arch}.context_length", "max_position_embeddings"),
            (f"{arch}.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
            (f"{arch}.rope.freq_base", "rope_theta"),
            (f"{arch}.rope.dimension_count", "rope_dimension_count"),
            (f"{arch}.attention.layer_norm_epsilon", "layer_norm_eps"),
        ]
        for gguf_key, hf_key in gguf_config_keys:
            val = get_gguf_val(gguf_key)
            if val is not None:
                config_dict[hf_key] = val

        if "hidden_size" in config_dict and "num_attention_heads" in config_dict:
            config_dict[
                "head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]

        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if key in special_token_id_map:
                config_dict[key] = special_token_id_map[key]

        config_dict["hidden_act"] = "silu"
        config_dict["initializer_range"] = 0.02
        config_dict["tie_word_embeddings"] = True
        config_dict["use_cache"] = True
        config_dict["torch_dtype"] = "bfloat16"
        config_dict["attention_bias"] = False
        config_dict["attention_dropout"] = 0.0

        quantization_config = getattr(conv.config, 'quantization_config', None)
        if quantization_config is not None:
            if isinstance(quantization_config, dict):
                config_dict["quantization_config"] = quantization_config
            elif hasattr(quantization_config, 'to_dict'):
                config_dict["quantization_config"] = quantization_config.to_dict()

        import json
        config_path = os.path.join(conv.config_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info("Saved config to %s", config_path)

        tokenizer_dict = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
        }
        tokenizer_dict["added_tokens"] = added_tokens_entries
        tokenizer_dict["normalizer"] = {"type": "NFC"}
        tokenizer_dict["pre_tokenizer"] = pre_tokenizer
        tokenizer_dict["post_processor"] = post_processor
        tokenizer_dict["decoder"] = decoder

        model_dict = {
            "type": hf_model_type,
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {
                tok: idx
                for tok, idx in base_vocab_tokens
            },
        }
        if hf_model_type == "BPE" and merges_list:
            model_dict["merges"] = [decode_token(m) for m in merges_list]

        tokenizer_dict["model"] = model_dict

        tokenizer_path = os.path.join(conv.config_dir, 'tokenizer.json')
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer_dict, f, indent=2)
        logger.info("Saved tokenizer to %s", tokenizer_path)

        bos_str = get_special_token_str(special_token_id_map.get("bos_token_id"))
        eos_str = get_special_token_str(special_token_id_map.get("eos_token_id"))
        unk_str = get_special_token_str(special_token_id_map.get("unk_token_id"))
        sep_str = get_special_token_str(special_token_id_map.get("sep_token_id"))
        pad_str = get_special_token_str(special_token_id_map.get("pad_token_id"))

        tokenizer_config_dict = {
            "tokenizer_class": tokenizer_class,
            "model_max_length": 131072,
            "clean_up_tokenization_spaces": False,
            "add_prefix_space": add_prefix_space,
            "errors": "replace",
            "split_special_tokens": False,
        }

        add_bos = get_gguf_val("tokenizer.ggml.add_bos_token")
        if add_bos is not None:
            tokenizer_config_dict["add_bos_token"] = bool(add_bos)

        if bos_str is not None:
            tokenizer_config_dict["bos_token"] = bos_str
        else:
            tokenizer_config_dict["bos_token"] = None
        if eos_str is not None:
            tokenizer_config_dict["eos_token"] = eos_str
        else:
            tokenizer_config_dict["eos_token"] = None
        if unk_str is not None:
            tokenizer_config_dict["unk_token"] = unk_str
        else:
            tokenizer_config_dict["unk_token"] = None
        if sep_str is not None:
            tokenizer_config_dict["sep_token"] = sep_str
        if pad_str is not None:
            tokenizer_config_dict["pad_token"] = pad_str

        added_tokens_decoder = {}
        additional_special_tokens = []
        for entry in added_tokens_entries:
            if entry["special"]:
                is_primary = (entry["id"] == special_token_id_map.get("bos_token_id")
                              or entry["id"] == special_token_id_map.get("eos_token_id")
                              or entry["id"] == special_token_id_map.get("unk_token_id")
                              or entry["id"] == special_token_id_map.get("sep_token_id")
                              or entry["id"] == special_token_id_map.get("pad_token_id"))
                if not is_primary:
                    additional_special_tokens.append(entry["content"])
            added_tokens_decoder[str(entry["id"])] = {
                "content": entry["content"],
                "lstrip": entry["lstrip"],
                "rstrip": entry["rstrip"],
                "normalized": entry["normalized"],
                "single_word": entry["single_word"],
                "special": entry["special"],
            }
        tokenizer_config_dict["added_tokens_decoder"] = added_tokens_decoder
        if additional_special_tokens:
            tokenizer_config_dict["additional_special_tokens"] = additional_special_tokens

        chat_template = get_gguf_val("tokenizer.chat_template")
        if chat_template is not None:
            tokenizer_config_dict["chat_template"] = chat_template

        tokenizer_config_path = os.path.join(conv.config_dir, 'tokenizer_config.json')
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config_dict, f, indent=2)
        logger.info("Saved tokenizer_config to %s", tokenizer_config_path)

        generation_config_dict = {}
        if "bos_token_id" in special_token_id_map:
            generation_config_dict["bos_token_id"] = special_token_id_map["bos_token_id"]
        if "eos_token_id" in special_token_id_map:
            generation_config_dict["eos_token_id"] = special_token_id_map["eos_token_id"]
        if "pad_token_id" in special_token_id_map:
            generation_config_dict["pad_token_id"] = special_token_id_map["pad_token_id"]

        generation_config_path = os.path.join(conv.config_dir, 'generation_config.json')
        with open(generation_config_path, 'w') as f:
            json.dump(generation_config_dict, f, indent=2)
        logger.info("Saved generation_config to %s", generation_config_path)

        vocab_dict = {tok: idx for tok, idx in base_vocab_tokens}
        vocab_path = os.path.join(conv.config_dir, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(vocab_dict, f)
        logger.info("Saved vocab to %s", vocab_path)

        if merges_list:
            merges_path = os.path.join(conv.config_dir, 'merges.txt')
            with open(merges_path, 'w') as f:
                f.write("#version: 0.2\n")
                for merge in merges_list:
                    f.write(decode_token(merge) + "\n")
            logger.info("Saved merges to %s", merges_path)

        configuration_dict = {
            "framework": "pytorch",
            "task": "text-generation",
            "allow_remote": True
        }
        configuration_path = os.path.join(conv.config_dir, 'configuration.json')
        with open(configuration_path, 'w') as f:
            json.dump(configuration_dict, f)
        logger.info("Saved configuration to %s", configuration_path)

    def set_linear_weight(self, conv, path: str, weight_dict: dict, do_lora: bool = False):
        tensor_info = self.model.get_tensor_info(path)
        logger.debug('tensor info of %s : %s', path, tensor_info)

        weight_shape = None

        if tensor_info and tensor_info.get('is_quantized', False):
            block_idx_match = re.search(r'model\.layers\.(\d+)', path)
            block_idx = int(block_idx_match.group(1)) if block_idx_match else -1
            if block_idx >= 0 and self._blocks_full_float_fallback.get(block_idx, False):
                tensor_gs = get_quant_type_group_size(
                    tensor_info.get('quant_type')) if tensor_info.get('quant_type') else 32
                logger.warning(
                    "%s (group_size=%d) falls back to float due to group_size mismatch in this block.",
                    path, tensor_gs)
                data = self.model.read(path)
                weight_dict[path + ".weight"] = np.ascontiguousarray(np.transpose(data, (1, 0)))
                weight_shape = weight_dict[path + ".weight"].shape
                return

            converted = self.quant_converter.convert_to_llmconv_format(self.model,
                                                                       path,
                                                                       transpose=True)

            if converted['is_quantized']:
                weight_dict[path + ".qweight"] = converted['qweight']
                weight_shape = converted['qweight'].shape
                weight_dict[path + ".scales"] = converted['scales']

                if 'qzeros' in converted:
                    weight_dict[path + ".qzeros"] = converted['qzeros']

                self.quantized_tensors[path] = {
                    **tensor_info,
                    'converted_bits': converted['bits'],
                    'converted_group_size': converted['group_size'],
                }
            else:
                weight_dict[path + ".weight"] = converted['weight']
                weight_shape = converted['weight'].shape
        else:
            weight_path = path + ".weight"
            if self.model.is_exist(weight_path):
                data = self.model.read(weight_path)
                weight_dict[weight_path] = np.ascontiguousarray(np.transpose(data, (1, 0)))
                weight_shape = weight_dict[weight_path].shape
            else:
                raise RuntimeError(f"Can't find key: {path}.weight")

        bias_path = path + ".bias"
        bias_exists = self.model.is_exist(bias_path)
        logger.debug("Bias exists %s? %s", bias_path, bias_exists)
        if bias_exists:
            try:
                gguf_name = self.model._map_key_to_gguf(bias_path)
                logger.debug("Bias mapping %s -> %s", bias_path, gguf_name)

                if gguf_name and gguf_name in self.model.tensor_map:
                    logger.debug("Mapped tensor %s exists in tensor map", gguf_name)
                    bias_data = self.model.read(bias_path)
                    logger.debug("Bias read shape %s, dtype %s", bias_data.shape, bias_data.dtype)
                    if len(bias_data.shape
                           ) == 2 and weight_shape is not None and bias_data.shape == weight_shape:
                        logger.warning(
                            "Bias tensor %s has weight matrix shape %s. Treating as missing bias.",
                            bias_path, bias_data.shape)
                    else:
                        if len(bias_data.shape) != 1:
                            bias_data = bias_data.flatten()
                        if path + ".qweight" in weight_dict:
                            out_dim = weight_dict[path + ".qweight"].shape[0]
                        elif path + ".weight" in weight_dict:
                            out_dim = weight_dict[path + ".weight"].shape[1]
                        else:
                            out_dim = bias_data.shape[0]
                        if bias_data.shape[0] != out_dim:
                            logger.warning(
                                "Bias length %d does not match output dimension %d. Truncating/padding.",
                                bias_data.shape[0], out_dim)
                            if bias_data.shape[0] > out_dim:
                                bias_data = bias_data[:out_dim]
                            else:
                                pad = np.zeros(out_dim - bias_data.shape[0], dtype=bias_data.dtype)
                                bias_data = np.concatenate([bias_data, pad])
                        weight_dict[bias_path] = bias_data
                else:
                    logger.debug("Mapped tensor %s doesn't exist or mapping failed. Skipping bias.",
                                 gguf_name)
            except RuntimeError as e:
                logger.warning("Failed to read bias tensor %s: %s. Skipping bias.", bias_path, e)
        else:
            logger.debug("Bias %s doesn't exist in GGUF. MLIR will use none_op.", bias_path)

        if do_lora:
            if path + ".weight" in weight_dict:
                weight_data = weight_dict[path + ".weight"]
                K = weight_data.shape[1]
                N = weight_data.shape[0]
                conv.set_linear_lora_weight(weight_dict, path, K, N)

    def set_common_weight(self, conv, path: str, weight_dict: dict, type=None):

        weight_path = path + ".weight"
        if self.model.is_exist(weight_path):
            data = self.model.read(weight_path)

            if 'q_norm' in path or 'k_norm' in path:
                data = data.reshape(1, 1, 1, conv.head_dim)

            if type == WeightType.ZEROCENTERED_RMSNORM:
                data = data + 1.0
            weight_dict[weight_path] = data
        elif self.model.is_exist(path):
            data = self.model.read(path)
            weight_dict[path] = data
        else:
            raise RuntimeError(f"Can't find key: {path} or {weight_path}")

        bias_path = path + ".bias"
        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)

    def is_block_float_fallback(self, block_idx: int) -> bool:
        return self._blocks_full_float_fallback.get(block_idx, False)

    def is_lmhead_float_fallback(self) -> bool:
        return self._lmhead_float_fallback

    def check_lmhead_quant_consistency(self, conv):
        if not conv.tie_word_embeddings:
            lmhead = conv.model_info.weights[LlmList.LMHEAD]
            lmhead_key = lmhead + ".weight"
            lmhead_info = self.model.get_tensor_info(lmhead_key)
            if lmhead_info and lmhead_info.get('is_quantized', False):
                lmhead_qt = lmhead_info.get('quant_type')
                lmhead_gs = get_quant_type_group_size(lmhead_qt) if lmhead_qt else 32
                if lmhead_gs != conv.q_group_size:
                    logger.warning(
                        "lm_head uses %s (group_size=%d) but model uses group_size=%d. lm_head falls back to float32.",
                        lmhead_qt, lmhead_gs, conv.q_group_size)
                    self._lmhead_float_fallback = True
                else:
                    self._lmhead_float_fallback = False
            else:
                self._lmhead_float_fallback = False
        else:
            self._lmhead_float_fallback = False

    def save_quantized_embedding(self, conv):
        embedding = conv.model_info.weights[LlmList.EMBEDING]
        embedding_key = embedding + ".weight"
        embedding_info = self.model.get_tensor_info(embedding_key)
        if embedding_info and embedding_info.get('is_quantized', False):
            logger.info("Embedding tensor %s is quantized in GGUF. Preserving quantization.",
                        embedding_key)
            converted = self.quant_converter.convert_to_llmconv_format(self.model,
                                                                       embedding_key,
                                                                       transpose=False)
            if converted['is_quantized']:
                quant_weights = {
                    embedding + ".qweight": converted['qweight'],
                    embedding + ".scales": converted['scales'],
                }
                if 'qzeros' in converted:
                    quant_weights[embedding + ".qzeros"] = converted['qzeros']
                quant_npz = "embedding_quant_weights.npz"
                np.savez(quant_npz, **quant_weights)
                logger.info("Saved quantized embedding arrays to %s", quant_npz)

    def save_quantized_lmhead(self, conv):
        lmhead = conv.model_info.weights[LlmList.LMHEAD]
        lmhead_key = lmhead + ".weight"
        if not conv.tie_word_embeddings and not self._lmhead_float_fallback:
            lmhead_info = self.model.get_tensor_info(lmhead_key)
            if lmhead_info and lmhead_info.get('is_quantized', False):
                logger.info("LM head tensor %s is quantized in GGUF. Preserving quantization.",
                            lmhead_key)
                converted = self.quant_converter.convert_to_llmconv_format(self.model,
                                                                           lmhead_key,
                                                                           transpose=True)
                if converted['is_quantized']:
                    quant_weights = {
                        lmhead + ".qweight": converted['qweight'],
                        lmhead + ".scales": converted['scales'],
                    }
                    if 'qzeros' in converted:
                        quant_weights[lmhead + ".qzeros"] = converted['qzeros']
                    quant_npz = "lm_head_quant_weights.npz"
                    np.savez(quant_npz, **quant_weights)
                    logger.info("Saved quantized lm_head arrays to %s", quant_npz)
                    # quant_mode/quant_bits/q_group_size already set by init_quantization

    def compile_block_args(self, conv, layer_id, is_cache=False):
        full_fallback = self._blocks_full_float_fallback.get(layer_id, False)
        quantize_param = conv.half_precision_quantize if full_fallback else conv.quantize
        extra_args = []
        if not full_fallback:
            extra_args.append(f'--q_group_size {conv.q_group_size}')
        if not full_fallback and conv.symmetric:
            extra_args.append('--q_symmetric')
        return quantize_param, extra_args
