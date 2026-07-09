# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import os
import re

from .LlmInfo import WeightType
from transform.MLIRImporter import Platform
from .LlmInfo import LlmType
from .transformers_compat import Config, build_rope_parameters
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
        full_fallback = self._blocks_full_float_fallback.get(layer_id, False)
        if full_fallback:
            quantize_param = conv.half_precision_quantize
            extra_args = []
        else:
            info = self._block_quant_info.get(layer_id)
            if info and info.quant_bits != 16:
                quantize_param = f"w{info.quant_bits}{conv.half_precision_quantize}"
                extra_args = [f'--q_group_size {info.q_group_size}']
            else:
                quantize_param = conv.quantize
                extra_args = [f'--q_group_size {conv.q_group_size}']
        return quantize_param, extra_args


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
            if conv.quant_mode not in ["gptq", "awq", "compressed-tensors", "auto-round", "fp8"]:
                raise NotImplementedError(f"Not support quantization method: {conv.quant_mode}")
            if conv.quant_mode != "compressed-tensors" and conv.quant_mode != "fp8":
                conv.q_group_size = conv.quantization_config["group_size"]
                conv.quant_bits = conv.quantization_config["bits"]
            if conv.quant_mode == "fp8":
                conv.activation_scheme = conv.quantization_config["activation_scheme"]
                conv.fmt = conv.quantization_config["fmt"]
                conv.block_size = conv.quantization_config["weight_block_size"]
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
            elif conv.quant_mode == "fp8":
                real_quantize = conv.get_qtype(dtype, 16)
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

    def fp32_to_fp8(self, x: np.ndarray, fmt: str = "e4m3") -> np.ndarray:
        """
        convert fp32 to fp8
        """
        x = np.asarray(x, dtype=np.float32)

        if fmt.lower() == "e4m3":
            ebits, mbits, bias = 4, 3, 7
            has_inf = False
            max_finite_exp_field = (1 << ebits) - 1  # 15
            max_finite_mant_field = (1 << mbits) - 2  # 6
        elif fmt.lower() == "e5m2":
            ebits, mbits, bias = 5, 2, 15
            has_inf = True
            max_finite_exp_field = ((1 << ebits) - 1) - 1  # 30
            max_finite_mant_field = (1 << mbits) - 1  # 3
        else:
            raise ValueError("fmt only support 'e4m3' and 'e5m2'")

        sign_bit = 1 << (ebits + mbits)
        exp_mask_all_ones = (1 << ebits) - 1
        mant_mask = (1 << mbits) - 1

        out = np.zeros_like(x, dtype=np.uint8)

        sign = np.signbit(x).astype(np.uint8)
        out |= sign << (ebits + mbits)

        ax = np.abs(x)

        is_nan = np.isnan(ax)
        is_inf = np.isinf(ax)
        is_zero = (ax == 0)

        out[is_nan] = ((sign[is_nan] <<
                        (ebits + mbits)) | (exp_mask_all_ones << mbits) | 1).astype(np.uint8)

        if np.any(is_inf):
            if has_inf:
                out[is_inf] = ((sign[is_inf] <<
                                (ebits + mbits)) | (exp_mask_all_ones << mbits)).astype(np.uint8)
            else:
                out[is_inf] = ((sign[is_inf] << (ebits + mbits)) | (max_finite_exp_field << mbits)
                               | max_finite_mant_field).astype(np.uint8)

        normal_mask = ~(is_nan | is_inf | is_zero)
        if not np.any(normal_mask):
            return out

        vals = ax[normal_mask]

        m, e = np.frexp(vals)
        m2 = m * 2.0
        e2 = e - 1

        exp_field = e2 + bias
        overflow = exp_field > max_finite_exp_field

        is_normal_num = (exp_field >= 1) & (~overflow)
        is_subnormal = exp_field <= 0

        enc = np.zeros_like(vals, dtype=np.uint8)

        if np.any(is_normal_num):
            mn = m2[is_normal_num] - 1.0  # [0,1)
            frac = np.round(mn * (1 << mbits)).astype(np.int32)

            expn = exp_field[is_normal_num].astype(np.int32)

            carry = frac == (1 << mbits)
            if np.any(carry):
                frac[carry] = 0
                expn[carry] += 1

            ov2 = expn > max_finite_exp_field
            if np.any(ov2):
                idx_sat = np.where(is_normal_num)[0][ov2]
                if has_inf:
                    enc[idx_sat] = np.uint8(exp_mask_all_ones << mbits)
                else:
                    enc[idx_sat] = np.uint8((max_finite_exp_field << mbits) | max_finite_mant_field)
                keep = ~ov2
                if np.any(keep):
                    idx_keep = np.where(is_normal_num)[0][keep]
                    enc[idx_keep] = ((expn[keep] << mbits) | (frac[keep] & mant_mask)).astype(
                        np.uint8)
            else:
                idx = np.where(is_normal_num)[0]
                enc[idx] = ((expn << mbits) | (frac & mant_mask)).astype(np.uint8)

        if np.any(is_subnormal):
            vs = vals[is_subnormal]
            scale = np.float32(2.0**(1 - bias - mbits))
            mant = np.round(vs / scale).astype(np.int32)
            mant = np.clip(mant, 0, mant_mask)
            idx = np.where(is_subnormal)[0]
            enc[idx] = mant.astype(np.uint8)

        if np.any(overflow):
            idx = np.where(overflow)[0]
            if has_inf:
                enc[idx] = np.uint8(exp_mask_all_ones << mbits)
            else:
                enc[idx] = np.uint8((max_finite_exp_field << mbits) | max_finite_mant_field)

        sign_local = sign[normal_mask]
        enc |= (sign_local << (ebits + mbits)).astype(np.uint8)

        out[normal_mask] = enc
        return out

    # Vectorized: replaced row/col-wise Python loops with numpy stride-based
    # broadcasts and pack operations (e.g. unpacked[0::2] | (unpacked[1::2] << 4)),
    # reducing unpack_qweight from K*compress_ratio iterations to compress_ratio
    # numpy ops and unpack_qzeros similarly. Also fixed a latent bug where
    # pack_int8_zeros was unassigned when bits=8 and need_int8_zeros=True.
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
            order_map = [i for i in range(compress_ratio)]
            unpacked_weights = np.zeros((K * compress_ratio, N), dtype=dtype)
            for p in range(compress_ratio):
                shift = bits * order_map[p]
                unpacked_weights[p::compress_ratio, :] = (qweight >> shift) & mask
            if bits == 4:
                pack_int8_weights = (unpacked_weights[0::2, :]
                                     | (unpacked_weights[1::2, :] << 4)).astype(np.uint8)
            else:
                pack_int8_weights = unpacked_weights.astype("uint8")
        elif quant_mode == "awq":
            order_map = [0, 4, 1, 5, 2, 6, 3, 7]
            unpacked_weights = np.zeros((K, N * compress_ratio), dtype=dtype)
            for p in range(compress_ratio):
                shift = bits * order_map[p]
                unpacked_weights[:, p::compress_ratio] = (qweight >> shift) & mask
            if bits == 4:
                pack_int8_weights = (unpacked_weights[0::2, :]
                                     | (unpacked_weights[1::2, :] << 4)).astype(np.uint8)
            else:
                pack_int8_weights = unpacked_weights.astype("uint8")
        else:
            raise NotImplementedError(f"Not support now: {quant_mode}")

        for p in range(compress_ratio):
            shift = bits * order_map[p]
            unpacked_zeros[:, p::compress_ratio] = (qzeros >> shift) & mask

        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")

        if need_int8_zeros:
            if quant_mode == "gptq":
                unpacked_zeros += 1
            if bits == 4:
                pack_int8_zeros = (unpacked_zeros[0::2, :]
                                   | (unpacked_zeros[1::2, :] << 4)).astype(np.uint8)
            else:
                pack_int8_zeros = unpacked_zeros.astype("uint8")
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
        if not is_quant and conv.quant_mode != "fp8":
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
        elif conv.quant_mode == "fp8":
            weight_path = path + ".weight"
            scale_path = path + ".weight_scale_inv"
            weight_data = self.model.read(weight_path)
            weight_data = self.fp32_to_fp8(weight_data, conv.fmt)
            scale_data = self.model.read(scale_path)
            weight_dict[weight_path] = weight_data
            weight_dict[scale_path] = scale_data
            K = weight_data.shape[1]
            N = weight_data.shape[0]

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
from .gguf_compat import GGMLQuantizationType


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


def create_gguf_config(gguf_reader,
                       quantize='w8bf16',
                       seq_length=4096,
                       group_size=None,
                       mmproj_reader=None):
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
    if vocab is None:
        tokens = get_val("tokenizer.ggml.tokens")
        vocab_size = len(tokens) if tokens else 151936
    else:
        vocab_size = vocab

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

    _has_output_weight = any(t.name == "output.weight" for t in gguf_reader.tensors)

    quantization_config = None
    if quantized_tensors:
        bits = 8
        quant_method = "gptq"

        detected_bits = 8  # default
        has_q4 = any(qt in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
                            GGMLQuantizationType.Q4_K) for qt in quant_type_counts)
        has_higher = any(qt in (GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1,
                                GGMLQuantizationType.Q8_K, GGMLQuantizationType.Q6_K,
                                GGMLQuantizationType.Q5_K) for qt in quant_type_counts)
        for qt, count in quant_type_counts.items():
            if qt in [
                    GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q4_K
            ]:
                detected_bits = 4
                break
            elif qt in [
                    GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1, GGMLQuantizationType.Q8_K,
                    GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q5_K
            ]:
                detected_bits = 8

        if 'w4' in quantize and detected_bits != 4:
            if not has_q4:
                logger.error(
                    "quantize=%s requires 4-bit weights, but GGUF model has %d-bit quantization (types: %s). Please use -q w8f16 or w8bf16 instead.",
                    quantize, detected_bits, quant_type_counts)
                raise RuntimeError(
                    f"quantize={quantize} mismatch with GGUF {detected_bits}-bit weights")
            logger.warning(
                "Mixed quantization detected (types: %s). Using fallback for non-4bit blocks.",
                quant_type_counts)
        elif 'w8' in quantize and detected_bits != 8:
            if not has_higher:
                logger.error(
                    "quantize=%s requires 8-bit weights, but GGUF model has %d-bit quantization (types: %s). Please use -q w4f16 or w4bf16 instead.",
                    quantize, detected_bits, quant_type_counts)
                raise RuntimeError(
                    f"quantize={quantize} mismatch with GGUF {detected_bits}-bit weights")
            logger.warning(
                "Mixed quantization detected (types: %s). Using fallback for non-8bit blocks.",
                quant_type_counts)
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

    # Build the text config as an attribute-access dict (Config). This replaces the
    # former per-arch transformers config classes (Qwen3Config/Qwen2Config/
    # LlamaConfig/GemmaConfig/Qwen3_5Config); downstream code only reads attributes,
    # uses .get() / `in`, and checks isinstance(..., dict), so Config is a drop-in.
    extra_kwargs = {}
    if arch_lower == 'qwen35':
        full_attention_interval = get_val(f"{architecture}.full_attention_interval") or 4
        text_layer_types = [
            "full_attention" if
            (i % full_attention_interval == (full_attention_interval - 1)) else "linear_attention"
            for i in range(num_hidden_layers)
        ]
        ssm_conv_kernel = get_val(f"{architecture}.ssm.conv_kernel") or 4
        ssm_state_size = get_val(f"{architecture}.ssm.state_size") or 128
        ssm_group_count = get_val(f"{architecture}.ssm.group_count") or 16
        extra_kwargs["full_attention_interval"] = full_attention_interval
        extra_kwargs["linear_conv_kernel_dim"] = ssm_conv_kernel
        extra_kwargs["linear_key_head_dim"] = ssm_state_size
        extra_kwargs["linear_value_head_dim"] = ssm_state_size
        extra_kwargs["linear_num_key_heads"] = ssm_group_count
        extra_kwargs["linear_num_value_heads"] = ssm_group_count
        extra_kwargs["layer_types"] = text_layer_types
        extra_kwargs["text_config"] = Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=seq_length,
            rms_norm_eps=rms_norm_eps,
            hidden_act="silu",
            layer_types=text_layer_types,
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=ssm_conv_kernel,
            linear_key_head_dim=ssm_state_size,
            linear_value_head_dim=ssm_state_size,
            linear_num_key_heads=ssm_group_count,
            linear_num_value_heads=ssm_group_count,
            attn_output_gate=True,
            mamba_ssm_dtype="float32",
            model_type="qwen3_5_text",
            tie_word_embeddings=not _has_output_weight,
            partial_rotary_factor=0.25,
        )

    config = Config(
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
        dtype=dtype_str,
        tie_word_embeddings=(not _has_output_weight if arch_lower == 'qwen35' else True),
        **extra_kwargs,
    )
    # Mirror transformers' standardize_rope_params so config.rope_parameters exists;
    # the mrope post-processing below relies on it being present and updatable.
    config.rope_parameters = Config(build_rope_parameters(config))
    if arch_lower == 'qwen35':
        config.model_type = "qwen3_5"
        if 'text_config' in config:
            config.text_config.rope_parameters = Config(build_rope_parameters(config.text_config))
    else:
        config.model_type = architecture
    if quantization_config:
        config.quantization_config = quantization_config

    dimension_sections = get_val(f"{architecture}.rope.dimension_sections")
    if dimension_sections is not None:
        mrope_section = dimension_sections[:3] if len(
            dimension_sections) >= 3 else dimension_sections
        section_list = list(mrope_section)
        config.rope_scaling = _dict_to_config({"mrope_section": section_list, "type": "default"})
        config.rope_parameters.update({
            "rope_theta": rope_theta,
            "mrope_section": section_list,
            "mrope_interleaved": True,
            "rope_type": "default",
        })
        if arch_lower == 'qwen35':
            config.rope_parameters["partial_rotary_factor"] = 0.25
            if hasattr(config, 'text_config'):
                config.text_config.rope_parameters = _dict_to_config(config.rope_parameters)

    if mmproj_reader:
        is_internvl = False
        if architecture == "qwen2":
            tags = get_val("general.tags") or []
            if isinstance(tags, list) and "internvl" in tags:
                is_internvl = True
                config.model_type = "internvl_chat"
        _attach_vision_config(config, mmproj_reader, hidden_size)
        if is_internvl:
            config.model_type = "qwen2"
            config.tie_word_embeddings = not _has_output_weight
            llm_config = config
            vision_config = getattr(config, 'vision_config', None)
            internvl_config = _dict_to_config({
                "model_type":
                "internvl_chat",
                "llm_config":
                llm_config,
                "vision_config":
                vision_config,
                "downsample_ratio":
                0.5,
                "quantization_config":
                getattr(config, 'quantization_config', None),
            })
            return internvl_config

    return config


def _attach_vision_config(config, mmproj_reader, llm_hidden_size):
    """Attach a vision_config to the model config object from mmproj GGUF metadata.

    This is needed for VLM GGUF conversion where the vision encoder config
    is stored in the mmproj GGUF file (arch=clip) rather than in the LLM GGUF.
    """

    def get_clip_val(key):
        field = mmproj_reader.get_field(key)
        if field is None:
            return None
        try:
            return field.contents()
        except Exception:
            return None

    clip_vision_hidden_size = get_clip_val("clip.vision.embedding_length")
    clip_vision_num_heads = get_clip_val("clip.vision.attention.head_count")
    clip_vision_depth = get_clip_val("clip.vision.block_count")
    clip_vision_intermediate_size = get_clip_val("clip.vision.feed_forward_length")
    clip_vision_patch_size = get_clip_val("clip.vision.patch_size")
    clip_vision_image_size = get_clip_val("clip.vision.image_size")
    clip_vision_spatial_merge_size = get_clip_val("clip.vision.spatial_merge_size")
    clip_vision_projection_dim = get_clip_val("clip.vision.projection_dim")
    clip_vision_layer_norm_eps = get_clip_val("clip.vision.attention.layer_norm_epsilon")
    clip_use_gelu = get_clip_val("clip.use_gelu")
    clip_is_deepstack = get_clip_val("clip.vision.is_deepstack_layers")
    clip_projector_type = get_clip_val("clip.projector_type")

    if clip_vision_hidden_size is None:
        logger.warning("No clip.vision fields found in mmproj GGUF; skipping vision_config")
        return

    deepstack_visual_indexes = []
    if clip_is_deepstack:
        try:
            deepstack_visual_indexes = [i for i, v in enumerate(clip_is_deepstack) if v]
        except Exception:
            logger.warning("Could not parse clip.vision.is_deepstack_layers")

    image_size = clip_vision_image_size if clip_vision_image_size else 768
    patch_size = clip_vision_patch_size if clip_vision_patch_size else 16
    num_position_embeddings = (image_size // patch_size)**2

    arch_field = mmproj_reader.get_field("general.architecture")
    mmproj_arch = arch_field.contents() if arch_field else "clip"
    llm_arch = getattr(config, 'model_type', '')

    vision_config = None
    if llm_arch in ('qwen3vl', 'qwen3_vl'):
        vision_config = Config(
            depth=clip_vision_depth if clip_vision_depth else 24,
            hidden_size=clip_vision_hidden_size if clip_vision_hidden_size else 1024,
            hidden_act="gelu_pytorch_tanh" if clip_use_gelu else "silu",
            intermediate_size=clip_vision_intermediate_size
            if clip_vision_intermediate_size else 4096,
            num_heads=clip_vision_num_heads if clip_vision_num_heads else 16,
            in_channels=3,
            patch_size=patch_size,
            spatial_merge_size=clip_vision_spatial_merge_size
            if clip_vision_spatial_merge_size else 2,
            temporal_patch_size=2,
            out_hidden_size=llm_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
        )
    elif llm_arch in ('qwen2_5vl', 'qwen2_5_vl'):
        vision_config = Config(
            depth=clip_vision_depth if clip_vision_depth else 24,
            hidden_size=clip_vision_hidden_size if clip_vision_hidden_size else 1024,
            hidden_act="gelu_pytorch_tanh" if clip_use_gelu else "silu",
            intermediate_size=clip_vision_intermediate_size
            if clip_vision_intermediate_size else 4096,
            num_heads=clip_vision_num_heads if clip_vision_num_heads else 16,
            in_channels=3,
            patch_size=patch_size,
            spatial_merge_size=clip_vision_spatial_merge_size
            if clip_vision_spatial_merge_size else 2,
            temporal_patch_size=2,
            out_hidden_size=llm_hidden_size,
            fullatt_block_indexes=deepstack_visual_indexes,
        )
    elif llm_arch in ('qwen3_5', 'qwen35'):
        vision_config = Config(
            depth=clip_vision_depth if clip_vision_depth else 24,
            hidden_size=clip_vision_hidden_size if clip_vision_hidden_size else 1024,
            hidden_act="gelu_pytorch_tanh" if clip_use_gelu else "silu",
            intermediate_size=clip_vision_intermediate_size
            if clip_vision_intermediate_size else 4096,
            num_heads=clip_vision_num_heads if clip_vision_num_heads else 16,
            in_channels=3,
            patch_size=patch_size,
            spatial_merge_size=clip_vision_spatial_merge_size
            if clip_vision_spatial_merge_size else 2,
            temporal_patch_size=2,
            out_hidden_size=llm_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
        )
    elif llm_arch in ('qwen2vl', 'qwen2_vl'):
        vision_config = Config(
            depth=clip_vision_depth if clip_vision_depth else 32,
            embed_dim=clip_vision_hidden_size if clip_vision_hidden_size else 1280,
            hidden_size=llm_hidden_size,
            hidden_act="quick_gelu" if clip_use_gelu else "silu",
            num_heads=clip_vision_num_heads if clip_vision_num_heads else 16,
            in_channels=3,
            patch_size=patch_size,
            spatial_merge_size=clip_vision_spatial_merge_size
            if clip_vision_spatial_merge_size else 2,
            temporal_patch_size=2,
        )
    elif llm_arch in ('internvl_chat', ):
        internvl_hidden_size = clip_vision_hidden_size if clip_vision_hidden_size else 1024
        internvl_num_heads = clip_vision_num_heads if clip_vision_num_heads else 16
        internvl_head_dim = internvl_hidden_size // internvl_num_heads
        vision_config = _dict_to_config({
            "hidden_size":
            internvl_hidden_size,
            "intermediate_size":
            clip_vision_intermediate_size if clip_vision_intermediate_size else 4096,
            "num_attention_heads":
            internvl_num_heads,
            "num_hidden_layers":
            clip_vision_depth if clip_vision_depth else 24,
            "num_channels":
            3,
            "patch_size":
            patch_size,
            "image_size":
            image_size,
            "num_position_embeddings":
            num_position_embeddings,
            "layer_norm_eps":
            clip_vision_layer_norm_eps if clip_vision_layer_norm_eps else 1e-6,
            "hidden_act":
            "gelu" if clip_use_gelu else "silu",
            "qkv_bias":
            True,
            "qk_normalization":
            False,
            "use_flash_attn":
            True,
            "head_dim":
            internvl_head_dim,
            "model_type":
            "intern_vit_6b",
            "initializer_range":
            1e-10,
            "initializer_factor":
            0.1,
            "norm_type":
            "layer_norm",
        })
    else:
        logger.warning("No built-in vision config for arch '%s'; "
                       "falling back to generic Config", llm_arch)
        vision_config = _dict_to_config({
            "hidden_size":
            clip_vision_hidden_size if clip_vision_hidden_size else 1024,
            "num_heads":
            clip_vision_num_heads if clip_vision_num_heads else 16,
            "depth":
            clip_vision_depth if clip_vision_depth else 24,
            "intermediate_size":
            clip_vision_intermediate_size if clip_vision_intermediate_size else 4096,
            "patch_size":
            patch_size,
            "image_size":
            image_size,
            "spatial_merge_size":
            clip_vision_spatial_merge_size if clip_vision_spatial_merge_size else 2,
            "in_channels":
            3,
            "temporal_patch_size":
            2,
            "num_position_embeddings":
            num_position_embeddings,
            "layer_norm_eps":
            clip_vision_layer_norm_eps if clip_vision_layer_norm_eps else 1e-6,
            "hidden_act":
            "gelu_pytorch_tanh" if clip_use_gelu else "silu",
            "projector_type":
            clip_projector_type if clip_projector_type else "qwen3vl_merger",
            "deepstack_visual_indexes":
            deepstack_visual_indexes,
        })

    config.vision_config = vision_config
    logger.info(
        "Attached vision_config from mmproj GGUF: hidden_size=%d, depth=%d, "
        "deepstack_visual_indexes=%s", clip_vision_hidden_size, clip_vision_depth,
        deepstack_visual_indexes)


PREPROCESSOR_SIZE_TABLE = {
    'qwen3vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 25165824,
            'shortest_edge': 4096
        },
        'processor_class': 'Qwen3VLProcessor',
        'image_processor_type': 'Qwen2VLImageProcessorFast',
        'video_processor_type': 'Qwen3VLVideoProcessor',
    },
    'qwen3_vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 25165824,
            'shortest_edge': 4096
        },
        'processor_class': 'Qwen3VLProcessor',
        'image_processor_type': 'Qwen2VLImageProcessorFast',
        'video_processor_type': 'Qwen3VLVideoProcessor',
    },
    'qwen3_5': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 25165824,
            'shortest_edge': 4096
        },
        'processor_class': 'Qwen3VLProcessor',
        'image_processor_type': 'Qwen2VLImageProcessorFast',
        'video_processor_type': 'Qwen3VLVideoProcessor',
    },
    'qwen2_5vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 25165824,
            'shortest_edge': 4096
        },
        'processor_class': 'Qwen2_5_VLProcessor',
        'image_processor_type': 'Qwen2_5_VLImageProcessorFast',
        'video_processor_type': 'Qwen2_5_VLVideoProcessor',
    },
    'qwen2_5_vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 25165824,
            'shortest_edge': 4096
        },
        'processor_class': 'Qwen2_5_VLProcessor',
        'image_processor_type': 'Qwen2_5_VLImageProcessorFast',
        'video_processor_type': 'Qwen2_5_VLVideoProcessor',
    },
    'qwen2vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'processor_class': 'Qwen2VLProcessor',
        'image_processor_type': 'Qwen2VLImageProcessorFast',
        'video_processor_type': 'Qwen2VLVideoProcessor',
    },
    'qwen2_vl': {
        'preprocessor': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'video': {
            'longest_edge': 16777216,
            'shortest_edge': 65536
        },
        'processor_class': 'Qwen2VLProcessor',
        'image_processor_type': 'Qwen2VLImageProcessorFast',
        'video_processor_type': 'Qwen2VLVideoProcessor',
    },
    'internvl_chat': {
        'processor_class': 'InternVLProcessor',
        'image_processor_type': 'CLIPFeatureExtractor',
        'clip_preprocessor': True,
    },
}


def _generate_preprocessor_configs(llm_arch, mmproj_reader, config_dir):
    """Generate preprocessor_config.json and video_preprocessor_config.json
    from mmproj GGUF metadata for VLM models.

    These configs are required by AutoProcessor.from_pretrained() in the
    BModel inference pipeline for correct image/video preprocessing.
    """
    if mmproj_reader is None:
        return

    arch_info = PREPROCESSOR_SIZE_TABLE.get(llm_arch)
    if arch_info is None:
        logger.warning(
            "No preprocessor size table entry for arch '%s'; "
            "skipping preprocessor/video_preprocessor config generation", llm_arch)
        return

    def get_clip_val(key):
        field = mmproj_reader.get_field(key)
        if field is None:
            return None
        try:
            return field.contents()
        except Exception:
            return None

    clip_vision_patch_size = get_clip_val("clip.vision.patch_size")
    clip_vision_spatial_merge_size = get_clip_val("clip.vision.spatial_merge_size")
    clip_vision_image_mean = get_clip_val("clip.vision.image_mean")
    clip_vision_image_std = get_clip_val("clip.vision.image_std")

    patch_size = clip_vision_patch_size if clip_vision_patch_size else 16
    merge_size = clip_vision_spatial_merge_size if clip_vision_spatial_merge_size else 2
    image_mean = list(clip_vision_image_mean) if clip_vision_image_mean else [0.5, 0.5, 0.5]
    image_std = list(clip_vision_image_std) if clip_vision_image_std else [0.5, 0.5, 0.5]

    import json

    if arch_info.get('clip_preprocessor'):
        image_size = get_clip_val("clip.vision.image_size") or 448
        preprocessor_dict = {
            "crop_size": image_size,
            "do_center_crop": True,
            "do_normalize": True,
            "do_resize": True,
            "feature_extractor_type": arch_info.get('image_processor_type', 'CLIPFeatureExtractor'),
            "image_mean": image_mean,
            "image_std": image_std,
            "resample": 3,
            "size": image_size,
        }
        preprocessor_path = os.path.join(config_dir, 'preprocessor_config.json')
        with open(preprocessor_path, 'w') as f:
            json.dump(preprocessor_dict, f, indent=2)
        logger.info("Saved preprocessor_config to %s", preprocessor_path)
        return

    preprocessor_dict = {
        "size": arch_info['preprocessor'],
        "patch_size": patch_size,
        "temporal_patch_size": 2,
        "merge_size": merge_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "processor_class": arch_info['processor_class'],
        "image_processor_type": arch_info['image_processor_type'],
    }
    preprocessor_path = os.path.join(config_dir, 'preprocessor_config.json')
    with open(preprocessor_path, 'w') as f:
        json.dump(preprocessor_dict, f, indent=2)
    logger.info("Saved preprocessor_config to %s", preprocessor_path)

    video_preprocessor_dict = {
        "size": arch_info['video'],
        "patch_size": patch_size,
        "temporal_patch_size": 2,
        "merge_size": merge_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "processor_class": arch_info['processor_class'],
        "video_processor_type": arch_info['video_processor_type'],
    }
    video_preprocessor_path = os.path.join(config_dir, 'video_preprocessor_config.json')
    with open(video_preprocessor_path, 'w') as f:
        json.dump(video_preprocessor_dict, f, indent=2)
    logger.info("Saved video_preprocessor_config to %s", video_preprocessor_path)


def _dict_to_config(d):
    """Convert a dict to a dict-like object that supports both .get() and attribute access.

    Uses :class:`transformers_compat.Config`, which recursively wraps nested dicts so
    that ``cfg.vision_config.patch_size`` works at any depth.
    """
    return Config(d)


@dataclass
class LayerQuantInfo:
    quant_bits: int = 16
    symmetric: bool = True
    q_group_size: int = 0
    mixed: bool = False
    fallback_action: str = "none"


class GGUFModelHandle(ModelHandle):

    ARCH_TO_MODEL_TYPE = {
        "qwen3vl": "qwen3_vl",
        "qwen2vl": "qwen2_vl",
        "qwen2_5vl": "qwen2_5_vl",
        "mllama": "mllama",
        "llama": "llama",
        "llama3": "llama",
        "qwen3": "qwen3",
        "qwen35": "qwen3_5",
        "qwen2": "qwen2",
        "qwen2_moe": "qwen2_moe",
        "gemma": "gemma",
        "gemma2": "gemma2",
        "chatglm": "chatglm",
    }

    VLM_ARCHS = {"qwen3vl", "qwen2vl", "qwen2_5vl", "mllama", "qwen35", "internvl_chat"}

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
        self._block_quant_info = {}
        self._mixed_quant_fallback = False

    def load_mmproj(self, mmproj_path: str):
        """Load mmproj GGUF file for vision models.

        Merges vision encoder and projector tensors from the mmproj GGUF
        into the main tensor map so they are accessible via
        self.model.read() and self.model.is_exist().
        """
        self.model.load_mmproj(mmproj_path)

    def read(self, key: str) -> np.ndarray:
        return self.model.read(key)

    def is_exist(self, key: str) -> bool:
        return self.model.is_exist(key)

    def get_tensor_info(self, key: str):
        return self.model.get_tensor_info(key)

    def read_quantized(self, key: str):
        return self.model.read_quantized(key)

    def init_quantization(self, conv):
        from .gguf_compat import GGMLQuantizationType

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
            self._resolve_layer_quantization(conv)
            if self.quantized_tensors:
                first_tensor = next(iter(self.quantized_tensors.values()))
                quant_type = first_tensor['quant_type']
                group_size = get_quant_type_group_size(quant_type)
            else:
                group_size = get_gguf_group_size(self.model.reader)
            conv.q_group_size = group_size
            self.quant_converter.group_size = group_size
            info0 = self.layer_quant_infos.get("block.0")
            conv.quant_bits = info0.quant_bits if info0 else 8
            conv.symmetric = info0.symmetric if info0 else True
            conv.half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"
            logger.info("Auto-detected GGUF group size: %d", conv.q_group_size)
            if self._mixed_quant_fallback:
                logger.warning(
                    "  Note: Blocks with mismatched group_size will have ALL linears fallback to float"
                )
            self._detect_block_float_fallbacks(conv)
        else:
            conv.half_precision_quantize = "bf16" if "bf16" in conv.quantize else "f16"

    @staticmethod
    def _is_quant_type_symmetric(quant_type) -> bool:
        from .gguf_compat import GGMLQuantizationType
        qn = quant_type.name if hasattr(quant_type, 'name') else str(quant_type)
        if qn in ('Q4_0', 'Q5_0', 'Q8_0'):
            return True
        if qn.startswith('Q') and '_K' in qn:
            return True
        return False

    @staticmethod
    def _quant_type_to_output_bits(quant_type) -> int:
        from .gguf_compat import GGMLQuantizationType
        if quant_type in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1):
            return 4
        if quant_type in (GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1,
                          GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1,
                          GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                          GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K):
            return 8
        return 8

    def _resolve_layer_quantization(self, conv):
        from .gguf_compat import GGMLQuantizationType
        from collections import defaultdict

        FLOAT_TYPES = {
            GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16
        }

        layer_prefixes = defaultdict(list)
        for tensor in self.model.reader.tensors:
            name = tensor.name
            qt = tensor.tensor_type
            if qt in FLOAT_TYPES:
                continue
            if name.startswith("model.visual."):
                key = "vit"
            elif name.startswith("model.layers."):
                m = re.match(r'model\.layers\.(\d+)', name)
                key = f"block.{m.group(1)}" if m else name
            elif name.startswith("model.token_embd"):
                key = "embedding"
            elif name.startswith("model.output"):
                key = "lm_head"
            else:
                continue
            gs = get_quant_type_group_size(qt)
            layer_prefixes[key].append((qt, gs))

        self.layer_quant_infos = {}
        self._tensor_widen_targets = {}

        for key, tensor_entries in layer_prefixes.items():
            qt_set = {e[0] for e in tensor_entries}
            gs_set = {e[1] for e in tensor_entries}
            all_sym = all(self._is_quant_type_symmetric(qt) for qt in qt_set)

            if len(qt_set) == 1:
                qt = next(iter(qt_set))
                info = LayerQuantInfo(
                    quant_bits=self._quant_type_to_output_bits(qt),
                    symmetric=all_sym,
                    q_group_size=next(iter(gs_set)),
                    mixed=False,
                    fallback_action="none",
                )
                self.layer_quant_infos[key] = info
                continue

            if len(gs_set) > 1:
                info = LayerQuantInfo(
                    quant_bits=16,
                    symmetric=True,
                    q_group_size=max(gs_set),
                    mixed=True,
                    fallback_action="float",
                )
                self.layer_quant_infos[key] = info
                continue

            common_gs = next(iter(gs_set))
            bit_set = {self._quant_type_to_output_bits(qt) for qt in qt_set}

            if 4 in bit_set and 8 not in bit_set:
                info = LayerQuantInfo(
                    quant_bits=4,
                    symmetric=True,
                    q_group_size=common_gs,
                    mixed=True,
                    fallback_action="widen_to_q4_1",
                )
                self.layer_quant_infos[key] = info
            else:
                info = LayerQuantInfo(
                    quant_bits=8,
                    symmetric=True,
                    q_group_size=common_gs,
                    mixed=True,
                    fallback_action="widen_to_q8_0",
                )
                self.layer_quant_infos[key] = info

        if self.layer_quant_infos:
            logger.info("Layer quantization resolved: %d layers detected",
                        len(self.layer_quant_infos))
            for key, info in self.layer_quant_infos.items():
                logger.info("  %s: bits=%d sym=%s gs=%d action=%s", key, info.quant_bits,
                            info.symmetric, info.q_group_size, info.fallback_action)

    def _detect_block_float_fallbacks(self, conv):
        num_layers = getattr(conv.llm_config, 'num_hidden_layers', 0)
        model_q_group_size = conv.q_group_size
        from .gguf_compat import GGMLQuantizationType

        for idx in range(num_layers):
            layer_info = self.layer_quant_infos.get(f"block.{idx}")
            linear_paths = self._get_linear_paths_for_block(conv, idx)
            has_mismatch = False

            if layer_info is None or layer_info.fallback_action == "none":
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
                        "  Block %d: group_size mismatch. ALL linears (%s) fallback to float.", idx,
                        ', '.join(linear_names))
                else:
                    self._blocks_full_float_fallback[idx] = False
                if layer_info:
                    self._block_quant_info[idx] = layer_info
            elif layer_info.fallback_action == "float":
                self._blocks_full_float_fallback[idx] = True
                linear_names = [p.split('.')[-1] for p in linear_paths]
                logger.warning(
                    "  Block %d: mixed quant cannot unify. ALL linears (%s) fallback to float.",
                    idx, ', '.join(linear_names))
            elif layer_info.fallback_action == "widen_to_q4_1":
                self._blocks_full_float_fallback[idx] = False
                self._block_quant_info[idx] = layer_info
                for path in linear_paths:
                    tensor_info = self.model.get_tensor_info(path)
                    if tensor_info and tensor_info.get(
                            'is_quantized',
                            False) and tensor_info.get('quant_type') == GGMLQuantizationType.Q4_0:
                        self._tensor_widen_targets[path] = "q4_1"
                logger.info("  Block %d: Q4_0->Q4_1 widening, compile with w4%s", idx,
                            conv.half_precision_quantize)
            elif layer_info.fallback_action == "widen_to_q8_0":
                self._blocks_full_float_fallback[idx] = False
                self._block_quant_info[idx] = layer_info
                for path in linear_paths:
                    tensor_info = self.model.get_tensor_info(path)
                    if tensor_info and tensor_info.get('is_quantized', False):
                        qt = tensor_info.get('quant_type')
                        if qt in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1):
                            self._tensor_widen_targets[path] = "q8_0"
                logger.info("  Block %d: Q4->Q8_0 widening, compile with w8%s", idx,
                            conv.half_precision_quantize)

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
        gguf_arch_for_keys = arch
        if arch == "qwen2":
            tags = get_gguf_val("general.tags") or []
            if isinstance(tags, list) and "internvl" in tags:
                arch = "internvl_chat"
        tokenizer_model = get_gguf_val("tokenizer.ggml.model") or "gpt2"
        tokenizer_pre = get_gguf_val("tokenizer.ggml.pre") or "default"
        if arch.startswith("qwen3"):
            tokenizer_pre = "qwen2"
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
                if is_special and tok.startswith("<|fim") or tok.startswith(
                        "<|repo") or tok.startswith("<|file"):
                    is_special = False
                added_tokens_entries.append({
                    "id": idx,
                    "content": tok,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
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

        model_type = type(self).ARCH_TO_MODEL_TYPE.get(arch, arch)

        arch_to_tokenizer_class = {
            "qwen3": "Qwen2Tokenizer",
            "qwen35": "Qwen2Tokenizer",
            "qwen2": "Qwen2Tokenizer",
            "llama": "LlamaTokenizer",
            "llama3": "LlamaTokenizer",
            "gemma": "GemmaTokenizer",
            "gemma2": "GemmaTokenizerFast",
            "internvl_chat": "Qwen2Tokenizer",
        }
        tokenizer_class = arch_to_tokenizer_class.get(arch)
        if tokenizer_class is None:
            tokenizer_class = "Qwen2Tokenizer" if arch.startswith("qwen") else "LlamaTokenizer"

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

        VLM_CONFIG_ARCHS = {
            "qwen3vl": {
                "model_type": "qwen3_vl",
                "architectures": "Qwen3VLForConditionalGeneration",
                "text_model_type": "qwen3_vl_text",
                "vision_model_type": "qwen3_vl",
                "vlm_tokens": {
                    "image_token_id": "<|image_pad|>",
                    "video_token_id": "<|video_pad|>",
                    "vision_start_token_id": "<|vision_start|>",
                    "vision_end_token_id": "<|vision_end|>",
                },
            },
            "qwen2_5vl": {
                "model_type": "qwen2_5_vl",
                "architectures": "Qwen2_5_VLForConditionalGeneration",
                "text_model_type": "qwen2_5_vl_text",
                "vision_model_type": "qwen2_5_vl",
                "vlm_tokens": {
                    "image_token_id": "<|image_pad|>",
                    "video_token_id": "<|video_pad|>",
                    "vision_start_token_id": "<|vision_start|>",
                    "vision_end_token_id": "<|vision_end|>",
                },
            },
            "qwen35": {
                "model_type": "qwen3_5",
                "architectures": "Qwen3_5ForConditionalGeneration",
                "text_model_type": "qwen3_5_text",
                "vision_model_type": "qwen3_5",
                "vlm_tokens": {
                    "image_token_id": "<|image_pad|>",
                    "video_token_id": "<|video_pad|>",
                    "vision_start_token_id": "<|vision_start|>",
                    "vision_end_token_id": "<|vision_end|>",
                },
            },
            "qwen2vl": {
                "model_type": "qwen2_vl",
                "architectures": "Qwen2VLForConditionalGeneration",
                "text_model_type": "qwen2_vl_text",
                "vision_model_type": "qwen2_vl",
                "vlm_tokens": {
                    "image_token_id": "<|image_pad|>",
                    "video_token_id": "<|video_pad|>",
                    "vision_start_token_id": "<|vision_start|>",
                    "vision_end_token_id": "<|vision_end|>",
                },
            },
            "internvl_chat": {
                "model_type": "internvl_chat",
                "architectures": "InternVLChatModel",
                "text_model_type": "qwen2",
                "vision_model_type": "intern_vit_6b",
                "vlm_tokens": {
                    "image_token_id": "<IMG_CONTEXT>",
                    "vision_start_token_id": "<|vision_start|>",
                    "vision_end_token_id": "<|vision_end|>",
                },
            },
        }

        vlm_info = VLM_CONFIG_ARCHS.get(arch)

        _base_gguf_config_keys = [
            (f"{gguf_arch_for_keys}.embedding_length", "hidden_size"),
            (f"{gguf_arch_for_keys}.block_count", "num_hidden_layers"),
            (f"{gguf_arch_for_keys}.attention.head_count", "num_attention_heads"),
            (f"{gguf_arch_for_keys}.attention.head_count_kv", "num_key_value_heads"),
            (f"{gguf_arch_for_keys}.vocab_size", "vocab_size"),
            (f"{gguf_arch_for_keys}.feed_forward_length", "intermediate_size"),
            (f"{gguf_arch_for_keys}.context_length", "max_position_embeddings"),
            (f"{gguf_arch_for_keys}.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
            (f"{gguf_arch_for_keys}.rope.freq_base", "rope_theta"),
            (f"{gguf_arch_for_keys}.attention.layer_norm_epsilon", "layer_norm_eps"),
        ]

        if vlm_info:
            config_dict = {
                "architectures": [vlm_info["architectures"]],
                "model_type": vlm_info["model_type"],
            }

            text_config_dict = {}

            gguf_config_keys = _base_gguf_config_keys
            for gguf_key, hf_key in gguf_config_keys:
                val = get_gguf_val(gguf_key)
                if val is not None:
                    text_config_dict[hf_key] = val

            if "hidden_size" in text_config_dict and "num_attention_heads" in text_config_dict:
                text_config_dict["head_dim"] = text_config_dict["hidden_size"] // text_config_dict[
                    "num_attention_heads"]

            for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
                if key in special_token_id_map:
                    text_config_dict[key] = special_token_id_map[key]

            text_config_dict["hidden_act"] = "silu"
            text_config_dict["initializer_range"] = 0.02
            _has_output = any(t.name == "output.weight" for t in self.model.reader.tensors)
            text_config_dict["tie_word_embeddings"] = not _has_output if arch in (
                "qwen35", "internvl_chat") else True
            text_config_dict["use_cache"] = True
            text_config_dict["attention_bias"] = False
            text_config_dict["attention_dropout"] = 0.0
            if arch == "qwen35":
                text_config_dict["attn_output_gate"] = True
            text_config_dict["model_type"] = vlm_info["text_model_type"]
            text_config_dict["dtype"] = "bfloat16"
            text_config_dict["vocab_size"] = len(tokens_list)

            dimension_sections = get_gguf_val(f"{arch}.rope.dimension_sections")
            if dimension_sections is not None:
                mrope_section = dimension_sections[:3] if len(
                    dimension_sections) >= 3 else dimension_sections
                mrope_section_list = list(mrope_section)
                if arch == "qwen35":
                    text_config_dict["rope_parameters"] = {
                        "rope_theta": text_config_dict.get("rope_theta", 10000000),
                        "mrope_interleaved": True,
                        "mrope_section": mrope_section_list,
                        "rope_type": "default",
                        "partial_rotary_factor": 0.25,
                    }
                else:
                    text_config_dict["rope_parameters"] = {
                        "mrope_interleaved": True,
                        "mrope_section": mrope_section_list,
                        "rope_type": "default",
                    }
                    text_config_dict["rope_scaling"] = text_config_dict["rope_parameters"]

            full_attention_interval = get_gguf_val(f"{arch}.full_attention_interval")
            if full_attention_interval is not None:
                num_layers = text_config_dict.get("num_hidden_layers", 28)
                text_config_dict["full_attention_interval"] = full_attention_interval
                text_config_dict["layer_types"] = [
                    "full_attention" if (i % full_attention_interval == (full_attention_interval -
                                                                         1)) else "linear_attention"
                    for i in range(num_layers)
                ]

            ssm_conv_kernel = get_gguf_val(f"{arch}.ssm.conv_kernel")
            if ssm_conv_kernel is not None:
                text_config_dict["linear_conv_kernel_dim"] = ssm_conv_kernel

            ssm_state_size = get_gguf_val(f"{arch}.ssm.state_size")
            if ssm_state_size is not None:
                text_config_dict["linear_key_head_dim"] = ssm_state_size
                text_config_dict["linear_value_head_dim"] = ssm_state_size

            ssm_group_count = get_gguf_val(f"{arch}.ssm.group_count")
            if ssm_group_count is not None:
                text_config_dict["linear_num_key_heads"] = ssm_group_count
                text_config_dict["linear_num_value_heads"] = ssm_group_count

            if arch == "qwen35":
                text_config_dict["mamba_ssm_dtype"] = "float32"

            if arch == "qwen35":
                text_config_dict["partial_rotary_factor"] = 0.25
                text_config_dict.pop("rope_theta", None)

            if arch == "internvl_chat":
                text_config_dict["torch_dtype"] = text_config_dict.get("dtype", "bfloat16")
                text_config_dict.pop("dtype", None)
                text_config_dict["architectures"] = ["Qwen2ForCausalLM"]
                text_config_dict["use_cache"] = False
                text_config_dict["use_sliding_window"] = False
                text_config_dict["sliding_window"] = None
                text_config_dict["max_window_layers"] = 70
                text_config_dict["rope_scaling"] = {
                    "factor": 2.0,
                    "rope_type": "dynamic",
                    "type": "dynamic"
                }
                if "bos_token_id" not in text_config_dict:
                    bos = special_token_id_map.get("bos_token_id")
                    pad = special_token_id_map.get("pad_token_id")
                    text_config_dict["bos_token_id"] = bos if bos else pad
                text_config_dict["eos_token_id"] = text_config_dict.get("bos_token_id")
                text_config_dict.pop("pad_token_id", None)
                config_dict["llm_config"] = text_config_dict
                config_dict["hidden_size"] = text_config_dict.get("hidden_size", 1536)
                config_dict["tie_word_embeddings"] = text_config_dict.get(
                    "tie_word_embeddings", False)
                config_dict["torch_dtype"] = "bfloat16"
                config_dict["_commit_hash"] = None
                config_dict["image_fold"] = None
                config_dict["system_message"] = None
                config_dict["template"] = "internvl2_5"
                config_dict["select_layer"] = -1
                config_dict["pad2square"] = False
                config_dict["use_backbone_lora"] = 0
                config_dict["use_llm_lora"] = 0
                config_dict["transformers_version"] = None
                mmproj_reader = self.model.mmproj_reader
                if mmproj_reader is not None:

                    def _mmproj_val(key):
                        f = mmproj_reader.get_field(key)
                        return f.contents() if f else None

                    scale_factor = _mmproj_val("clip.vision.projector.scale_factor") or 2
                    config_dict["downsample_ratio"] = 1.0 / scale_factor
                    preproc_min = _mmproj_val("clip.vision.preproc_min_tiles") or 1
                    preproc_max = _mmproj_val("clip.vision.preproc_max_tiles") or 12
                    config_dict["dynamic_image_size"] = True
                    config_dict["use_thumbnail"] = True
                    config_dict["min_dynamic_patch"] = preproc_min
                    config_dict["max_dynamic_patch"] = preproc_max
                    config_dict["ps_version"] = "v2"
                    vis_image_size = _mmproj_val("clip.vision.image_size") or 448
                    config_dict["force_image_size"] = vis_image_size
                else:
                    config_dict["downsample_ratio"] = 0.5
                    config_dict["dynamic_image_size"] = True
                    config_dict["force_image_size"] = 448
                    config_dict["use_thumbnail"] = True
                    config_dict["min_dynamic_patch"] = 1
                    config_dict["max_dynamic_patch"] = 12
                    config_dict["ps_version"] = "v2"
            else:
                config_dict["text_config"] = text_config_dict
                config_dict["tie_word_embeddings"] = not _has_output if arch == "qwen35" else True

            for token_key, token_str in vlm_info["vlm_tokens"].items():
                for idx in range(len(tokens_list)):
                    tok = decode_token(tokens_list[idx])
                    if tok == token_str:
                        config_dict[token_key] = idx
                        break

            if arch != "internvl_chat":
                config_dict["transformers_version"] = "4.57.0.dev0"
        else:
            config_dict = {
                "architectures": [f"{arch.title()}ForCausalLM"],
                "model_type": model_type,
            }

            gguf_config_keys = _base_gguf_config_keys + [
                (f"{arch}.dimension_count", "rope_dimension_count"),
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

        vision_config = getattr(conv.config, 'vision_config', None)
        if vision_config is not None:
            vc_dict = {}
            for attr in [
                    'hidden_size', 'num_heads', 'depth', 'intermediate_size', 'patch_size',
                    'image_size', 'spatial_merge_size', 'in_channels', 'temporal_patch_size',
                    'num_position_embeddings', 'layer_norm_eps', 'hidden_act', 'projector_type',
                    'deepstack_visual_indexes', 'out_hidden_size', 'mlp_ratio', 'embed_dim',
                    'window_size', 'fullatt_block_indexes'
            ]:
                val = getattr(vision_config, attr, None)
                if val is not None:
                    vc_dict[attr] = val
            _internvl_aliases = {
                'num_attention_heads': 'num_heads',
                'num_hidden_layers': 'depth',
                'num_channels': 'in_channels',
            }
            for src_key, dst_key in _internvl_aliases.items():
                if vc_dict.get(dst_key) is None:
                    val = getattr(vision_config, src_key, None)
                    if val is not None:
                        vc_dict[dst_key] = val
            _internvl_extra_attrs = [
                'num_attention_heads',
                'num_hidden_layers',
                'num_channels',
                'qkv_bias',
                'qk_normalization',
                'use_flash_attn',
                'head_dim',
                'drop_path_rate',
                'norm_type',
                'initializer_factor',
            ]
            for attr in _internvl_extra_attrs:
                val = getattr(vision_config, attr, None)
                if val is not None:
                    vc_dict[attr] = val
            if vlm_info:
                vc_dict["model_type"] = vlm_info["vision_model_type"]
                vc_dict["initializer_range"] = 1e-10
            config_dict["vision_config"] = vc_dict

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
            "tokenizer_class":
            tokenizer_class,
            "model_max_length":
            get_gguf_val(f"{gguf_arch_for_keys}.context_length")
            or get_gguf_val(f"{arch}.context_length") or 32768,
            "clean_up_tokenization_spaces":
            False,
            "add_prefix_space":
            add_prefix_space,
            "errors":
            "replace",
            "split_special_tokens":
            False,
        }

        if model_type in ("qwen3_5", "qwen35", "qwen3_vl"):
            tokenizer_config_dict["pretokenize_regex"] = (
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
                "\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
                "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

        arch_info = PREPROCESSOR_SIZE_TABLE.get(model_type)
        if arch_info and getattr(self.model, 'mmproj_reader', None):
            tokenizer_config_dict["processor_class"] = arch_info['processor_class']

        if vlm_info and getattr(self.model, 'mmproj_reader', None):
            extra_special = {}
            for config_key, output_key in [
                ("image_token_id", "image_token"),
                ("video_token_id", "video_token"),
                ("vision_start_token_id", "vision_bos_token"),
                ("vision_end_token_id", "vision_eos_token"),
            ]:
                token_str = vlm_info["vlm_tokens"].get(config_key)
                if token_str:
                    extra_special[output_key] = token_str
            if extra_special:
                tokenizer_config_dict["extra_special_tokens"] = extra_special

        add_bos = get_gguf_val("tokenizer.ggml.add_bos_token")
        if add_bos is not None:
            add_bos = bool(add_bos)
            tokenizer_config_dict["add_bos_token"] = add_bos

        if add_bos is not None and not add_bos:
            tokenizer_config_dict["bos_token"] = None
        elif bos_str is not None:
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
        pad_token_id = special_token_id_map.get("pad_token_id")
        bos_token_id_val = special_token_id_map.get("bos_token_id")
        if pad_str is not None and pad_token_id is not None and bos_token_id_val is not None:
            if not add_bos and pad_token_id == bos_token_id_val:
                tokenizer_config_dict["pad_token"] = None
            else:
                tokenizer_config_dict["pad_token"] = pad_str
        elif pad_str is not None:
            tokenizer_config_dict["pad_token"] = pad_str

        added_tokens_decoder = {}
        additional_special_tokens = []
        for entry in added_tokens_entries:
            if entry["special"]:
                is_primary = (entry["id"] == special_token_id_map.get("bos_token_id")
                              or entry["id"] == special_token_id_map.get("unk_token_id")
                              or entry["id"] == special_token_id_map.get("sep_token_id"))
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
            chat_template_dict = {"chat_template": chat_template}
            chat_template_path = os.path.join(conv.config_dir, 'chat_template.json')
            with open(chat_template_path, 'w') as f:
                json.dump(chat_template_dict, f, indent=2)
            logger.info("Saved chat_template to %s", chat_template_path)

        tokenizer_config_path = os.path.join(conv.config_dir, 'tokenizer_config.json')
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config_dict, f, indent=2)
        logger.info("Saved tokenizer_config to %s", tokenizer_config_path)

        added_tokens_dict = {entry["content"]: entry["id"] for entry in added_tokens_entries}
        if added_tokens_dict:
            added_tokens_path = os.path.join(conv.config_dir, 'added_tokens.json')
            with open(added_tokens_path, 'w') as f:
                json.dump(added_tokens_dict, f, indent=2)
            logger.info("Saved added_tokens to %s", added_tokens_path)

        special_tokens_map = {}
        for tok_key_name, tok_name in [
            ("bos_token_id", "bos_token"),
            ("eos_token_id", "eos_token"),
            ("unk_token_id", "unk_token"),
            ("sep_token_id", "sep_token"),
            ("pad_token_id", "pad_token"),
        ]:
            tok_id = special_token_id_map.get(tok_key_name)
            if tok_id is not None:
                entry = added_tokens_decoder.get(str(tok_id))
                if entry:
                    special_tokens_map[tok_name] = {
                        "content": entry["content"],
                        "lstrip": entry["lstrip"],
                        "normalized": entry["normalized"],
                        "rstrip": entry["rstrip"],
                        "single_word": entry["single_word"],
                    }
        if additional_special_tokens:
            special_tokens_map["additional_special_tokens"] = additional_special_tokens
        if special_tokens_map:
            special_tokens_path = os.path.join(conv.config_dir, 'special_tokens_map.json')
            with open(special_tokens_path, 'w') as f:
                json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
            logger.info("Saved special_tokens_map to %s", special_tokens_path)

        generation_config_dict = {}
        if "bos_token_id" in special_token_id_map:
            generation_config_dict["bos_token_id"] = special_token_id_map["bos_token_id"]
        if "eos_token_id" in special_token_id_map:
            if vlm_info and "bos_token_id" in special_token_id_map:
                generation_config_dict["eos_token_id"] = [
                    special_token_id_map["eos_token_id"],
                    special_token_id_map["bos_token_id"],
                ]
            else:
                generation_config_dict["eos_token_id"] = special_token_id_map["eos_token_id"]
        if "pad_token_id" in special_token_id_map:
            generation_config_dict["pad_token_id"] = special_token_id_map["pad_token_id"]

        sampling_temp = get_gguf_val("general.sampling.temp")
        sampling_top_k = get_gguf_val("general.sampling.top_k")
        sampling_top_p = get_gguf_val("general.sampling.top_p")
        if sampling_temp is not None or sampling_top_k is not None or sampling_top_p is not None:
            generation_config_dict["do_sample"] = True
        if sampling_top_p is not None:
            generation_config_dict["top_p"] = sampling_top_p
        if sampling_top_k is not None:
            generation_config_dict["top_k"] = sampling_top_k
        if sampling_temp is not None:
            generation_config_dict["temperature"] = sampling_temp
        generation_config_dict["repetition_penalty"] = 1.0
        generation_config_dict["transformers_version"] = "4.56.0"

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
            "task": "image-text-to-text" if vlm_info else "text-generation",
            "allow_remote": True
        }
        configuration_path = os.path.join(conv.config_dir, 'configuration.json')
        with open(configuration_path, 'w') as f:
            json.dump(configuration_dict, f)
        logger.info("Saved configuration to %s", configuration_path)

        mmproj_reader = self.model.mmproj_reader
        if mmproj_reader is not None:
            llm_arch = getattr(conv.config, 'model_type', '')
            _generate_preprocessor_configs(llm_arch, mmproj_reader, conv.config_dir)

    def set_linear_weight(self, conv, path: str, weight_dict: dict, do_lora: bool = False):
        tensor_info = self.model.get_tensor_info(path)
        logger.debug('tensor info of %s : %s', path, tensor_info)

        weight_shape = None

        # Simple quant types (Q4_0, Q4_1, Q8_0, Q8_1) rearrange packed bits
        # directly without dequant→float32→requant, preserving accuracy.
        # Other quant types (K-quant etc.) still fall back to float dequantization.
        is_vision_path = path.startswith("model.visual.") or path.startswith(
            "vision_model.") or path.startswith("mlp1.")
        is_internvl_vit = path.startswith("vision_model.") or path.startswith("mlp1.")
        is_simple_quant = tensor_info and tensor_info.get('quant_type') in (
            GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1)

        if tensor_info and tensor_info.get('is_quantized', False) and (not is_vision_path
                                                                       or is_simple_quant):
            if is_internvl_vit:
                data = self.model.read(path + ".weight")
                if "attn.qkv" in path:
                    weight_dict[path + ".weight"] = data
                else:
                    weight_dict[path + ".weight"] = np.ascontiguousarray(np.transpose(data, (1, 0)))
                weight_shape = weight_dict[path + ".weight"].shape
            else:
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
                else:
                    quant_info = self.model.get_tensor_info(path)
                    quant_shape = (quant_info['shape'][1],
                                   quant_info['shape'][0]) if quant_info else None
                    widen_target = self._tensor_widen_targets.get(path)
                    if widen_target == "q4_1":
                        raw_data = self.model.read_quantized(path)
                        converted = self.quant_converter.widen_q4_to_q4_1(raw_data, quant_shape)
                    elif widen_target == "q8_0":
                        raw_data = self.model.read_quantized(path)
                        qt = tensor_info.get('quant_type')
                        converted = self.quant_converter.widen_q4_to_q8(raw_data, qt, quant_shape)
                    else:
                        converted = self.quant_converter.convert_to_llmconv_format(self.model,
                                                                                   path,
                                                                                   transpose=False)

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
                            'pre_transposed': converted.get('pre_transposed', False),
                        }
                    else:
                        weight_dict[path + ".weight"] = converted['weight']
                        weight_shape = converted['weight'].shape
        else:
            weight_path = path + ".weight"
            if self.model.is_exist(weight_path):
                data = self.model.read(weight_path)
                if is_internvl_vit and "attn.qkv" in path:
                    weight_dict[weight_path] = data
                else:
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

                if gguf_name and (gguf_name in self.model.tensor_map or gguf_name.startswith("__")):
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
                        if is_internvl_vit:
                            bias_data = bias_data.reshape(1, 1, -1)
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

            if len(data.shape) == 1:
                data = data.reshape(1, 1, -1)
            elif len(data.shape) == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])

            if type == WeightType.ZEROCENTERED_RMSNORM:
                pass  # GGUF stores zero-centered norms natively
            weight_dict[weight_path] = data
        if self.model.is_exist(path):
            data = self.model.read(path)
            if len(data.shape) == 1:
                data = data.reshape(1, 1, -1)
            elif len(data.shape) == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])
            weight_dict[path] = data
        if not self.model.is_exist(weight_path) and not self.model.is_exist(path):
            raise RuntimeError(f"Can't find key: {path} or {weight_path}")

        bias_path = path + ".bias"
        if self.model.is_exist(bias_path):
            bias_data = self.model.read(bias_path)
            bias_data = bias_data.reshape(1, 1, -1)
            weight_dict[bias_path] = bias_data

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
                                                                           transpose=False)
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
        return quantize_param, extra_args
