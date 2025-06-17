import numpy as np
import torch
from .utils import *
import transform.TpuLang as tpul


def test_qwen2_7b_block_cache(tester: TPULANG_CONVERTER):

    @tpulang(tester.chip)
    def _test_qwen2_7b_block_cache(dtype, file_path, cos_sin_path, group_size, bits):
        model_weights = torch.load(file_path, map_location=torch.device('cpu'))

        tensors = read_weights(model_weights)

        # preprocess in tpulang
        cos_sin_weight = torch.load(cos_sin_path, map_location=torch.device('cpu'))
        mrope_section = [16, 24, 24]
        cos, sin = [], []
        for i in range(3):
            cos_split = cos_sin_weight['cos'].split(mrope_section * 2, dim=-1)[i][i].float().numpy()
            sin_split = cos_sin_weight['sin'].split(mrope_section * 2, dim=-1)[i][i].float().numpy()
            cos.append(
                tpul.Tensor(dtype="float32",
                            shape=list(cos_split.shape),
                            data=cos_split,
                            ttype="coeff"))
            sin.append(
                tpul.Tensor(dtype="float32",
                            shape=list(sin_split.shape),
                            data=sin_split,
                            ttype="coeff"))

        seq_length = tester.seq_len
        num_key_value_heads = 4
        head_dim = 128
        hidden_size = tensors['q_proj']['weights'].shape[1]

        hidden_states = rand_data((1, 1, hidden_size), 'float32')
        position_ids = np.array(3 * [[range(1)]], dtype=np.int32)
        attention_mask = np.ones((1, 1, 1, seq_length + 1), dtype=np.float32)
        past_k_cache = rand_data((1, seq_length, num_key_value_heads, head_dim), 'float32')
        past_v_cache = rand_data((1, seq_length, num_key_value_heads, head_dim), 'float32')

        hidden_states = tpul.Tensor(dtype="float32",
                                    shape=list(hidden_states.shape),
                                    data=hidden_states,
                                    name="input_states")
        position_ids = tpul.Tensor(dtype="int32",
                                   shape=list(position_ids.shape),
                                   data=position_ids,
                                   name="position_ids")
        attention_mask = tpul.Tensor(dtype="float32",
                                     shape=list(attention_mask.shape),
                                     data=attention_mask,
                                     name="attention_mask")
        past_k_cache = tpul.Tensor(dtype="float32",
                                   shape=list(past_k_cache.shape),
                                   data=past_k_cache,
                                   name="past_k")
        past_v_cache = tpul.Tensor(dtype="float32",
                                   shape=list(past_v_cache.shape),
                                   data=past_v_cache,
                                   name="past_v")

        output, past_k, past_v = tpul.qwen2_block_cache(hidden_states,
                                                        position_ids,
                                                        attention_mask,
                                                        past_k_cache,
                                                        past_v_cache,
                                                        tensors['q_proj']['weights'],
                                                        tensors['q_proj']['scales'],
                                                        tensors['q_proj']['zeros'],
                                                        tensors['q_proj']['bias'],
                                                        tensors['k_proj']['weights'],
                                                        tensors['k_proj']['scales'],
                                                        tensors['k_proj']['zeros'],
                                                        tensors['k_proj']['bias'],
                                                        tensors['v_proj']['weights'],
                                                        tensors['v_proj']['scales'],
                                                        tensors['v_proj']['zeros'],
                                                        tensors['v_proj']['bias'],
                                                        tensors['o_proj']['weights'],
                                                        tensors['o_proj']['scales'],
                                                        tensors['o_proj']['zeros'],
                                                        tensors['o_proj']['bias'],
                                                        tensors['down_proj']['weights'],
                                                        tensors['down_proj']['scales'],
                                                        tensors['down_proj']['zeros'],
                                                        tensors['gate_proj']['weights'],
                                                        tensors['gate_proj']['scales'],
                                                        tensors['gate_proj']['zeros'],
                                                        tensors['up_proj']['weights'],
                                                        tensors['up_proj']['scales'],
                                                        tensors['up_proj']['zeros'],
                                                        tensors['input_layernorm_weight'],
                                                        tensors['post_attention_layernorm_weight'],
                                                        cos,
                                                        sin,
                                                        out_dtype=dtype,
                                                        group_size=group_size,
                                                        weight_bits=bits,
                                                        hidden_size=hidden_size,
                                                        quant_method="gptq",
                                                        out_name="output")

        tester.compile_and_check(
            tester.unique_name("Qwen2_7B_Block_Cache"),
            [hidden_states, position_ids, attention_mask, past_k_cache, past_v_cache],
            [output, past_k, past_v])

    _test_qwen2_7b_block_cache("float32", tester.file_path, tester.cos_sin_path, 128, tester.bits)
