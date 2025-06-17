import numpy as np
import torch
from .utils import *
import transform.TpuLang as tpul


def test_qwen2_7b_block(tester: TPULANG_CONVERTER):

    @tpulang(tester.chip)
    def _test_qwen2_7b_block(dtype, file_path, cos_sin_path, group_size, bits):
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
        hidden_size = tensors['q_proj']['weights'].shape[1]
        hidden_states = rand_data((1, seq_length, hidden_size), 'float32')
        position_ids = np.array(3 * [[range(seq_length)]], dtype=np.int32)
        attention_mask = rand_data((1, 1, seq_length, seq_length), 'float32')

        # if use real input
        # hidden_states = np.load("/workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/ori_input.npz")["hidden_states"]
        # position_ids = np.load("/workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/ori_input.npz")["position_ids"].astype("float32")
        # attention_mask = np.load("/workspace/Qwen2-VL-7B-Instruct-GPTQ-Int4/ori_input.npz")["attention_mask"]

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

        output, past_k, past_v = tpul.qwen2_block(hidden_states,
                                                  position_ids,
                                                  attention_mask,
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

        tester.compile_and_check(tester.unique_name("Qwen2_7B_Block"),
                                 [hidden_states, position_ids, attention_mask],
                                 [output, past_k, past_v])

    _test_qwen2_7b_block("float32", tester.file_path, tester.cos_sin_path, 128, tester.bits)
