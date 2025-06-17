import numpy as np
import transform.TpuLang as tpul


class TPULANG_CONVERTER(object):
    ID = 0

    def __init__(self, chip, mode, file_path, cos_sin_path, bits, seq_len):
        self.chip = chip
        self.mode = mode
        self.file_path = file_path
        self.cos_sin_path = cos_sin_path
        self.bits = bits
        self.seq_len = seq_len

    def unique_name(self, name):
        name = "{}_{}".format(name, TPULANG_CONVERTER.ID)
        TPULANG_CONVERTER.ID += 1
        return name

    def compile_and_check(self, model_name, inputs, outputs):
        for input in inputs:
            assert input.ttype == "neuron", "coeff Tensor should not be input {}".format(input.name)

        tpul.compile_f32(model_name,
                         inputs,
                         outputs,
                         cmp=True,
                         mode=self.mode,
                         no_save=False,
                         top_mlir_inference=True,
                         tpu_mlir_inference=True,
                         log_level='normal')


def rand_data(shape, dtype, min=-10, max=10, seed=None, int_satu=False):
    if seed is not None:
        np.random.seed(seed)
    if dtype in ['float32', 'float16']:
        return np.clip(np.random.randn(*shape).astype(dtype), min, max)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        if int_satu:
            return np.clip(np.random.randint(0, 127, size=shape).astype(dtype), min, max)
        else:
            return np.random.randint(0, 127, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


def tpulang(chip):

    def wrapper(func):

        def decorate(*args, **kwargs):
            tpul.init(chip)
            ret = func(*args, **kwargs)
            tpul.deinit()
            return ret

        return decorate

    return wrapper


def to_tpul_tensor(key, model_weights, use_float, dtype):
    tensor = model_weights[key]
    data = tensor.float().numpy() if use_float else tensor.numpy()
    return tpul.Tensor(dtype=dtype, shape=list(data.shape), data=data, ttype="coeff")


def read_weights(model_weights):
    # matmul
    projections = {
        "q_proj": {
            "prefix": "self_attn.q_proj",
            "has_bias": True
        },
        "k_proj": {
            "prefix": "self_attn.k_proj",
            "has_bias": True
        },
        "v_proj": {
            "prefix": "self_attn.v_proj",
            "has_bias": True
        },
        "o_proj": {
            "prefix": "self_attn.o_proj",
            "has_bias": False
        },
        "gate_proj": {
            "prefix": "mlp.gate_proj",
            "has_bias": False
        },
        "up_proj": {
            "prefix": "mlp.up_proj",
            "has_bias": False
        },
        "down_proj": {
            "prefix": "mlp.down_proj",
            "has_bias": False
        },
    }

    tensors = {}

    for name, cfg in projections.items():
        prefix = cfg["prefix"]
        tensors[name] = {
            "weights":
            to_tpul_tensor(f"{prefix}.qweight", model_weights, use_float=False, dtype="int32"),
            "scales":
            to_tpul_tensor(f"{prefix}.scales", model_weights, use_float=True, dtype="float32"),
            "zeros":
            to_tpul_tensor(f"{prefix}.qzeros", model_weights, use_float=False, dtype="int32"),
            "bias":
            to_tpul_tensor(f"{prefix}.bias", model_weights, use_float=True, dtype="float32")
            if cfg["has_bias"] else None,
        }

    # ln
    input_layernorm_weight = model_weights['input_layernorm.weight'].float().numpy()
    post_attention_layernorm_weight = model_weights['post_attention_layernorm.weight'].float(
    ).numpy()
    input_layernorm_weight = tpul.Tensor(dtype="float32",
                                         shape=list(input_layernorm_weight.shape),
                                         data=input_layernorm_weight,
                                         ttype="coeff")
    post_attention_layernorm_weight = tpul.Tensor(dtype="float32",
                                                  shape=list(post_attention_layernorm_weight.shape),
                                                  data=post_attention_layernorm_weight,
                                                  ttype="coeff")
    tensors['input_layernorm_weight'] = input_layernorm_weight
    tensors['post_attention_layernorm_weight'] = post_attention_layernorm_weight

    return tensors
