# E5M2的Fake Quantize

import os
import yaml
from easydict import EasyDict
import torch
from sophgo_mq.utils import is_symmetric_quant
from .quantize_base import QuantizeBase
from ..utils.hook import PerChannelLoadHook
from tools.qat.sophgo_mq.FP8_Emulator.pytquant.cpp import fpemu_cpp
if torch.cuda.is_available():
    from tools.qat.sophgo_mq.FP8_Emulator.pytquant.cuda import fpemu_cuda

_version_under_1100 = int(torch.__version__.split('.')[1]) < 10
# List the supported rounding method for E5M2:
mode_list = ["E5M2_RTE", "E5M2_RNE", "E5M2_STOCHASTIC", "E5M2_RNAZ",
             "E5M2_RNTZ", "E5M2_RPINF", "E5M2_RNINF", "E5M2_DAZ_RNE", 
             "E5M2_DAZ_STOCHASITC", "E5M2_DAZ_RNAZ", "E5M2_DAZ_RNTZ"]
# Get the config
def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config

def get_flt_max(mode):
    if mode.lower() == "e5m2":
        return float(57344.0) 
    elif mode.lower() == "e4m3":
        return float(448.0)
    
def get_flt_min(mode):
    if mode.lower() =="e5m2":
        return float(1.5258789E-05) # Min Subnormal
    elif mode.lower() == "e4m3":
        return float(1.9531250E-03)

def quantize_to_integer(tensor, mode, inplace=False):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    nbits = int(mode.split("INT")[1])-1
    q_min = -1*2**nbits
    q_max = (2**nbits)-1
    q_min = -128
    q_max = 127
    if mode == "INT4":
        q_min = -8
        q_max = 7
    # compute scale and zero_point 
    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = q_min - (min_val / scale)
    # Quantize the input tensor using int8 representation
    qtensor = torch.round((tensor / scale) + zero_point)
    # Clamp the values to the int8 range
    qtensor = torch.clamp(qtensor, q_min, q_max)
    # Dequantize the tensor
    dqtensor = scale * (qtensor - zero_point)

    if inplace is True:
        tensor.data.copy_(dqtensor)
        return tensor
    
    return dqtensor

# The function below excuted the FP8 quantization
def fpemu_device_fn(tensor, mode, inplace=True, scale=1.0, zero_point=0.0, quant_min=float(1.5258789E-05), quant_max=float(57344.0), is_per_channel=True):
    if "INT8" in mode or "INT4" in mode:
        return quantize_to_integer(tensor, mode.split("_")[0], inplace=inplace)

    if is_per_channel:
        if tensor.is_cuda :
            X = fpemu_cuda.FPEmuOp_cuda_per_channel.apply(tensor, mode, inplace, scale, zero_point, quant_min, quant_max)
        else :
            X = fpemu_cpp.FPEmuOp_cpp_per_channel.apply(tensor, mode, inplace, scale, zero_point, quant_min, quant_max)
    else:
        if tensor.is_cuda :
            X = fpemu_cuda.FPEmuOp_cuda_per_tensor.apply(tensor, mode, inplace, scale, zero_point, quant_min, quant_max)
        else :
            X = fpemu_cpp.FPEmuOp_cpp_per_tensor.apply(tensor, mode, inplace, scale, zero_point, quant_min, quant_max)

    return X

class E5M2FakeQuantize(QuantizeBase):
    """This is fp8 E5M2 Quantization Emulator."""

    def __init__(self, observer, **observer_kwargs):
        super(E5M2FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.load_state_dict_hook = PerChannelLoadHook(self)
        self.data_type = "fp8_e5m2"

    def forward(self, X):
        tensor_q = torch.zeros_like(X)
        #scaling_method = parse_config('config.yaml').quant.scaling_method
        scaling_method = "mean"
        if self.observer_enabled[0] == 1: # enable the observer
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            if scaling_method.lower() == "mean":
                mean = torch.mean(abs(torch.flatten(X.detach())))
                mean = abs(mean) if abs(mean) > 1e-5 else get_flt_min("e5m2")
                if abs(mean) > 0.0:
                    _scale = get_flt_min("e5m2") / abs(mean)
            elif scaling_method.lower() == "max":
                vmax = torch.max(abs(torch.flatten(X.detach()))) 
                _scale = vmax / get_flt_max("e5m2")
                _scale = torch.tensor(6.55e+04) if _scale.item() > 3.275e+04 else _scale
            else:
                _scale = torch.tensor(1.0)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
            ### TMP workaround
            # self.zero_point = self.zero_point.float()
            # self.scale = torch.ones_like(self.zero_point).to(X.device).float()

        if self.fake_quant_enabled[0] == 1: # enable fake quantize
            if is_symmetric_quant(self.qscheme):
                self.zero_point.data.zero_()
            else:
                self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()
            work_mode = "E5M2_RNE"
            quant_min=get_flt_min("e5m2")
            quant_max=get_flt_max("e5m2")
            if scaling_method.lower() == "mean":
                mean = torch.mean(abs(torch.flatten(X.detach())))
                mean = abs(mean) if abs(mean) > 1e-5 else get_flt_min("e5m2")
                if abs(mean) > 0.0:
                    _scale = torch.tensor(get_flt_min("e5m2")) / abs(mean)
            elif scaling_method.lower() == "max":
                vmax = torch.max(abs(torch.flatten(X.detach())))
                _scale = vmax / get_flt_max("e5m2")
                _scale = torch.tensor(6.55e+04) if _scale.item() > 3.275e+04 else _scale
            else:
                _scale = torch.tensor(1.0)
            _scale = _scale.to(self.scale.device)
            self.scale.copy_(_scale)
            scale_fixed = torch.ones_like(self.zero_point).to(X.device).float()
            if self.is_per_channel: # per-channel
                X = fpemu_device_fn(X, mode=work_mode, inplace=False, scale=scale_fixed, zero_point=self.zero_point, quant_min=quant_min, quant_max=quant_max, is_per_channel=True)
            else: # per-tensor
                # Temporarily fix the scale to 1.0 to match the implement in tpu-mlir
                X = fpemu_device_fn(X, mode=work_mode, inplace=False, scale=scale_fixed, zero_point=self.zero_point, quant_min=quant_min, quant_max=quant_max, is_per_channel=False)
        return X
 
    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List', 
                   self.zero_point if self.ch_axis == -1 else 'List')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(E5M2FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
    
        super(E5M2FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
