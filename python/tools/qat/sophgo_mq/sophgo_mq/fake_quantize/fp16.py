import torch
import os
import yaml
from easydict import EasyDict
from .quantize_base import QuantizeBase
from ..utils.hook import PerChannelLoadHook

class Fp16FakeQuantize(QuantizeBase):
    """This is FP16 Quantization Emulator"""

    def __init__(self, observer, **observer_kwargs):
        super(Fp16FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.load_state_dict_hook = PerChannelLoadHook(self)
        self.dtype = 'FP16'

    def forward(self, X):
        tensor_quant = torch.zeros_like(X)
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel: # per-channel
                channels = X.shape[0]
                for c in range(channels):
                    sub_tensor = X.select(0, c).detach()
                    sub_tensor_quant = sub_tensor.to(torch.float16)
                    sub_tensor_quant = sub_tensor_quant.to(torch.float)
                    tensor_quant.select(0, c).data.copy_(sub_tensor_quant)
                X = tensor_quant
            else: # per-tensor
                X_quant = X.to(torch.float16)
                X = X_quant.to(torch.float)
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
        super(Fp16FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
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
        super(Fp16FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)

class BF16FakeQuantize(QuantizeBase):
    """This is BF16 Quantization Emulator."""
    def __init__(self, observer, **observer_kwargs):
        super(BF16FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.dtype = 'BF16'

    def forward(self, X):
        if self.fake_quant_enabled[0] == 1 and isinstance(X, torch.Tensor):
            X_quant = X.to(torch.bfloat16)
            X = X_quant.to(torch.float)
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}'.format(
                   self.fake_quant_enabled, self.observer_enabled)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(BF16FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(BF16FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
