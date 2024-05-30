import torch
import math

from sophgo_mq.fake_quantize.quantize_base import QuantizeBase
from sophgo_mq.utils.hook import PerChannelLoadHook

from sophgo_mq.fake_quantize.quantize_base import _version_under_1100
import ipdb
from scipy.stats import norm
NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
AF4=[-1.0, 
     -0.69441008, 
     -0.51243739,
      -0.3736951, 
     -0.25607552, 
     -0.14982478,
     -0.04934812,  
     0.0, 
     0.04273164, 
     0.12934483, 
     0.21961274, 
     0.31675666,
     0.42563882,  
     0.55496234,  
     0.72424863,  
     1.0,
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

#NF8 compute
offset=(1-(1/(255*2))+1-(1/(256*2)))*(1/2) #0.9981
v1 = norm.ppf(torch.linspace(offset, 0.5, 129)[:-1]).tolist()
v3 = (-norm.ppf(torch.linspace(offset, 0.5, 128)[:-1])).tolist()
v=v1+v3+[0]
NF8 = torch.Tensor(v)
NF8 = NF8.sort().values
NF8 /= NF8.max()

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1,"af4":AF4,"nf8":NF8}
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT,"af4":AF4,"nf8":NF8}
class FP4FakeQuantize(QuantizeBase):
    """This is fp4 Quantization Emulator..
    """
    def __init__(self, observer, **observer_kwargs):
        super(FP4FakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.load_state_dict_hook = PerChannelLoadHook(self)
        self.data_type="fp4_e2m1"
        self.quantile=1.0
        self.return_int=False

    def forward(self, X):
        assert self.data_type in FLOAT_MAPPING, "unexpected data type."
        allow_data = FLOAT_MAPPING[self.data_type]                 #float类型
        allow_data_bit = INT_MAPPING[self.data_type]               #int类型
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale = X.abs().max(1)[0] * self.quantile/ max(allow_data)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            _scale =X.abs().max(1)[0] * self.quantile/ max(allow_data)
            _scale.unsqueeze_(dim=-1)
            X = X/_scale
            # if self.data_type.lower=="nf4":
            #     cdf_values = [norm.cdf(x) for x in allow_data]
            #     intermediate_cdf_values = [(cdf_values[i] + cdf_values[i+1]) / 2 for i in range(len(allow_data) - 1)]
            #     mid_data = norm.ppf(intermediate_cdf_values)
            # else:
            #     mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
            mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
            q_X= torch.zeros_like(X)
            for i in range(len(allow_data)):
                data = allow_data_bit[i] if self.return_int else allow_data[i]
                if i == 0:
                    q_X += torch.where(X <= mid_data[i], data, 0)
                elif i == len(allow_data) - 1:
                    q_X += torch.where(X > mid_data[i - 1], data, 0)
                else:
                    q_X += torch.where((mid_data[i - 1] < X) & (X <= mid_data[i]), data, 0)
            # if self.return_int:
            #     return q_X.type(torch.int8), _scale.type(torch.float), None
            X=q_X * _scale
        return X

    @torch.jit.export
    def extra_repr(self):
        allow_data = FLOAT_MAPPING[self.data_type] 
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   min(allow_data), max(allow_data),
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List', 
                   self.zero_point if self.ch_axis == -1 else 'List')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FP4FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
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
        super(FP4FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)