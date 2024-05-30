import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Linear
from torch.nn.intrinsic import _FusedModule
from torch.nn.parameter import Parameter

from sophgo_mq.nn.intrinsic import LinearBn1d
import torch.nn.intrinsic as nni


class LinearBn1d_sophgo(Linear, _FusedModule):
    _version = 2
    _FLOAT_MODULE = LinearBn1d

    def __init__(self,
                 # ConvNd args
                 in_features, out_features, bias,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        Linear.__init__(self, in_features, out_features, False)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for Linear, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(LinearBn1d, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self 

    def bias_fake_quant(self, bias, scale_w, in_scale):
        if bias is not None:
            scale = scale_w*in_scale
            if torch.nonzero(scale).size()[0] != scale.numel():
                print('Linear error! scale has 0, scale:', scale)

            bias_q = bias/scale
            bias = (bias_q.round()-bias_q).detach() + bias_q
            bias = bias*scale
        return bias

    # def _forward(self, input):
    #     in_scale = self.input_fake_quantizer.scale #����һ��activation_fake_quant�ڵ��ȡscale
    #     conv = F.linear(input, self.weight_fake_quant(self.weight), 
    #         self.bias_fake_quant(self.bias, self.weight_fake_quant.scale, in_scale))
    #     return conv

    def _forward(self, input):
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        # input.shape = (batch_size, in_features, *)
        # scale_factor.shape = (out_feature, )
        # self.weight.shape = (out_feature, in_feature, *)
        # self.bias.shape = (out_feature, *)
        # output.shape = (batch_size, out_feature, *)
        if self.bn.affine:
            scale_factor = self.bn.weight / running_std
        else:
            scale_factor = 1. / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(input.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original Linear
        # will be added later
        # Linear layer takes permuted input since the format is (batch_size, *, in_features)
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
            fc_bias = self.bias 
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
            fc_bias = torch.zeros_like(zero_bias, device=scaled_weight.device)
        if self.bn.affine:
            full_bias = (fc_bias - self.bn.running_mean) / running_std * self.bn.weight + self.bn.bias 
        else:
            full_bias = (fc_bias - self.bn.running_mean) / running_std 
        in_scale = self.input_fake_quantizer.scale #����һ��activation_fake_quant�ڵ��ȡscale
        fquant_bias = self.bias_fake_quant(full_bias, self.weight_fake_quant.scale, in_scale)
        linear_out = F.linear(input, scaled_weight, fquant_bias)
        linear_orig = (linear_out - full_bias) / scale_factor.reshape(bias_shape) + fc_bias.reshape(bias_shape)
        linear_out = self.bn(linear_orig)
        return linear_out

    def forward(self, input):
        # return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
        return self._forward(input)

    def extra_repr(self):
        return super(LinearBn1d, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(LinearBn1d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, False,
                           bn.eps, bn.momentum,
                           False,
                           qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_linearbn

class Linear_sophgo(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def bias_fake_quant(self, bias, scale_w, in_scale):
        if bias is not None:
            scale = scale_w*in_scale
            if torch.nonzero(scale).size()[0] != scale.numel():
                print('Linear error! scale has 0, scale:', scale)
                scale[torch.abs(scale) < 1e-10] = 1e-10
                print('new scale:', scale)

            bias_q = bias/scale
            bias = (bias_q.round()-bias_q).detach() + bias_q
            bias = bias*scale
        return bias

    def _forward(self, input):
        assert hasattr(self, 'input_fake_quantizer')
        in_scale = self.input_fake_quantizer.scale #����һ��activation_fake_quant�ڵ��ȡscale
        # conv = F.linear(input, self.weight_fake_quant(self.weight),
            # self.bias_fake_quant(self.bias, self.weight_fake_quant.scale, in_scale))
        if self.bias is not None and self.weight_fake_quant.fake_quant_enabled[0] == 1:
            conv = F.linear(input, self.weight_fake_quant(self.weight),
            self.bias_fake_quant(self.bias, self.weight_fake_quant.scale, in_scale))
        else:
            conv = F.linear(input, self.weight_fake_quant(self.weight), self.bias)

        return conv

    def forward(self, input):
        # return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
        return self._forward(input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if type(mod) == nni.LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        return qat_linear

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear


class LinearReLU_sophgo(Linear_sophgo):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super(LinearReLU_sophgo, self).__init__(in_features, out_features, bias, qconfig)

    def forward(self, input):
        return F.relu(Linear_sophgo._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU_sophgo, cls).from_float(mod)

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.nn.intrinsic.LinearReLU(linear, relu)
