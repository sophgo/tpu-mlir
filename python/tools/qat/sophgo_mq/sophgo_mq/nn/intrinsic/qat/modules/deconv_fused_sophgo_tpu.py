import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.intrinsic import _FusedModule
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair, _single

from typing import TypeVar

import sophgo_mq.nn.intrinsic as qnni
import sophgo_mq.nn.qat as qnnqat
from sophgo_mq.utils.fusion import fuse_deconv_bn_weights

_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

MOD = TypeVar('MOD', bound=nn.modules.conv._ConvTransposeNd)


class _ConvTransposeBnNd(nn.modules.conv._ConvTransposeNd, _FusedModule):

    _version = 2
    _FLOAT_MODULE = MOD

    def __init__(
            self,
            # ConvTransposeBnNd args
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias,
            transposed,
            padding,
            output_padding,
            groups,
            dilation,
            padding_mode,
            # bn args
            # BatchNormNd args
            # num_features: out_channels
            eps=1e-05,
            momentum=0.1,
            # affine: True
            # track_running_stats: True
            # Args for this module
            freeze_bn=False,
            qconfig=None,
            dim=2):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        nn.modules.conv._ConvTransposeNd.__init__(self, in_channels, out_channels, kernel_size,
                                                  stride, padding, dilation, transposed,
                                                  output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for a QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        # ConvTranspose do per-channel quantize on output channel.
        if self.weight_fake_quant.ch_axis != -1:
            self.weight_fake_quant.ch_axis = 1
            self.weight_fake_quant.activation_post_process.ch_axis = 1
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

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
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ConvTransposeBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    # def _forward(self, input):
    #     assert self.bn.running_var is not None
    #     running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
    #     scale_factor = self.bn.weight / running_std
    #     weight_shape = [1] * len(self.weight.shape)
    #     weight_shape[1] = -1
    #     bias_shape = [1] * len(self.weight.shape)
    #     bias_shape[1] = -1
    #     scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
    #     # using zero bias here since the bias for original conv
    #     # will be added later
    #     if self.bias is not None:
    #         zero_bias = torch.zeros_like(self.bias)
    #     else:
    #         zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
    #     deconv = self._convtransposed_forward(input, scaled_weight, zero_bias)
    #     deconv_orig = deconv / scale_factor.reshape(bias_shape)
    #     if self.bias is not None:
    #         deconv_orig = deconv_orig + self.bias.reshape(bias_shape)
    #     deconv = self.bn(deconv_orig)
    #     return deconv

    def bias_fake_quant_proc(self, bias, scale_w, in_scale):
        scale = scale_w * in_scale
        if torch.nonzero(scale).size()[0] != scale.numel():
            print('error! scale has 0, scale:', scale)
        bias_q = bias / scale
        bias = (bias_q.round() - bias_q).detach() + bias_q
        # bias_q = torch.clamp(bias_q, -2147483648, 2147483647)
        bias = bias * scale
        return bias

    # def _forward(self, input):
    #     assert self.bn.running_var is not None
    #     running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
    #     scale_factor = self.bn.weight / running_std
    #     weight_shape = [1] * len(self.weight.shape)
    #     weight_shape[1] = -1
    #     bias_shape = [1] * len(self.weight.shape)
    #     bias_shape[1] = -1
    #     scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
    #     if self.weight_fake_quant.fake_quant_enabled[0] == 1:
    #         _, fused_bias = fuse_deconv_bn_weights(self.weight, self.bias,
    #                             self.bn.running_mean, self.bn.running_var, self.bn.eps, self.bn.weight, self.bn.bias)
    #         in_scale = self.input_fake_quantizer.scale #从上一个activation_fake_quant节点获取scale
    #         scale_fused_bias = self.bias_fake_quant_proc(fused_bias, self.weight_fake_quant.scale, in_scale)
    #         diff_fused_bias = fused_bias - scale_fused_bias

    #     if self.bias is not None:
    #         zero_bias = torch.zeros_like(self.bias)
    #     else:
    #         zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
    #     conv = self._convtransposed_forward(input, scaled_weight, zero_bias)
    #     conv_orig = conv / scale_factor.reshape(bias_shape)
    #     if self.bias is not None:
    #         conv_orig = conv_orig + self.bias.reshape(bias_shape)
    #     conv = self.bn(conv_orig)
    #     if self.weight_fake_quant.fake_quant_enabled[0] == 1:
    #         conv -= diff_fused_bias.reshape(bias_shape) #这里从推导看应该是减
    #     return conv

    def _forward(self, input):
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[1] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
            conv_bias = self.bias
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
            conv_bias = torch.zeros_like(zero_bias, device=scaled_weight.device)
        if self.bn.affine:
            full_bias = (conv_bias -
                         self.bn.running_mean) / running_std * self.bn.weight + self.bn.bias
        else:
            full_bias = (conv_bias - self.bn.running_mean) / running_std
        # quant_bias = self.bias_fake_quant(full_bias)
        quant_bias = self.bias_fake_quant_proc(full_bias, self.weight_fake_quant.scale,
                                               self.input_fake_quantizer.scale)
        conv_with_bias = self._convtransposed_forward(input, scaled_weight, quant_bias)
        deconv_orig = (conv_with_bias - full_bias.reshape(bias_shape)
                       ) / scale_factor.reshape(bias_shape) + conv_bias.reshape(bias_shape)
        deconv = self.bn(deconv_orig)
        return deconv

    def _convtransposed_forward(self, x, w, b):
        raise NotImplementedError(
            'The sub-class must implement this function to forward in the needed dim-version!')

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvTransposeBnNd, self).extra_repr()

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

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
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

        super(_ConvTransposeBnNd,
              self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                          unexpected_keys, error_msgs)

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
        deconv, bn = mod[0], mod[1]
        qat_deconvbn = cls(deconv.in_channels, deconv.out_channels, deconv.kernel_size,
                           deconv.stride, deconv.bias is not None, deconv.transposed,
                           deconv.padding, deconv.output_padding, deconv.groups, deconv.dilation,
                           deconv.padding_mode, bn.eps, bn.momentum, False, qconfig)
        qat_deconvbn.weight = deconv.weight
        qat_deconvbn.bias = deconv.bias
        qat_deconvbn.bn.weight = bn.weight
        qat_deconvbn.bn.bias = bn.bias
        qat_deconvbn.bn.running_mean = bn.running_mean
        qat_deconvbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_deconvbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_deconvbn


class ConvTransposeBn2d_sophgo(_ConvTransposeBnNd, nn.ConvTranspose2d):
    _FLOAT_MODULE = qnni.ConvTransposeBn2d

    def __init__(
            self,
            # ConvTransposeBnNd args
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=None,
            transposed=True,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            padding_mode='zeros',
            # bn args
            # BatchNormNd args
            # num_features: out_channels
            eps=1e-05,
            momentum=0.1,
            # affine: True
            # track_running_stats: True
            # Args for this module
            freeze_bn=False,
            qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvTransposeBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, bias,
                                    transposed, padding, output_padding, groups, dilation,
                                    padding_mode, eps, momentum, freeze_bn, qconfig)

    def _convtransposed_forward(self, x, w, b):
        output_padding = self._output_padding(x, None, self.stride, self.padding, self.kernel_size,
                                              self.dilation)
        return F.conv_transpose2d(x, w, b, self.stride, self.padding, output_padding, self.groups,
                                  self.dilation)


class ConvTransposeBnReLU2d_sophgo(ConvTransposeBn2d_sophgo):
    _FLOAT_MODULE = qnni.ConvTransposeBnReLU2d

    def __init__(
            self,
            # ConvTransposeBnNd args
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=None,
            transposed=True,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            padding_mode='zeros',
            # bn args
            # BatchNormNd args
            # num_features: out_channels
            eps=1e-05,
            momentum=0.1,
            # affine: True
            # track_running_stats: True
            # Args for this module
            freeze_bn=False,
            qconfig=None):
        # super(ConvTransposeBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
        #                                             padding, dilation, groups, bias,
        #                                             padding_mode, eps, momentum,
        #                                             freeze_bn,
        #                                             qconfig)
        super(ConvTransposeBnReLU2d_sophgo, self).__init__(in_channels,
                                                           out_channels,
                                                           kernel_size,
                                                           stride=stride,
                                                           bias=bias,
                                                           transposed=transposed,
                                                           padding=padding,
                                                           output_padding=output_padding,
                                                           groups=groups,
                                                           dilation=dilation,
                                                           padding_mode=padding_mode,
                                                           eps=eps,
                                                           momentum=momentum,
                                                           freeze_bn=freeze_bn,
                                                           qconfig=qconfig)

    def forward(self, input):
        return F.relu(ConvTransposeBn2d_sophgo._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeBnReLU2d_sophgo, cls).from_float(mod)


class ConvTransposeReLU2d_sophgo(qnnqat.ConvTranspose2d_sophgo):
    _FLOAT_MODULE = qnni.ConvTransposeReLU2d
    _FLOAT_DECONV_MODULE = nn.ConvTranspose2d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(
            self,
            # ConvTransposeBnNd args
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=None,
            transposed=True,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            padding_mode='zeros',
            qconfig=None):

        super(ConvTransposeReLU2d_sophgo, self).__init__(in_channels,
                                                         out_channels,
                                                         kernel_size,
                                                         stride=stride,
                                                         bias=bias,
                                                         padding=padding,
                                                         output_padding=output_padding,
                                                         groups=groups,
                                                         dilation=dilation,
                                                         padding_mode=padding_mode,
                                                         qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'

    def forward(self, input, output_size=None):
        return F.relu(qnnqat.ConvTranspose2d_sophgo.forward(input, output_size))
