import torch
from torch.nn.parameter import Parameter
from torch.onnx import symbolic_helper

from sophgo_mq.fake_quantize.quantize_base import QuantizeBase
from sophgo_mq.utils import is_symmetric_quant, is_tracing_state
from sophgo_mq.utils.hook import PerChannelLoadHook


class LearnableFakeQuantize(QuantizeBase):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.
    """

    def __init__(self, observer, scale=1., zero_point=0., use_grad_scaling=True, **observer_kwargs):
        super(LearnableFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.use_grad_scaling = use_grad_scaling
        self.scale = Parameter(torch.tensor([scale]))
        self.zero_point = Parameter(torch.tensor([zero_point]))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        # Check whether the module will load a state dict;
        # Initialize the shape of per-channel 'scale' and 'zero-point' before copying values
        self.load_state_dict_hook = PerChannelLoadHook(self)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List[%s]' % str(self.scale.shape),
                   self.zero_point if self.ch_axis == -1 else 'List')

    def forward(self, X):
        x_ori = X
        # Learnable fake quantize have to zero_point.float() to make it learnable.
        if self.observer_enabled[0] == 1:  # or self.only_enable_observer:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled[
                0] == 1:  # and (not self.only_enable_observer or self.run_fquant_time > 0):
            # if self.run_fquant_time > 0:
            #     print('wxc1 run_fquant_time')
            #     self.run_fquant_time -= 1
            if is_symmetric_quant(self.qscheme):
                self.zero_point.data.zero_()
            else:
                self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

            if self.is_per_channel:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                if is_tracing_state():
                    X = FakeQuantizeLearnablePerchannelAffine.apply(X, self.scale, self.zero_point,
                                                                    self.ch_axis, self.quant_min,
                                                                    self.quant_max, grad_factor)
                else:
                    X = _fake_quantize_learnable_per_channel_affine_training(
                        X, self.scale, self.zero_point, self.ch_axis, self.quant_min,
                        self.quant_max, grad_factor)
                x_ori = X
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max)**0.5
                else:
                    grad_factor = 1.0
                scale, zero_point = self.scale, self.zero_point
                # if self.only_enable_observer:
                #     scale, zero_point = 1, 0
                X = torch._fake_quantize_learnable_per_tensor_affine(x_ori, scale, zero_point,
                                                                     self.quant_min, self.quant_max,
                                                                     grad_factor)
                diff = x_ori - X
                if self.only_enable_observer:
                    x_ori = X + diff.detach()
                else:
                    x_ori = X

        return x_ori


def _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min,
                                                         quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)


class FakeQuantizeLearnablePerchannelAffine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
        return _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis,
                                                                    quant_min, quant_max,
                                                                    grad_factor)

    @staticmethod
    def symbolic(g, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
        output = g.op("Sophgo_custom::FakeQuantizeLearnablePerchannelAffine",
                      x,
                      scale,
                      zero_point,
                      quant_min_i=quant_min,
                      quant_max_i=quant_max)

        input_shape = symbolic_helper._get_tensor_sizes(x)
        if input_shape is not None and hasattr(x.type(), 'with_sizes'):
            output_type = x.type().with_sizes(input_shape)
            output.setType(output_type)

        return output
