import torch.nn.qat.modules as nnqat
import torch.nn as nn
# import torch

from sophgo_mq.quantization.default_bias_fake_quant import bias_fake_quantizer

class Conv2d(nnqat.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', qconfig=None, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig) 
        self.bias_fake_quant = bias_fake_quantizer()

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias_fake_quant(self.bias)) 

class Conv2d_sophgo(nnqat.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', qconfig=None, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig) 

    def bias_fake_quant_proc(self, bias, scale_w, in_scale):
        scale = scale_w*in_scale
        bias_q = bias/scale
        bias = (bias_q.round()-bias_q).detach() + bias_q
        # bias_q = torch.clamp(bias_q, -2147483648, 2147483647)
        bias = bias*scale
        return bias

    def forward(self, input):
        #bias伪量化
        bias = self.bias
        if self.bias is not None and self.weight_fake_quant.fake_quant_enabled[0] == 1:
            in_scale = self.input_fake_quantizer.scale #从上一个activation_fake_quant节点获取scale
            bias = self.bias_fake_quant_proc(self.bias, self.weight_fake_quant.scale, in_scale)
        return self._conv_forward(input, self.weight_fake_quant(self.weight), bias)