# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    def _no_grad_trunc_normal_(tensor, mean, std, a, b):

        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()

            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            tensor.clamp_(min=a, max=b)
            return tensor

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


#--------------------------------------#
# GELU Activation Function Implementation
# Use approximate mathematical formulas
#--------------------------------------#
class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


#---------------------------------------------------------------------------------#
# LayerNorm supports two forms: channels_last (default) or channels_first.
# channels_last corresponds to input with shape (batch_size, height, width, channels)
# channels_first corresponds to input with shape (batch_size, channels, height, width).
#---------------------------------------------------------------------------------#
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#--------------------------------------------------------------------------------------------------------------#
# ConvNeXt Block has two equivalent implementations:
#   (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#   (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
# Use (2) in the code because it is slightly faster in PyTorch.
#--------------------------------------------------------------------------------------------------------------#
class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        #--------------------------#
        # 7x7 layer-by-layer convolution
        #--------------------------#
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        #--------------------------#
        # Use fully connected layer instead of 1x1 convolution
        #--------------------------#
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = GELU()
        #--------------------------#
        # Use fully connected layer instead of 1x1 convolution
        #--------------------------#
        self.pwconv2 = nn.Linear(4 * dim, dim)
        #--------------------------#
        # Add scaling coefficient
        #--------------------------#
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(
            (dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        #--------------------------#
        # Add DropPath regularization
        #--------------------------#
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        #--------------------------#
        # 7x7 Layer-by-Layer Convolution
        #--------------------------#
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        #--------------------------#
        # Replace 1x1 convolution with fully connected layer
        #--------------------------#
        x = self.pwconv1(x)
        x = self.act(x)
        #--------------------------#
        # Replace 1x1 convolution with a fully connected layer
        #--------------------------#
        x = self.pwconv2(x)
        #--------------------------#
        # Add scaling coefficient
        #--------------------------#
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        #--------------------------#
        # Add DropPath regularization
        #--------------------------#
        x = input + self.drop_path(x)
        return x


#-----------------------------------------------------#
#   ConvNeXt
#   A PyTorch impl of : `A ConvNet for the 2020s`
#   https://arxiv.org/pdf/2201.03545.pdf
#-----------------------------------------------------#
class ConvNeXt(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 **kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        #--------------------------------------------------#
        #   bs, 3, 224, 224 -> bs, 96, 56, 56
        #--------------------------------------------------#
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        #--------------------------------------------------#
        # Define three downsampling stages
        # Downsampling using 2x2 convolution with 2x2 stride and kernel size
        #--------------------------------------------------#
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        #--------------------------------------------------#
        # Define different dropout rates based on depth.
        #--------------------------------------------------#
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        #--------------------------------------------------#
        # The entire ConvNeXt, excluding the Stem, has four Stages.
        # Each Stage is a stack of multiple ConvNeXt Blocks.
        #--------------------------------------------------#
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i],
                      drop_path=dp_rates[cur + j],
                      layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i != 0:
                outs.append(x)
        return outs


model_urls = {
    "convnext_tiny_1k":
    "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/convnext_tiny_1k_224_ema_no_jit.pth",
    "convnext_small_1k":
    "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/convnext_small_1k_224_ema_no_jit.pth",
}


#------------------------------------------------------#
# Tiny is about the same size as Cspdarknet-L
#------------------------------------------------------#
def ConvNeXt_Tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu",
                                                        model_dir="./model_data")
        model.load_state_dict(checkpoint, strict=False)
        print("Load weights from ", url.split('/')[-1])
    return model


#------------------------------------------------------#
# Tiny is approximately the same size as Cspdarknet-X
#------------------------------------------------------#
def ConvNeXt_Small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu",
                                                        model_dir="./model_data")
        model.load_state_dict(checkpoint, strict=False)
        print("Load weights from ", url.split('/')[-1])
    return model
