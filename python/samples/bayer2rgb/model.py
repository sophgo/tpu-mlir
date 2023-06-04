#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tools.model_runner import model_inference
from PIL import Image
from pathlib import Path

# support BayerBG to rgb


class BayerNet(torch.nn.Module):

    def __init__(self):
        super(BayerNet, self).__init__()
        self.kernels = torch.tensor([
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],
            ],
            [
                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],
            ],
            [
                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],
            ],
            [
                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ],
        ]).view(5, 1, 3, 3)
        self.kernel0 = torch.tensor([
            [0, 0, 1.0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [1.0, 0, 0, 0, 0],
        ]).view(3, 5, 1, 1)
        self.kernel1 = torch.tensor([
            [0, 0, 0, 0, 1.0],
            [1.0, 0, 0, 0, 0],
            [0, 0, 0, 1.0, 0],
        ]).view(3, 5, 1, 1)
        self.kernel2 = torch.tensor([
            [0, 0, 0, 1.0, 0],
            [1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1.0],
        ]).view(3, 5, 1, 1)
        self.kernel3 = torch.tensor([
            [1.0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0, 1.0, 0, 0],
        ]).view(3, 5, 1, 1)
        self.kernels2 = torch.cat((self.kernel0, self.kernel1, self.kernel2, self.kernel3),
                                  0).view(12, 5, 1, 1)
        self.reflect_pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def unshuffle_dcr(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, 4 * c, h // 2, w // 2)
        return x

    def shuffle_dcr(self, x):
        n, c, h, w = x.shape
        x = x.view(n, 2, 2, c // 4, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // 4, h * 2, w * 2)
        return x

    def forward(self, x):
        # x: Bx1xHxW, bayer image
        # y: Bx3xHxW, rgb image
        x_pad = self.reflect_pad(x)  # (Bx1x(1+H+1)x(1+W+1))
        x_conv = nn.functional.conv2d(x_pad, self.kernels, stride=1)  #[B, 5, H, W]
        x_unshuffle = self.unshuffle_dcr(x_conv)  #[B, 4*5, H/2, W/2]
        a = x_unshuffle[:, :5, :, :]
        b = x_unshuffle[:, 5:10, :, :]
        c = x_unshuffle[:, 10:15, :, :]
        d = x_unshuffle[:, 15:20, :, :]
        a_conv = nn.functional.conv2d(a, self.kernel0, stride=1)
        b_conv = nn.functional.conv2d(b, self.kernel1, stride=1)
        c_conv = nn.functional.conv2d(c, self.kernel2, stride=1)
        d_conv = nn.functional.conv2d(d, self.kernel3, stride=1)
        e = torch.cat((a_conv,b_conv,c_conv,d_conv), 1)
        y = self.shuffle_dcr(e)  #[B, 3, H, W]
        return y


x = torch.randint(0, 256, (1, 1, 1024, 1024), dtype=torch.float32)

inputs = {'x': x.numpy()}
np.savez("input.npz", **inputs)
torch.jit.trace(BayerNet().eval(), x).save("bayer2rgb.pt")

