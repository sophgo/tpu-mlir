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


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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
torch.jit.trace(Net().eval(), x).save("bayer2rgb.pt")


def BayerBGGR2RGB(img: torch.Tensor) -> torch.Tensor:
    # BGGR -> RGB
    assert img.shape[1] == 1
    img_t = np.pad(
        img.numpy().astype(np.float32),
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="reflect",
    )
    # after padding
    B, G0, G1, R = 3, 2, 1, 0
    _pixel = lambda img, x, y: {
        B: [
            img[x][y],
            (img[x][y - 1] + img[x - 1][y] + img[x + 1][y] + img[x][y + 1]) / 4,
            (
                img[x - 1][y - 1]
                + img[x + 1][y - 1]
                + img[x - 1][y + 1]
                + img[x + 1][y + 1]
            )
            / 4,
        ],
        G0: [
            (img[x - 1][y] + img[x + 1][y]) / 2,
            img[x][y],
            (img[x][y - 1] + img[x][y + 1]) / 2,
        ],
        G1: [
            (img[x][y - 1] + img[x][y + 1]) / 2,
            img[x][y],
            (img[x - 1][y] + img[x + 1][y]) / 2,
        ],
        R: [
            (
                img[x - 1][y - 1]
                + img[x + 1][y - 1]
                + img[x - 1][y + 1]
                + img[x + 1][y + 1]
            )
            / 4,
            (img[x][y - 1] + img[x - 1][y] + img[x + 1][y] + img[x][y + 1]) / 4,
            img[x][y],
        ],
    }[x % 2 + (y % 2) * 2]
    n, c, h, w = img.shape
    res = np.zeros((n, 3, h, w), dtype=np.float32)
    for z in range(n):
        for x in range(0, w):
            for y in range(0, h):
                p = _pixel(img_t[z, 0, :, :], x + 1, y + 1)
                p.reverse()
                res[z, :, x, y] = p
    return torch.tensor(res)


def __check(img):
    y = Net().forward(img)
    z = BayerBGGR2RGB(img)
    return torch.all(y == z)


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Inference Keras Yolo3 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--input", type=str, required=True, help="Input image or dir for testing")
    parser.add_argument("--output", type=str, required=True, help="Output image or dir after detection")
    # yapf: enable
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert Path(args.input).is_file()
    bayerbggr_image = Image.open(args.input)
    assert len(bayerbggr_image.size) == 2
    h, w = 1024, 1024
    bayerbggr_image = bayerbggr_image.resize((h, w), Image.Resampling.BICUBIC)
    img = np.asarray(bayerbggr_image).reshape([1, 1, *bayerbggr_image.size])
    fname = args.model
    assert fname.endswith(".bmodel")
    output = model_inference({"x.1": img}, fname)
    assert len(output) == 1
    out_img = list(output.values())[0][0].transpose((1, 2, 0))
    out = Image.fromarray(out_img.astype(np.uint8), mode="RGB")
    out.save(args.output)


if __name__ == '__main__':
    main()
