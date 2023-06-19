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
from model import BayerNet

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
    y = BayerNet().forward(img)
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
