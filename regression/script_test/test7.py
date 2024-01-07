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
import onnx


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        a = self.conv1(x)
        b = torch.transpose(a, 1, 2)
        c = self.conv2(x)
        d = torch.transpose(c, 1, 2)
        e = b + d
        return e

torch.manual_seed(0)
x = torch.randn(4, 64, 100, 100).float()

inputs = {'x': x.numpy()}
np.savez("test7_input.npz", **inputs)

torch.onnx.export(
    Net(),
    (x),
    "test7.onnx",
    export_params=True,
    verbose=True,
    opset_version=13,  # export hardswish needed
    input_names=['x'])

