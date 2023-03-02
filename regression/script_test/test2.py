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
        self.conv = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x, y, z):
        a = self.conv(x)
        b = y + z
        c = a > b
        return c, b


x = torch.randn(4, 64, 8, 8).float()
y = torch.randn(8, 8).float()
z = torch.randn(4, 64, 8, 8).float()

inputs = {'x': x.numpy(), 'y': y.numpy(), 'z': z.numpy()}
np.savez("test_input.npz", **inputs)

torch.onnx.export(
    Net(),
    (x, y, z),
    "test.onnx",
    export_params=True,
    verbose=True,
    opset_version=13,  # export hardswish needed
    input_names=['x', 'y', 'z'])

# dataset
f = open("data_list", "w")
for i in range(10):
    x = torch.randn(4, 64, 8, 8).float()
    y = torch.randn(8, 8).float()
    z = torch.randn(4, 64, 8, 8).float()
    inputs = {'x': x.numpy(), 'y': y.numpy(), 'z': z.numpy()}
    name = "test_in{}.npz".format(i)
    np.savez(name, **inputs)
    f.write(name + "\n")
f.close()
