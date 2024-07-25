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

# test encrypt and decrypt bmodel


class NetA(torch.nn.Module):

    def __init__(self):
        super(NetA, self).__init__()
        self.filter = torch.randn((512, 128))
        self.act = torch.nn.SiLU()

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        b = self.act(a)
        return b


class NetB(torch.nn.Module):

    def __init__(self):
        super(NetB, self).__init__()
        self.filter = torch.randn((512, 128))
        self.act = torch.nn.ReLU()

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        b = self.act(a)
        return b


torch.manual_seed(0)
x = torch.randn(4, 512).float()

inputs = {'x': x.numpy()}
np.savez("test_encrypt_input.npz", **inputs)

torch.onnx.export(NetA(), (x),
                  "test_encrypt_a.onnx",
                  export_params=True,
                  verbose=True,
                  opset_version=13,
                  input_names=['x'])

torch.onnx.export(NetB(), (x),
                  "test_encrypt_b.onnx",
                  export_params=True,
                  verbose=True,
                  opset_version=13,
                  input_names=['x'])
