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

# test MatMul+TopK with 2 cores


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.filter = torch.randn((512, 128))

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        b = torch.topk(a, 1)
        return b


torch.manual_seed(0)
x = torch.randn(4, 512).float()

inputs = {'x': x.numpy()}
np.savez("test7_input.npz", **inputs)

torch.onnx.export(Net(), (x),
                  "test7.onnx",
                  export_params=True,
                  verbose=True,
                  opset_version=13,
                  input_names=['x'])
