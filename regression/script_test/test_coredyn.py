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
        self.filter0 = torch.randn((512, 128))
        self.filter1 = torch.randn((512, 128))

    def forward(self, x):
        a = torch.matmul(x, self.filter0)
        b = torch.matmul(x, self.filter1)
        c = a + b
        d = torch.topk(c, 1)
        return d


torch.manual_seed(0)
x = torch.randn(4, 512).float()

inputs = {'x': x.numpy()}
np.savez("test_coredyn_input.npz", **inputs)

torch.jit.trace(Net().eval(), x).save("test_coredyn.pt")
