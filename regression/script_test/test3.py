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


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = x.float() * 0.5
        a = self.conv1(y)
        b = self.relu1(a)
        c = self.conv2(b)
        d = self.relu2(c)
        return d


x = torch.randint(-10, 10, (4, 64, 8, 8), dtype=torch.int32)

inputs = {'x': x.numpy()}
np.savez("test3_input.npz", **inputs)
torch.jit.trace(Net().eval(), x).save("test3.pt")

# dataset
f = open("data3_list", "w")
for i in range(10):
    x = torch.randint(-10, 10, (4, 64, 8, 8), dtype=torch.int32)
    inputs = {'x': x.numpy()}
    name = "test3_in{}.npz".format(i)
    np.savez(name, **inputs)
    f.write(name + "\n")
f.close()
