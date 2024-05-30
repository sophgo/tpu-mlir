#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpemu.cmodel import simple

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(576, 120)

    def forward(self, x):
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net1 = Net()
print(net)
input = torch.randn((1, 576), dtype=torch.float32, device="cpu")
input_new = input.to(dtype=torch.float32, copy=True)

output = net(input)
#print("fc output:", output.size(), output)

target = torch.randn(120, dtype=torch.float32, device="cpu")  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
net.zero_grad()     # zeroes the gradient buffers of all parameters
#loss.backward(retain_graph=True)
loss.backward()
print("fc weight grads:", net.fc.weight.grad)


torch.addmm_back = torch.addmm
torch.matmul_back = torch.matmul
torch.addmm = simple.addmm
torch.matmul = simple.matmul

output1 = net1(input_new)
#print("fc output:", output1.size(), output1)

loss1 = criterion(output1, target)
net1.zero_grad()     # zeroes the gradient buffers of all parameters
loss1.backward()
print("fc weight grads:", net1.fc.weight.grad)

print('Linear wtgrads L2 distance : ', torch.dist(net.fc.weight.grad, net1.fc.weight.grad, 2))
