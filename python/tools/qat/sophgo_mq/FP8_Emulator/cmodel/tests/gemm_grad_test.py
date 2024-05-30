#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
from mpemu.cmodel import simple

m=1024 #356320 #356305
n=1024 #120 #256
k=1024 #576 #128 #2020

# Start here
a = torch.rand((m, k), dtype=torch.float32)
b = torch.rand((n, k), dtype=torch.float32)
c = torch.zeros((m, n), dtype=torch.float32)

a64 = a.to(dtype=torch.float64, copy=True)
b64 = b.to(dtype=torch.float64, copy=True)
c64 = c.to(dtype=torch.float64, copy=True)

a.requires_grad=True
b.requires_grad=True
c.requires_grad=True
a64.requires_grad=True
b64.requires_grad=True
c64.requires_grad=True

z = torch.addmm(c64, a64, b64.t())
z2 = simple.addmm(c, a, b.t())

print('Forward :L2 distance output: ', torch.dist(z2.to(dtype=torch.float64, copy=True), z, 2).item())

(z2[0, 0] + z2[0,1]).sum().backward()
(z[0, 0] + z[0,1]).sum().backward()

print('Backward : L2 distance a_grad: ', torch.dist(a.grad.to(dtype=torch.float64, copy=True), a64.grad, 2).item())
print('Backward : L2 distance b_grad: ', torch.dist(b.grad.to(dtype=torch.float64, copy=True), b64.grad, 2).item())
print('Backward : L2 distance c_grad: ', torch.dist(c.grad.to(dtype=torch.float64, copy=True), c64.grad, 2).item())
