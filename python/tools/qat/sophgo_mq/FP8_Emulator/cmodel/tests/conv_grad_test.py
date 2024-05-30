#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import time
from mpemu.cmodel import simple

n = 64
c = 256
h = 28
w = 28
k = 512
r = 3
s = 3
stride = 2
pad = 2

a = torch.rand((n,c,h,w), dtype=torch.float32)
b = torch.rand((k,c,r,s), dtype=torch.float32)
bias = torch.rand((k), dtype=torch.float32)

a64 = a.to(dtype=torch.float64, copy=True)
b64 = b.to(dtype=torch.float64, copy=True)

a.requires_grad=True
b.requires_grad=True
a64.requires_grad=True
b64.requires_grad=True

ref_time = time.time()
z = torch.nn.functional.conv2d(a64, b64, bias, stride=(stride,stride), padding=(pad,pad), dilation=(1,1), groups=1) 
ref_time = time.time()-ref_time

simple_time = time.time()
z2 = simple.conv2d(a, b, bias, stride=(stride,stride), padding=(pad,pad), dilation=(1,1), groups=1) 
simple_time = time.time()-simple_time

#print("Forward Time : ref_time: {}, simple_time: {} ".format(ref_time, simple_time))
print('Forward: L2 distance output : ', torch.dist(z2.to(dtype=torch.float64, copy=True), z, 2).item())

ref_time = time.time()
(z[0, 0] + z[0,1]).sum().backward()
ref_time = time.time()-ref_time

simple_time = time.time()
(z2[0, 0] + z2[0,1]).sum().backward()
simple_time = time.time()-simple_time

#print("BackProp Time : ref_time: {}, simple_time: {}".format(ref_time, simple_time))
torch.set_printoptions(profile="full")
print('Backward: L2 distance input_grad: ', torch.dist(a.grad.to(dtype=torch.float64, copy=True), a64.grad, 2).item())
print('Backward: L2 distance weight_grad: ', torch.dist(b.grad.to(dtype=torch.float64, copy=True), b64.grad, 2).item())
