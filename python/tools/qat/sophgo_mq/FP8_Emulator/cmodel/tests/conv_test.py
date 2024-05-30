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

n = 2
c = 64
h = 28
w = 28
k = 64
r = 3 
s = 3

input = torch.rand((n,c,h,w), dtype=torch.float32)
weight = torch.rand((k,c,r,s), dtype=torch.float32)
bias = torch.rand((k), dtype=torch.float32)

output_ref = torch.nn.functional.conv2d(input, weight, bias, stride=(2,2), padding=(2,2), dilation=(1,1), groups=1)
output_simple = simple.conv2d(input, weight, bias, stride=(2,2), padding=(2,2), dilation=(1,1), groups=1)

torch.set_printoptions(profile="full")

print('Forward : L2 distance (simple) : ', torch.dist(output_ref, output_simple, 2).item())
