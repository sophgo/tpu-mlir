#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import fuse 

m=1 #356320 #356305
n=120 #256
k=576 #128 #2020

# Start here
a = torch.rand((5, 107, 1024), dtype=torch.float32)
b = torch.rand((1024, 1024), dtype=torch.float32)
c = torch.zeros((m, n), dtype=torch.float32)

a64 = a.to(dtype=torch.float64, copy=True)
b64 = b.to(dtype=torch.float64, copy=True)
c64 = c.to(dtype=torch.float64, copy=True)

z = torch.matmul(a64, b64)
z3 = torch.matmul(a, b)
for a1 in a :
    print("-->", a1.size())

#z2l = tuple([fuse.tmul.matmul(a1, b) for a1 in a])
#z2 = torch.stack(z2l)
#z2 = torch.stack(tuple([fuse.tmul.matmul(a1, b) for a1 in a]))
z2 = fuse.tmul.matmul(a, b)

#print (z3.size(), z2.size())

#print("torch :", z3.size(), z3)
#print("Ours : ", z2.size(), z2)
print('32b: L2 distance : ', torch.dist(z, z3.to(dtype=torch.float64, copy=True), 2))
print('ours: L2 distance : ', torch.dist(z, z2.to(dtype=torch.float64, copy=True), 2))

