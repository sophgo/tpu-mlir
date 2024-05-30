#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import cpp.fpemu as fpemu_cpp
import mpemu
import sys 

sys.path.append('../../mpemu')
# importing
from mpemu.qutils import fpemu_device_fn 


if torch.cuda.is_available():
    import cuda.fpemu as fpemu_cuda

def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        
        np.testing.assert_allclose(x, y, rtol=0.125, err_msg="Conflict : ", verbose=verbose)

def print_tensor(first, second, verbose, sparse):
    print('printing ...')
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        if sparse :
            x = x.cpu().to_dense().detach().numpy()
            y = y.cpu().to_dense().detach().numpy()
        else :
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]

def check_forward(variables, with_cuda, verbose, sparse):
    if with_cuda:
        if not torch.cuda.is_available():
            print('CUDA is not supported on this platform ... ')
        else :
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E5M2_RNE")
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E5M2_STOCHASTIC")
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E4M3_RNE")
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E4M3_STOCHASTIC")
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E3M4_RNE")
            #cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "E3M4_STOCHASTIC")
            cuda_values = fpemu_cuda.FPEmuOp.apply(variables.cuda(), "FP4_NEAREST")#, False, 1.0, True, 4)
            print('Forward: CUDA ... ', end='')
            print_tensor(variables, cuda_values, verbose, sparse)
    else :
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E5M2_RNE")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E5M2_STOCHASTIC")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E4M3_RNE")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E4M3_STOCHASTIC")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E3M4_RNE")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "E3M4_STOCHASTIC")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "FP4_NEAREST")#, False, 1.0, True, 4)
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "INT8")
        #cpp_values = fpemu_cpp.FPEmuOp.apply(variables, "INT4")
        cpp_values = fpemu_device_fn(variables, "INT4", inplace=False, scale=1.0)
        print('Forward: C++ ... ', end='')
        print_tensor(variables, cpp_values, verbose, sparse)

    if not verbose :
        print('Done.')
        print('Use the option --verbose to see results')
    else :
        print('Done.')
 

parser = argparse.ArgumentParser()
parser.add_argument('-sp', '--sparse', action='store_true')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')

options = parser.parse_args()
if options.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {'dtype': torch.float32,
          'device': device,
          'requires_grad': True}

'''
#FP4 values 
input = torch.tensor(np.array([[ 0.0000000,
                                 0.000244140625,
                                 0.0009766,
                                 0.0039063,
                                 0.0156250,
                                 0.0625000,
                                 0.2500000,
                                 1.0000000 ]]), dtype=torch.float32)
'''
input = torch.tensor(np.array([[ 57344.00     ,       -61440.0,        65504.0,         -500.0, 
                                 448.0        ,         -480.0,           30.0,          -31.0,
                                 26.0         ,           15.0, -7.6505613e-00,  5.9832452e-00,
                                 1.5625032e-02,  3.1725775e-02,  4.3268750e-02,  6.2655000e-02,
                                 1.9545313e-03,  3.9045625e-03,  5.8845638e-03,  7.8089750e-03,
                                -6.0151856e-02,  6.9784373e-03, -1.6634936e+03, -7.6505613e-04,  
                                 1.9545313e-03,  3.9045625e-03,  5.8845638e-03,  7.8089750e-03,
                                 1.5629950e-02,  1.5225650e-02, -3.1256500e-02,  9.3857500e-02,    
                                 7.9284608e-01,  2.8815269e-01, -1.1787039e-01,  6.1035156e-05,
                                -1.1787039e-06, -1.0749481e-01, -1.3085605e+00,  6.5981364e-01, 
                                -7.0325255e-02,  2.7448297e-01,  5.5694544e-01, -2.3220782e-01,
                                -5.9746221e-02,  15.23213444  , -0.00004323   ,  1.9767435e-04,
                                -1.2203161e+00,  2.9099861e-01, -7.9642259e-02,  1.3200364e+00,
                                -1.5196867e+00, -1.2530587e+00, -2.0159689e-03, -1.9767643e+00,
                                 6.0834163e-04,  7.8943473e-05,  7.8247029e-04, -6.4658634e-05,
                                -2.3020705e-06, -1.5630834e-05, -7.4762434e-07,  2.1336775e-06]]), dtype=torch.float32)
#'''
#input = torch.randn(4, 16)

if options.sparse :
    check_forward(input.to_sparse(), options.cuda, options.verbose, options.sparse)
else :
    check_forward(input, options.cuda, options.verbose, options.sparse)

