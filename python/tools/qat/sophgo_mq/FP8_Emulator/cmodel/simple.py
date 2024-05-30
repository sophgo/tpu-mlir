#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
from torch import nn
from torch.autograd import Function
import sys

import simple_gemm_dev
import simple_conv2d_dev
# backup the original torch functions 
fallback_addmm = torch.addmm
fallback_matmul = torch.matmul
fallback_mm = torch.mm

def is_transposed(input):
    if input.is_contiguous():
        return input, False
    elif input.t().is_contiguous():
        return input.t(), True
    else:
        return input.contiguous(), False

def addmm(input, mat1, mat2, beta=1.0, alpha=1.0, out=None):
    if input.dtype == torch.float32 and mat1.dtype == torch.float32 and \
                  mat1.dim() == 2 and mat2.dim() == 2 and mat1.size(1) == mat2.size(0):
        if out:
            output = out
        else:
            output = torch.zeros([mat1.size(0), mat2.size(1)])
        a_mat, a_trans = is_transposed(mat1)
        b_mat, b_trans = is_transposed(mat2)
        output = SimpleAddmm.apply(output, input, a_mat, b_mat, alpha, beta, a_trans, b_trans)
        ret = output
    else:
        warnings.warn('simple.addmm does not support the input dimensions - input :{}, mat1: {}, mat2: {}, falling back to torch.addmm'.format(
                              input.size(), mat1.size(), mat2.size()))
        ret = fallback_addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)
    return ret

def matmul(input, other, out=None):
    if input.dtype == torch.float32 and other.dtype == torch.float32 and \
                input.dim() == 2 and other.dim() == 2 and input.size(1) == other.size(0):
        if out:
            output = out
        else:
            output = torch.zeros([input.size(0), other.size(1)])
        a_mat, a_trans = is_transposed(input)
        b_mat, b_trans = is_transposed(other)
        output = SimpleMatmul.apply(output, a_mat, b_mat, 1.0, a_trans, b_trans)
        return output
    # Batch MatMul implementation
    elif input.dtype == torch.float32 and other.dtype == torch.float32 and \
                input.dim() == 3 and other.dim() == 2 and input.size(2) == other.size(0):
        if out:
            output = out
        else:
            output = torch.zeros([input.size(0), input.size(1), other.size(1)])
        a_mat, a_trans = is_transposed(input)
        b_mat, b_trans = is_transposed(other)
        output = torch.stack(tuple([SimpleMatmul.apply(out1, a_mat1, b_mat, 1.0, a_trans, b_trans) \
                                  for a_mat1, out1 in zip(a_mat, output)]))
        return output
    else:
        warnings.warn('simple.matmul does not support the input dimensions - input :{}, other: {}, falling back to torch.matmul'.format(
                              input.size(), other.size()))
        return fallback_matmul(input, other, out=out)

def mm(input, mat2, out=None):
    if input.dtype == torch.float32 and mat2.dtype == torch.float32 and \
        input.dim() == 2 and mat2.dim() == 2 and input.size(1) == mat2.size(0):
        if out:
            output = out
        else:
            output = torch.zeros([input.size(0), mat2.size(1)])
        a_mat, a_trans = is_transposed(input)
        b_mat, b_trans = is_transposed(mat2)
        output = SimpleMatmul.apply(output, a_mat, b_mat, 1.0, a_trans, b_trans)
        return output
    else:
        warnings.warn('simple.mm does not support the input dimensions - input :{}, mat2: {}, falling back to torch.mm'.format(
                              input.size(), mat2.size()))
        return fallback_mm(input, mat2, out=out)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    N = input.size()[0]
    C = input.size()[1]
    H = input.size()[2]
    W = input.size()[3]
    K = weight.size()[0]
    C1 = weight.size()[1]
    R = weight.size()[2]
    S = weight.size()[3]
  
    if dilation[0] > 1:
        sys.exit("ERROR: simple_conv2d does not support dilated convolutions.")
    if padding[0] != padding[1]:
        sys.exit("ERROR: simple_conv2d does not support non-uniform padding; pad_h must be equal to pad_w.")
    if groups > 1:
        sys.exit("ERROR: simple_conv2d does not support grouped convolutions; set groups to 1.")

    H_out = ((H + (2*padding[0]) - dilation[0] * (R-1) -1)/stride[0]) + 1
    W_out = ((W + (2*padding[1]) - dilation[1] * (S-1) -1)/stride[1]) + 1
    output = torch.empty([N, K, int(H_out), int(W_out)])
    output = SimpleConv2dFunction.apply(output, input, weight, bias, stride, padding, dilation, groups)
    return output

class SimpleAddmm(Function):
    @staticmethod
    def forward(ctx, output, input, mat1, mat2, alpha, beta, a_trans, b_trans):
        ctx.save_for_backward(mat1, mat2)
        ctx.a_trans = a_trans
        ctx.b_trans = b_trans
        ctx.alpha = alpha

        simple_gemm_dev.gemm(output, mat1, mat2, alpha, a_trans, b_trans)
        output += beta * input;
        ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mat1, mat2 = ctx.saved_tensors

        alpha = ctx.alpha
        a_trans = ctx.a_trans
        b_trans = ctx.b_trans

        grad_mat1 = torch.zeros_like(mat1)
        grad_mat2 = torch.zeros_like(mat2)
        grad_out, out_trans = is_transposed(grad_output)

        if a_trans:
            simple_gemm_dev.gemm(grad_mat1, mat2, grad_out, alpha, b_trans, not out_trans)
        else:
            simple_gemm_dev.gemm(grad_mat1, grad_out, mat2, alpha, out_trans, not b_trans)

        if b_trans:
            simple_gemm_dev.gemm(grad_mat2, grad_out, mat1, alpha, not out_trans, a_trans)
        else:
            simple_gemm_dev.gemm(grad_mat2, mat1, grad_out, alpha, not a_trans, out_trans)

        return (grad_output, grad_output, grad_mat1, grad_mat2, None, None, None, None)

class SimpleMatmul(Function):
    @staticmethod
    def forward(ctx, output, mat1, mat2, alpha, a_trans, b_trans):
        ctx.save_for_backward(mat1, mat2)
        ctx.a_trans = a_trans
        ctx.b_trans = b_trans
        ctx.alpha = alpha

        simple_gemm_dev.gemm(output, mat1, mat2, alpha, a_trans, b_trans)
        ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mat1, mat2 = ctx.saved_tensors
        alpha = ctx.alpha
        a_trans = ctx.a_trans
        b_trans = ctx.b_trans

        grad_mat1 = torch.empty_like(mat1)
        grad_mat2 = torch.empty_like(mat2)
        grad_out, out_trans = is_transposed(grad_output)

        if a_trans:
            simple_gemm_dev.gemm(grad_mat1, mat2, grad_out, alpha, b_trans, not out_trans)
        else:
            simple_gemm_dev.gemm(grad_mat1, grad_out, mat2, alpha, out_trans, not b_trans)

        if b_trans:
            simple_gemm_dev.gemm(grad_mat2, grad_out, mat1, alpha, not out_trans, a_trans)
        else:
            simple_gemm_dev.gemm(grad_mat2, mat1, grad_out, alpha, not a_trans, out_trans)
        return (grad_output, grad_mat1, grad_mat2, None, None, None)


class SimpleConv2dFunction(Function):
    @staticmethod
    def forward(ctx, output, inputs, weights, bias, stride, padding, dilation, groups):
        #print("### conv2d fwd called input size: ", inputs.size(), weights.size(), stride, padding, dilation, groups)
        ctx.save_for_backward(inputs, weights)#, bias)
        ctx.stride = stride#[0]
        ctx.padding = padding#[0]
        ctx.dilation = dilation#[0]
        ctx.groups = groups

        if bias is None:
           bias_fw = torch.zeros(output.size()[1])
        else :
           bias_fw = bias

        simple_conv2d_dev.conv2d_fp(output, inputs, weights, bias_fw, stride[0], padding[0], dilation[0], groups)
        ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #inputs, weights, bias = ctx.saved_tensors
        inputs, weights = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        #print("### conv2d bwd called input size: ", inputs.size(), weights.size(), stride, padding, dilation, groups)
        grad_inp = torch.zeros_like(inputs)
        grad_wts = torch.zeros_like(weights)

        simple_conv2d_dev.conv2d_bp(grad_inp, grad_output, weights, stride[0], padding[0], dilation[0], groups)
        simple_conv2d_dev.conv2d_wu(grad_wts, grad_output, inputs,  stride[0], padding[0], dilation[0], groups)
        return (grad_output, grad_inp, grad_wts, None, None, None, None, None)
