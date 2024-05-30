/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fpemu_cuda_forward(
    torch::Tensor input,
    std::string mode,
    int size,
    bool inplace,
    float scale,
    bool block_norm,
    int block_size);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fpemu_forward(
    torch::Tensor input,
    std::string mode,
    int size,
    bool inplace,
    float scale,
    bool block_norm,
    int block_size) {
  CHECK_INPUT(input);
  return fpemu_cuda_forward(input, mode, size, inplace, scale, block_norm, block_size); 
}

std::vector<torch::Tensor> fpemu_backward(
    torch::Tensor grad,
    std::string mode,
    int size,
    bool inplace,
    float scale,
    bool block_norm,
    int block_size) {
  CHECK_INPUT(grad);
  return fpemu_cuda_forward(grad, mode, size, inplace, scale, block_norm, block_size); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fpemu_forward, "FPEmu forward (CUDA)");
  m.def("backward", &fpemu_backward, "FPEmu backward (CUDA)");
}
