//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaReluOp(top::ReluOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float relu_limit = static_cast<float>(op.getReluLimit().convertToDouble());
  cuda::bmRelu(input, output, num, relu_limit);
}

void py_cuda::cudaReluOp(tpu::ReluOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float relu_limit = static_cast<float>(op.getReluLimit().convertToDouble());
  cuda::bmRelu(input, output, num, relu_limit);
}
