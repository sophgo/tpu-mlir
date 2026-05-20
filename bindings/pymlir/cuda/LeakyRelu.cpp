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

void py_cuda::cudaLeakyReluOp(top::LeakyReluOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float alpha = static_cast<float>(op.getAlpha().convertToDouble());
  cuda::bmLeakyRelu(input, output, num, alpha);
}

void py_cuda::cudaLeakyReluOp(tpu::LeakyReluOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float alpha = static_cast<float>(op.getAlpha()->convertToDouble());
  cuda::bmLeakyRelu(input, output, num, alpha);
}
