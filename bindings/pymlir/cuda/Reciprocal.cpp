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

void py_cuda::cudaReciprocalOp(top::ReciprocalOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float const_val = static_cast<float>(op.getConstVal().convertToDouble());
  bool do_relu = op.getDoRelu();
  float relu_limit = static_cast<float>(op.getReluLimit().convertToDouble());
  cuda::bmReciprocal(input, output, num, const_val, do_relu, relu_limit);
}

void py_cuda::cudaReciprocalOp(tpu::ReciprocalOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  float const_val = static_cast<float>(op.getConstVal().convertToDouble());
  bool do_relu = op.getDoRelu();
  float relu_limit = static_cast<float>(op.getReluLimit().convertToDouble());
  cuda::bmReciprocal(input, output, num, const_val, do_relu, relu_limit);
}
