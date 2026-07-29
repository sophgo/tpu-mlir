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

void py_cuda::cudaMaxOp(top::MaxOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]);
  auto b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  cuda::bmMax(a, b, output, num);
}
