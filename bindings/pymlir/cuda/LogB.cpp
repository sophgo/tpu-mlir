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
#include <cmath>

void py_cuda::cudaLogBOp(top::LogBOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto num = module::getNumElements(op.getOutput());
  int base = op.getBase();
  float log_base_inv = 1.0f / logf((float)base);
  cuda::bmLogB(input, output, num, log_base_inv);
}
