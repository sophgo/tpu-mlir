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

void py_cuda::cudaRequantIntAxisOp(tpu::RequantIntAxisOp op) {
  // void *input = getCudaData(op.getInput());
  // void *quant = getCudaData(op.getQuant());
  // void *output = getCudaData(op.getOutput());

  // cudaMulShift(input, , m, s, num, getCudnnType(op.getInput()));
}
