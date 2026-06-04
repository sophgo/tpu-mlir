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

void py_cuda::cudaConstantFillOp(top::ConstantFillOp op) {
  float value = op.getValue().convertToDouble();
  int num = module::getNumElements(op.getOutput());
  cuda::bmConstantFill(getCudaData(op.getOutput()), value, num);
}

void py_cuda::cudaConstantFillOp(tpu::ConstantFillOp op) {
  float value = op.getValue().convertToDouble();
  int num = module::getNumElements(op.getOutput());
  cuda::bmConstantFill(getCudaData(op.getOutput()), value, num);
}
