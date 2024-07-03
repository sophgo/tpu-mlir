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

void py_cuda::cudaLutOp(tpu::LutOp op) {
  void *input = getCudaData(op.getInput());
  void *lut = getCudaData(op.getTable());
  void *output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getInput());
  cuda::lut256(input, lut, output, num, getCudaType(op.getInput()),
               getCudaType(op.getTable()));
}
