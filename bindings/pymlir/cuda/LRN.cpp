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

void py_cuda::cudaLRNOp(top::LRNOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w, false);
  int size = op.getSize();
  float alpha = op.getAlpha().convertToDouble();
  float beta  = op.getBeta().convertToDouble();
  float bias  = op.getBias().convertToDouble();
  cuda::bmLRN(getCudaData(op.getInput()), getCudaData(op.getOutput()),
              n, c, h, w, size, alpha, beta, bias);
}

void py_cuda::cudaLRNOp(tpu::LRNOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w, false);
  int size = op.getSize();
  float alpha = op.getAlpha().convertToDouble();
  float beta  = op.getBeta().convertToDouble();
  float bias  = op.getBias().convertToDouble();
  cuda::bmLRN(getCudaData(op.getInput()), getCudaData(op.getOutput()),
              n, c, h, w, size, alpha, beta, bias);
}
