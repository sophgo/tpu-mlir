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

void py_cuda::cudaGridSamplerOp(top::GridSamplerOp op) {
  auto in_shape = module::getShape(op.getInput());
  int n = in_shape[0], c = in_shape[1], h = in_shape[2], w = in_shape[3];
  auto out_shape = module::getShape(op.getOutput());
  int oh = out_shape[2], ow = out_shape[3];

  cuda::bmGridSampler(getCudaData(op.getInput()), getCudaData(op.getGrid()),
                       getCudaData(op.getOutput()),
                       n, c, h, w, oh, ow,
                       op.getMode(), op.getPaddingMode(), op.getAlignCorners());
}

void py_cuda::cudaGridSamplerOp(tpu::GridSamplerOp op) {
  auto in_shape = module::getShape(op.getInput());
  int n = in_shape[0], c = in_shape[1], h = in_shape[2], w = in_shape[3];
  auto out_shape = module::getShape(op.getOutput());
  int oh = out_shape[2], ow = out_shape[3];

  cuda::bmGridSampler(getCudaData(op.getInput()), getCudaData(op.getGrid()),
                       getCudaData(op.getOutput()),
                       n, c, h, w, oh, ow,
                       op.getMode(), op.getPaddingMode(), op.getAlignCorners());
}
