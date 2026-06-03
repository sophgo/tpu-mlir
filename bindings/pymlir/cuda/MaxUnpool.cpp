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

void py_cuda::cudaMaxUnpoolOp(top::MaxUnpoolOp op) {
  auto input = getCudaData(op.getInput());
  auto mask = getCudaData(op.getMask());
  auto output = getCudaData(op.getOutput());
  int64_t n, c, oh, ow, out_h, out_w;
  module::getNCHW(op.getInput(), n, c, oh, ow);
  module::getNCHW(op.getOutput(), n, c, out_h, out_w);
  cuda::maxUnpool(input, mask, output, n, c, oh, ow,
                  op.getScaleH(), op.getScaleW(), out_h, out_w);
}

void py_cuda::cudaMaxUnpoolOp(tpu::MaxUnpoolOp op) {
  auto input = getCudaData(op.getInput());
  auto mask = getCudaData(op.getMask());
  auto output = getCudaData(op.getOutput());
  int64_t n, c, oh, ow, out_h, out_w;
  module::getNCHW(op.getInput(), n, c, oh, ow);
  module::getNCHW(op.getOutput(), n, c, out_h, out_w);
  cuda::maxUnpool(input, mask, output, n, c, oh, ow,
                  op.getScaleH(), op.getScaleW(), out_h, out_w);
}
