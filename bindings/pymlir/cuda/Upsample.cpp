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

void py_cuda::cudaUpsampleOp(tpu::UpsampleOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  int64_t n, c, ih, iw;
  module::getNCHW(op.getInput(), n, c, ih, iw);
  int sw = op.getScaleW();
  int sh = op.getScaleH();
  cuda::upsample4D(input, output, n, c, ih, iw, sh, sw,
                   module::getDtypeSize(op.getInput()));
}
