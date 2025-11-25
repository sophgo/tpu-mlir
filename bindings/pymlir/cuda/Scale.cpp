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

void py_cuda::cudaScaleOp(top::ScaleOp op) {
  void *input = getCudaData(op.getInput());
  void *scale = getCudaData(op.getScale());
  void *bias = getCudaData(op.getBias());
  bool relu = op.getDoRelu();
  void *output = getCudaData(op.getOutput());
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2, n3, c3, h3, w3;
  module::getNCHW(op.getInput(), n0, c0, h0, w0);
  module::getNCHW(op.getScale(), n1, c1, h1, w1);
  module::getNCHW(op.getBias(), n2, c2, h2, w2);
  module::getNCHW(op.getOutput(), n3, c3, h3, w3);

  cuda::scale4D(input, scale, bias, output, relu, n0, c0, h0, w0,
                n1, c1, h1, w1, n2, c2, h2, w2, n3, c3, h3, w3);
}
