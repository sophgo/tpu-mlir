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

void py_cuda::cudaDivConstOp(top::DivConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  float const_val = static_cast<float>(op.getConstVal().convertToDouble());
  bool is_reverse = op.getIsReverse();
  bool do_relu = op.getDoRelu();
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w, false);
  cuda::divConst4DF32(input, output, const_val, is_reverse, do_relu, n, c, h, w);
}
