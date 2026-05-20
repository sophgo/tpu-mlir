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

// ==========================================================================
// Div
// ==========================================================================

void py_cuda::cudaDivOp(top::DivOp op) {
  auto inputs = op.getInputs();
  auto out = op.getOutput();
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(inputs[0], n0, c0, h0, w0, false);
  module::getNCHW(inputs[1], n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);

  cuda::div4DF32(getCudaData(inputs[0]), getCudaData(inputs[1]),
                  getCudaData(out), op.getDoRelu(), op.getIsReverse(),
                  n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void py_cuda::cudaDivOp(tpu::DivOp op) {
  auto inputs = op.getInputs();
  auto out = op.getOutput();
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(inputs[0], n0, c0, h0, w0, false);
  module::getNCHW(inputs[1], n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);

  cuda::div4DF32(getCudaData(inputs[0]), getCudaData(inputs[1]),
                  getCudaData(out), op.getDoRelu(), op.getIsReverse(),
                  n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

// ==========================================================================
// DivConst
// ==========================================================================

void py_cuda::cudaDivConstOp(top::DivConstOp op) {
  float const_v = op.getConstVal().convertToDouble();
  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w, false);

  cuda::divConst4DF32(getCudaData(op.getInput()), const_v,
                       getCudaData(op.getOutput()),
                       op.getDoRelu(), op.getIsReverse(), n, c, h, w);
}
