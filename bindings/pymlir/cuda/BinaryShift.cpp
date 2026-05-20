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
// BinaryShift:  output = (a op b) >> shift  (Add/Sub/Mul)
// ==========================================================================

void py_cuda::cudaBinaryShiftOp(top::BinaryShiftOp op) {
  auto a = op.getInput1(), b = op.getInput2(), out = op.getOutput();
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(a, n0, c0, h0, w0, false);
  module::getNCHW(b, n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);

  auto mode = op.getMode().str();
  bool rev = op.getIsReverse();

  if (mode == "Add") {
    cuda::add4DF32(getCudaData(a), 1.0f, getCudaData(b), 1.0f,
                    getCudaData(out), false,
                    n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (mode == "Sub") {
    cuda::sub4DF32(getCudaData(a), getCudaData(b), getCudaData(out),
                    false, rev,
                    n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else {
    // Mul
    cuda::mul4DF32(getCudaData(a), getCudaData(b), getCudaData(out), false,
                    n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  }
}

// ==========================================================================
// BinaryConstShift: output = (input op scale) >> shift
// ==========================================================================

void py_cuda::cudaBinaryConstShiftOp(top::BinaryConstShiftOp op) {
  auto in = op.getInput(), out = op.getOutput();
  float scale = (float)op.getScale();
  int64_t n, c, h, w;
  module::getNCHW(out, n, c, h, w, false);

  auto mode = op.getMode().str();
  bool rev = op.getIsReverse();

  if (mode == "Add") {
    cuda::addConst4DF32(getCudaData(in), scale, getCudaData(out), false,
                         n, c, h, w);
  } else if (mode == "Sub") {
    cuda::subConst4DF32(getCudaData(in), scale, getCudaData(out),
                         false, rev, n, c, h, w);
  } else {
    cuda::mulConst4DF32(getCudaData(in), scale, getCudaData(out), false,
                         n, c, h, w);
  }
}
