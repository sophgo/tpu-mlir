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

void py_cuda::cudaDepackRawOp(top::DepackRawOp op) {
  auto in_shape = module::getShape(op.getInput());
  int n = in_shape[0];
  int ph = op.getPaddingH(), pw = op.getPaddingW();
  int ih = in_shape[2] - ph, iw = in_shape[3] - pw;
  float white = op.getWhiteLevel().convertToDouble();
  float black = op.getBlackLevel().convertToDouble();

  auto order = module::getI64Array(op.getChannelOrder());
  int c0 = order->at(0), c1 = order->at(1);
  int c2 = order->at(2), c3 = order->at(3);

  cuda::bmDepackRaw(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     n, ih, iw, ph, pw, white, black, c0, c1, c2, c3);
}

void py_cuda::cudaDepackRawOp(tpu::DepackRawOp op) {
  auto in_shape = module::getShape(op.getInput());
  int n = in_shape[0];
  int ph = op.getPaddingH(), pw = op.getPaddingW();
  int ih = in_shape[2] - ph, iw = in_shape[3] - pw;
  float white = op.getWhiteLevel().convertToDouble();
  float black = op.getBlackLevel().convertToDouble();

  auto order = module::getI64Array(op.getChannelOrder());
  int c0 = order->at(0), c1 = order->at(1);
  int c2 = order->at(2), c3 = order->at(3);

  cuda::bmDepackRaw(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     n, ih, iw, ph, pw, white, black, c0, c1, c2, c3);
}
