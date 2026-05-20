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

void py_cuda::cudaInterpOp(top::InterpOp op) {
  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(op.getInput(), n, c, ih, iw, false);
  module::getNCHW(op.getOutput(), n, c, oh, ow, false);

  auto mode = op.getMode().str();
  auto coord_mode = op.getCoordMode().str();
  bool align_corners = (coord_mode == "align_corners");
  bool half_pixel = (coord_mode == "half_pixel" ||
                     coord_mode == "pytorch_half_pixel");

  if (mode == "linear") {
    cuda::bmInterpBilinear(getCudaData(op.getInput()),
                            getCudaData(op.getOutput()),
                            n, c, ih, iw, oh, ow, align_corners, half_pixel);
  } else {
    cuda::bmInterpNearest(getCudaData(op.getInput()),
                           getCudaData(op.getOutput()),
                           n, c, ih, iw, oh, ow);
  }
}

void py_cuda::cudaInterpOp(tpu::InterpOp op) {
  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(op.getInput(), n, c, ih, iw, false);
  module::getNCHW(op.getOutput(), n, c, oh, ow, false);

  auto mode = op.getMode();
  auto coord_mode = op.getCoordMode();
  using CM = tpu::ResizeCoordMode;
  bool align_corners = (coord_mode == CM::align_corners);
  bool half_pixel = (coord_mode == CM::half_pixel ||
                     coord_mode == CM::pytorch_half_pixel);

  if (mode == tpu::ResizeMode::linear) {
    cuda::bmInterpBilinear(getCudaData(op.getInput()),
                            getCudaData(op.getOutput()),
                            n, c, ih, iw, oh, ow, align_corners, half_pixel);
  } else {
    cuda::bmInterpNearest(getCudaData(op.getInput()),
                           getCudaData(op.getOutput()),
                           n, c, ih, iw, oh, ow);
  }
}
