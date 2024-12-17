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

void py_cuda::cudaPReluOp(tpu::PReluOp op) {
  bool is_cv18xx = module::isCV18xx();
  if (!module::isUniformQuantized(op.getOutput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto shift = op.getRshift();
  auto shift_pos = op.getRshiftPos().value();
  auto multiplier_pos = op.getMultiplierPos().value();
  auto num_slope = module::getNumElements(op.getSlope());
  auto in_shape = module::getShape(op.getInput());
  int64_t num_inner = 1;
  int64_t num_outer = 1;
  if (in_shape.size() > 1) {
    num_outer = std::accumulate(in_shape.begin(), in_shape.begin() + 2, 1,
                                std::multiplies<int64_t>());
    num_inner = std::accumulate(in_shape.begin() + 2, in_shape.end(), 1,
                                std::multiplies<int64_t>());
  } else {
    num_outer = in_shape[0];
    num_inner = 1;
  }
  void *input = getCudaData(op.getInput());
  void *slope = getCudaData(op.getSlope());
  void *output = getCudaData(op.getOutput());
  if (is_cv18xx) {
    cuda::cvPReluInt8(input, slope, output, num_outer, num_inner, num_slope,
                      multiplier_pos, shift_pos, shift);
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }
}
