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
  auto num_elem = module::getNumElements(op.getOutput());
  auto out_type = module::getStorageType(op.getOutput());
  if (out_type.isa<FloatType>()) {
    auto shape = module::getShape(op.getInput());
    int64_t num_inner = 1;
    if (shape.size() > 1) {
      num_inner = std::accumulate(shape.begin() + 2, shape.end(), 1,
                                  std::multiplies<int64_t>());
    }
    int64_t num_outer = num_elem / num_inner;
    int64_t num_slope = module::getNumElements(op.getSlope());
    if (module::getStorageType(op.getInput()).isF32()) {
      auto input = getCudaData(op.getInput());
      auto slope = getCudaData(op.getSlope());
      auto output = getCudaData(op.getOutput());
      cuda::PReluF32(input, slope, output, num_outer, num_inner, num_slope);
    } else {
      auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto slope_f32 = newCudaData(op.getSlope(), cuda::DT_F32);
      auto out_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
      cuda::PReluF32(in_f32.get(), slope_f32.get(), out_f32.get(), num_outer, num_inner, num_slope);
      cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num_elem,
                        cuda::DT_F32, getCudaType(op.getOutput()));
    }
  } else if (module::isAsymmetric()) {
    UNREACHABLE_OP("Not Implemented", op);
  } else {
    auto shift = op.getRshift();
    int64_t shift_pos, multiplier_pos;
    if (is_cv18xx) {
      shift_pos = op.getRshiftPos().value();
      multiplier_pos = op.getMultiplierPos().value();
    }
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
      cuda::PReluInt8(input, slope, shift, output, num_outer, num_inner, num_slope);
    }
  }
}
