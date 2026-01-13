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
#include "tpu_mlir/Support/MathUtils.h"


void py_cuda::cudaRequantFpOp(tpu::RequantFpOp op) {
  float scale_v = op.getScale().convertToDouble();
  float offset_v = op.getOffset().convertToDouble();
  // auto round_mode = round_mode_convert(op.getRoundMode());
  auto mode = op.getQuantMode();
  cuda::rounding_mode_t rmode = cuda::RD_HALF_TO_EVEN;

  auto requant = [&op, mode, scale_v, offset_v, rmode, this](void * in_f32) {
    auto out_8 = getCudaData(op.getOutput());
    switch(mode) {
      case tpu::RequantMode::MultiplierShift:
        if (!module::isUniformQuantized(op.getOutput())) {
          UNREACHABLE_OP("output type wrong", op);
        }
        cuda::mulShiftFloat(in_f32, out_8, scale_v, offset_v, rmode, module::getNumElements(op.getInput()), getCudaType(op.getOutput()));
        break;
      case tpu::RequantMode::OnlyScale:
        if (!module::getStorageType(op.getOutput()).isFloat8E4M3FN())
          UNREACHABLE_OP("output type wrong", op);
        cuda::quantF8(in_f32, out_8, scale_v, module::getNumElements(op.getInput()));
        break;
      default:
        UNREACHABLE_OP("output type wrong", op);
    }
  };

  if (module::getStorageType(op.getInput()).isF32()) {
    auto in_f32 = getCudaData(op.getInput());
    requant(in_f32);
  } else {
    auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    requant(in_f32.get());
    in_f32.reset();
  }
}

void py_cuda::cudaRequantFpOp(top::RequantFpOp op) {
  UNREACHABLE_OP("not implemented op", op);
}
