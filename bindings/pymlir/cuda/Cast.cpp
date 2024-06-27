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

cuda_rmode_t rmode_convert(tpu::RoundMode mode) {
  switch (mode) {
  case tpu::RoundMode::HalfAwayFromZero:
    return CUDA_HALF_AWAY_FROM_ZERO;
  case tpu::RoundMode::HalfUp:
    return CUDA_HALF_UP;
  case tpu::RoundMode::HalfDown:
    return CUDA_HALF_DOWN;
  case tpu::RoundMode::HalfToEven:
    return CUDA_HALF_TO_EVEN;
  case tpu::RoundMode::HalfToOdd:
    return CUDA_HALF_TO_ODD;
  case tpu::RoundMode::HalfTowardsZero:
    return CUDA_HALF_TOWARDS_ZERO;
  case tpu::RoundMode::TowardsZero:
    return CUDA_TOWARDS_ZERO;
  case tpu::RoundMode::Up:
    return CUDA_UP;
  case tpu::RoundMode::Down:
    return CUDA_DOWN;
  default:
    break;
  }
  llvm_unreachable("Not Implemented");
  return CUDA_HALF_AWAY_FROM_ZERO;
}

void py_cuda::cudaCastOp(tpu::CastOp op) {
  auto num_elem = module::getNumElements(op.getOutput());
  auto in_type = module::getStorageType(op.getInput());
  auto out_type = module::getStorageType(op.getOutput());
  bool isInQuant = module::isUniformQuantized(op.getInput());
  bool isOutQuant = module::isUniformQuantized(op.getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  bool is_cv18xx = module::isCV18xx();
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  if (isOutQuant && fInput) {
    auto qtype = module::getUniformQuantizedType(op.getOutput());
    auto scale = qtype.getScale();
    if (is_cv18xx) {
      cudaCVQuantInt8(input, output, BF16(1. / scale), num_elem);
    } else {
      auto rmode = rmode_convert(op.getRoundMode());
      cudaF32ToInt8(input, output, 1. / scale, num_elem,
                    !out_type.isUnsignedInteger(8), rmode);
    }
    return;
  } else if (fOutput && isInQuant) {
    auto qtype = module::getUniformQuantizedType(op.getInput());
    auto scale = qtype.getScale();
    if (is_cv18xx) {
      cudaCVScaleToF32(input, output, BF16(scale), num_elem);
    } else {
      cudaInt8ToF32(input, output, scale, num_elem,
                    !qtype.isUnsignedInteger(8));
    }
    return;
  }
  UNREACHABLE_OP("Not Implemented", op);
}
