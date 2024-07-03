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

cuda::rounding_mode_t rmode_convert(tpu::RoundMode mode) {
  switch (mode) {
  case tpu::RoundMode::HalfAwayFromZero:
    return cuda::RD_HALF_AWAY_FROM_ZERO;
  case tpu::RoundMode::HalfUp:
    return cuda::RD_HALF_UP;
  case tpu::RoundMode::HalfDown:
    return cuda::RD_HALF_DOWN;
  case tpu::RoundMode::HalfToEven:
    return cuda::RD_HALF_TO_EVEN;
  case tpu::RoundMode::HalfToOdd:
    return cuda::RD_HALF_TO_ODD;
  case tpu::RoundMode::HalfTowardsZero:
    return cuda::RD_HALF_TOWARDS_ZERO;
  case tpu::RoundMode::TowardsZero:
    return cuda::RD_TOWARDS_ZERO;
  case tpu::RoundMode::Up:
    return cuda::RD_UP;
  case tpu::RoundMode::Down:
    return cuda::RD_DOWN;
  default:
    break;
  }
  llvm_unreachable("Not Implemented");
  return cuda::RD_HALF_AWAY_FROM_ZERO;
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
      bool is_bf16 = in_type.isBF16();
      cuda::cvQuantInt8(input, output, BF16(1. / scale), num_elem, is_bf16);
    } else if (in_type.isF32()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::f32ScaleToInt8(input, output, 1. / scale, num_elem,
                           !out_type.isUnsignedInteger(8), rmode);
    } else if (in_type.isBF16()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::bf16ScaleToInt8(input, output, 1. / scale, num_elem,
                            !out_type.isUnsignedInteger(8), rmode);
    } else if (in_type.isF16()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::f16ScaleToInt8(input, output, 1. / scale, num_elem,
                           !out_type.isUnsignedInteger(8), rmode);
    } else {
      UNREACHABLE_OP("Not Implemented", op);
    }
    return;
  } else if (fOutput && isInQuant) {
    auto qtype = module::getUniformQuantizedType(op.getInput());
    auto scale = qtype.getScale();
    if (is_cv18xx) {
      if (out_type.isF32()) {
        cuda::cvScaleToF32(input, output, BF16(scale), num_elem);
      } else {
        cuda::cvScaleToBF16(input, output, BF16(scale), num_elem);
      }
    } else {
      if (out_type.isF32()) {
        cuda::int8ScaleToF32(input, output, scale, num_elem,
                             !qtype.isUnsignedInteger(8));
      } else if (out_type.isBF16()) {
        cuda::int8ScaleToBF16(input, output, scale, num_elem,
                              !qtype.isUnsignedInteger(8));
      } else if (out_type.isF16()) {
        cuda::int8ScaleToF16(input, output, scale, num_elem,
                             !qtype.isUnsignedInteger(8));
      } else {
        UNREACHABLE_OP("Not Implemented", op);
      }
    }
    return;
  } else {
    cuda::convertType(input, output, num_elem, getCudaType(op.getInput()),
                      getCudaType(op.getOutput()));
  }
}
