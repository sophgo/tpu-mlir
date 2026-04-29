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

cuda::rounding_mode_t rmode_convert(std::string mode) {
  if (mode == "HalfAwayFromZero") {
    return cuda::RD_HALF_AWAY_FROM_ZERO;
  } else if (mode == "HalfUp") {
    return cuda::RD_HALF_UP;
  } else if (mode == "HalfDown") {
    return cuda::RD_HALF_DOWN;
  } else if (mode == "HalfToEven") {
    return cuda::RD_HALF_TO_EVEN;
  } else if (mode == "HalfToOdd") {
    return cuda::RD_HALF_TO_ODD;
  } else if (mode == "HalfTowardsZero") {
    return cuda::RD_HALF_TOWARDS_ZERO;
  } else if (mode == "TowardsZero") {
    return cuda::RD_TOWARDS_ZERO;
  } else if (mode == "Up") {
    return cuda::RD_UP;
  } else if (mode == "Down") {
    return cuda::RD_DOWN;
  } else {
    llvm_unreachable("Not Implemented");
    return cuda::RD_UNKNOWN;
  }
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
    auto zero_point = qtype.getZeroPoint();
    if (is_cv18xx) {
      bool is_bf16 = in_type.isBF16();
      cuda::cvQuantInt8(input, output, BF16(1. / scale), num_elem, is_bf16, zero_point);
    } else if (in_type.isF32()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::f32ScaleToInt8(input, output, 1. / scale, num_elem,
                            !out_type.isUnsignedInteger(8), rmode, zero_point);
    } else if (in_type.isBF16()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::bf16ScaleToInt8(input, output, 1. / scale, num_elem,
                            !out_type.isUnsignedInteger(8), rmode, zero_point);
    } else if (in_type.isF16()) {
      auto rmode = rmode_convert(op.getRoundMode());
      cuda::f16ScaleToInt8(input, output, 1. / scale, num_elem,
                           !out_type.isUnsignedInteger(8), rmode, zero_point);
    } else {
      UNREACHABLE_OP("Not Implemented", op);
    }
    return;
  } else if (fOutput && isInQuant) {
    auto qtype = module::getUniformQuantizedType(op.getInput());
    auto scale = qtype.getScale();
    auto zero_point = qtype.getZeroPoint();
    auto stype = module::getStorageType(op.getInput());
    if (stype.isInteger(8)) {
      if (is_cv18xx) {
        if (out_type.isF32()) {
          cuda::cvScaleToF32(input, output, BF16(scale), num_elem, zero_point);
        } else {
          cuda::cvScaleToBF16(input, output, BF16(scale), num_elem, zero_point);
        }
      } else {
        if (out_type.isF32()) {
          cuda::int8ScaleToF32(input, output, scale, num_elem,
                              !stype.isUnsignedInteger(), zero_point);
        } else if (out_type.isBF16()) {
          cuda::int8ScaleToBF16(input, output, scale, num_elem,
                                !stype.isUnsignedInteger(), zero_point);
        } else if (out_type.isF16()) {
          cuda::int8ScaleToF16(input, output, scale, num_elem,
                              !stype.isUnsignedInteger(), zero_point);
        } else {
          UNREACHABLE_OP("Not Implemented", op);
        }
      }
    } else if (stype.isInteger(16)) {
      if (out_type.isF32()) {
        cuda::int16ScaleToF32(input, output, scale, num_elem, zero_point);
      } else if (out_type.isBF16()) {
        cuda::int16ScaleToBF16(input, output, scale, num_elem, zero_point);
      } else if (out_type.isF16()) {
        cuda::int16ScaleToF16(input, output, scale, num_elem, zero_point);
      } else {
        UNREACHABLE_OP("Not Implemented data type", op);
      }
    }
  } else {
    cuda::rounding_mode_t rmode = is_cv18xx ? cuda::RD_TOWARDS_ZERO : cuda::RD_HALF_TO_EVEN;
    cuda::convertType(input, output, num_elem, getCudaType(op.getInput()),
                      getCudaType(op.getOutput()), rmode);
  }
}

void py_cuda::cudaCastOp(top::CastOp op) {
  auto to = op.getTo();
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_elem = module::getNumElements(op.getOutput());
  if (to == "INT32") {
    auto round_mode = rmode_convert(op.getRoundMode().str());
    cuda::convertType(input, output, num_elem, cuda::DT_F32, cuda::DT_INT32, round_mode);
    cuda::convertType(output, output, num_elem, cuda::DT_INT32, cuda::DT_F32);
  } else if (to == "F32") {
    CHECK_CUDA(cudaMemcpy(output, input, num_elem * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }
}