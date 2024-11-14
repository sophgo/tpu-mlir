//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::DequantIntAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DequantIntAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DequantIntAxisOp::inference(InferenceParameter &p) {

  auto shape = module::getShape(getOutput());
  auto mode = getQuantMode();
  auto rmode = round_mode_convert(getRoundMode());
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }

  if (mode == DequantMode::Normal) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int32_t tmp = (int32_t)p.inputs[0][offset] - zero_point;
          p.outputs[0][offset] = applyMultiplierAndRShift(
              tmp, multi, -shift_val, tpu::RequantMode::MultiplierShift, rmode);
        }
      }
    }
  } else if (mode == DequantMode::TFLite) {
    int64_t lshift_val = getLshift();
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi = p.inputs[1][c * 3];
      int64_t shift_val = p.inputs[1][c * 3 + 1];
      int64_t zero_point = p.inputs[1][c * 3 + 2];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int64_t tmp = ((int32_t)p.inputs[0][offset] - zero_point) * multi
                        << lshift_val;
          p.outputs[0][offset] = MultiplyByQuantizedMultiplier(
              tmp, 1, -shift_val, (RoundingMode)rmode);
        }
      }
    }
  } else {
    llvm_unreachable("no such dequant mode");
  }
  return success();
}

mlir::Type tpu::DequantIntAxisOp::type_verify(uint64_t opd_idx,
                                              TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(getInput());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

bool tpu::DequantIntAxisOp::support_multi_core() { return false; }
