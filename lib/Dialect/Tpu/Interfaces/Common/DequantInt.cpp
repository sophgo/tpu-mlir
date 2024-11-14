//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::DequantIntOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DequantIntOp::inference(InferenceParameter &p) {
  auto qtype = module::getUniformQuantizedType(getInput());
  int64_t num_elem = module::getNumElements(getInput());
  int64_t shift_val = getShift();
  int64_t mul_val = getMultiplier();
  int64_t offset = (int64_t)qtype.getZeroPoint();
  auto qmode = getQuantMode();
  auto rmode = round_mode_convert(getRoundMode());
  switch (qmode) {
  case DequantMode::Normal: {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int32_t tmp = (int32_t)p.inputs[0][idx] - offset;
      auto v = applyMultiplierAndRShift(
          tmp, mul_val, -shift_val, tpu::RequantMode::MultiplierShift, rmode);
      p.outputs[0][idx] = v;
    }
  } break;
  case DequantMode::TFLite: {
    int64_t lshift_val = getLshift();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int64_t tmp = ((int32_t)p.inputs[0][idx] - offset) * mul_val
                    << lshift_val;
      auto v = RightShiftRound(tmp, 31, ROUNDING_HALF_UP);
      v = RightShiftRound(v, -shift_val, (RoundingMode)rmode);
      p.outputs[0][idx] = v;
    }
  } break;
  default:
    llvm_unreachable("Unknown dequant mode");
    break;
  }
  return success();
}

mlir::Type tpu::DequantIntOp::type_verify(uint64_t opd_idx,
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

ArrayAttr tpu::DequantIntOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::DequantIntOp::support_multi_core() { return false; }
