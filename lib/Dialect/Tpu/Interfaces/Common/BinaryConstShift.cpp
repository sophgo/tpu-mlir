//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::BinaryConstShiftOp::init(InferenceParameter &p) {
  return success();
}

void tpu::BinaryConstShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::BinaryConstShiftOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  int64_t const_val = getScale();
  int32_t shift_val = getShift();
  auto rmode = round_mode_convert(getRoundMode());
  // bool is_satu = getSaturation();
  if (getMode().str() == "Add") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] + const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else if (getMode().str() == "Sub") {
    if (getIsReverse() == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = p.inputs[0][i] - const_val;
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = const_val - p.inputs[0][i];
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    }
  } else if (getMode().str() == "Mul") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] * const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else {
    UNREACHABLE_THIS("Not Implemented");
  }

  return success();
}

ArrayAttr tpu::BinaryConstShiftOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::BinaryConstShiftOp::support_multi_core() { return false; }
