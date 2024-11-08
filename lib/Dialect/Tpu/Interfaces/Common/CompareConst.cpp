//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::CompareConstOp::init(InferenceParameter &p) {
  return success();
}
void tpu::CompareConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CompareConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  const auto num_element = module::getNumElements(getOutput());
  float _const_val_ = getConstVal().convertToDouble();
  auto in_type = module::getStorageType(getInput());
  if (in_type.isF16()) {
    _const_val_ = F16(_const_val_);
  }
  if (in_type.isBF16()) {
    _const_val_ = BF16(_const_val_);
  }
  const float const_val_ = _const_val_;
  if (!getInversed()) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(p.inputs[0][i], const_val_, getMode());
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(const_val_, p.inputs[0][i], getMode());
    }
  }
  return success();
}

ArrayAttr tpu::CompareConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::CompareConstOp::support_multi_core() { return false; }
