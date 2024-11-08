//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ClipOp::init(InferenceParameter &p) { return success(); }
void tpu::ClipOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ClipOp::inference(InferenceParameter &p) {
  auto min_v = static_cast<float>(getMin().convertToDouble());
  auto max_v = static_cast<float>(getMax().convertToDouble());
  auto num_element = module::getNumElements(getOutput());
  assert(!module::isUniformQuantized(getOutput()) && "Not Implemented");

#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::min(max_v, std::max(min_v, val));
  }
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  return success();
}

LogicalResult tpu::ClipOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  return success();
}

ArrayAttr tpu::ClipOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::ClipOp::support_multi_core() { return false; }
