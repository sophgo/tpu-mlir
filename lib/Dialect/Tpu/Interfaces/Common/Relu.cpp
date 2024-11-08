//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ReluOp::init(InferenceParameter &p) { return success(); }
void tpu::ReluOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReluOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto limit = getReluLimit().convertToDouble();
  function_relu(p.inputs[0], p.outputs[0], module::getNumElements(getOutput()),
                limit, module::getStorageType(getOutput()));
  return success();
}

ArrayAttr tpu::ReluOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::ReluOp::support_multi_core() { return false; }
