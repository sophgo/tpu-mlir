//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SortOp::getFLOPs() {
  int num_elem = module::getNumElements(getInput());
  return num_elem * num_elem;
}

LogicalResult top::SortOp::init(InferenceParameter &p) { return success(); }
void top::SortOp::deinit(InferenceParameter &p) {}

LogicalResult top::SortOp::inference(InferenceParameter &p) {
  sort_param_t param = {.axis = (int)getAxis(), .descending = getDescending()};
  auto shape = module::getShape(getInput());
  int dims = shape.size();
  std::vector<int> shape_v(dims);
  for (int i = 0; i < dims; ++i) {
    shape_v[i] = shape[i];
  }
  sort_per_dim(param, shape_v.data(), dims, p.inputs[0], p.outputs[0],
               p.outputs[1]);
  return success();
}

void top::SortOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto axis = getAxis();
  if (axis < 0) {
    axis += in_shape.size();
    setAxis(axis);
  }
  if (!module::isNone(getValues())) {
    module::setShapeOrVerify(getResult(0), in_shape);
  }
  module::setShapeOrVerify(getResult(1), in_shape);
}
