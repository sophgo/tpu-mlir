//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReciprocalOp::getFLOPs() {
  return module::getNumElements(getOutput()) * (1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::ReciprocalOp::init(InferenceParameter &p) {
  return success();
}
void top::ReciprocalOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReciprocalOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  int64_t num_elem = module::getNumElements(getOutput());
  float const_s = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = const_s / p.inputs[0][i];
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

void top::ReciprocalOp::shape_inference() {
  common_shape_inference(getOperation());
}
