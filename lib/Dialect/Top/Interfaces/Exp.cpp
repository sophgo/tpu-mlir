//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ExpOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult top::ExpOp::init(InferenceParameter &p) { return success(); }
void top::ExpOp::deinit(InferenceParameter &p) {}

LogicalResult top::ExpOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::exp(val);
  }
  return success();
}

void top::ExpOp::shape_inference() { common_shape_inference(getOperation()); }
