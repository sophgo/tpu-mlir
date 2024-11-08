//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SigmoidOp::getFLOPs() {
  return module::getNumElements(getInput()) * 4;
}

LogicalResult top::SigmoidOp::init(InferenceParameter &p) { return success(); }
void top::SigmoidOp::deinit(InferenceParameter &p) {}

LogicalResult top::SigmoidOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
  bool log = getLog();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] =
        log ? std::log(1 / (1 + std::exp(-val))) : 1 / (1 + std::exp(-val));
  }
  return success();
}

void top::SigmoidOp::shape_inference() {
  common_shape_inference(getOperation());
}
