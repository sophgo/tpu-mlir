//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SqrtOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::SqrtOp::init(InferenceParameter &p) { return success(); }
void top::SqrtOp::deinit(InferenceParameter &p) {}

LogicalResult top::SqrtOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::sqrt(val);
  }
  return success();
}

void top::SqrtOp::shape_inference() { common_shape_inference(getOperation()); }
