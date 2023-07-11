//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::GELUOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult top::GELUOp::init(InferenceParameter &p) { return success(); }
void top::GELUOp::deinit(InferenceParameter &p) {}

LogicalResult top::GELUOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
  }
  return success();
}

void top::GELUOp::shape_inference() { common_shape_inference(getOperation()); }
