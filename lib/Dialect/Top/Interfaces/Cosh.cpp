//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::CoshOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult top::CoshOp::init(InferenceParameter &p) { return success(); }
void top::CoshOp::deinit(InferenceParameter &p) {}

LogicalResult top::CoshOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::cosh(val);
  }
  return success();
}

void top::CoshOp::shape_inference() { common_shape_inference(getOperation()); }
