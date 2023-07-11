//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ErfOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::ErfOp::init(InferenceParameter &p) { return success(); }
void top::ErfOp::deinit(InferenceParameter &p) {}

LogicalResult top::ErfOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = std::erf(p.inputs[0][i]);
  }
  return success();
}

void top::ErfOp::shape_inference() { common_shape_inference(getOperation()); }
