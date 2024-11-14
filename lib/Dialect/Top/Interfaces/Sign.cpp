//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SignOp::getFLOPs() {
  return module::getNumElements(getInput()) * 3;
}

LogicalResult top::SignOp::init(InferenceParameter &p) { return success(); }
void top::SignOp::deinit(InferenceParameter &p) {}

LogicalResult top::SignOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    if (val > 0) {
      p.outputs[0][i] = 1;
    } else if (val < 0) {
      p.outputs[0][i] = -1;
    } else {
      p.outputs[0][i] = 0;
    }
  }
  return success();
}

void top::SignOp::shape_inference() { common_shape_inference(getOperation()); }
