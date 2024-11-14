//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SoftplusOp::getFLOPs() {
  return module::getNumElements(getInput()) * 3;
}

LogicalResult top::SoftplusOp::init(InferenceParameter &p) { return success(); }

void top::SoftplusOp::deinit(InferenceParameter &p) {}

LogicalResult top::SoftplusOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  // For numerical stability the implementation reverts to the linear function
  // when input>threshold.
  int threshold = 20;
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    if (val > threshold) {
      p.outputs[0][i] = val;
    } else {
      p.outputs[0][i] = std::log(std::exp(val) + 1);
    }
  }
  return success();
}

void top::SoftplusOp::shape_inference() {
  common_shape_inference(getOperation());
}
