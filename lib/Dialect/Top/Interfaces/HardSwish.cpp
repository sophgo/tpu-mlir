//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::HardSwishOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult top::HardSwishOp::init(InferenceParameter &p) {
  return success();
}
void top::HardSwishOp::deinit(InferenceParameter &p) {}

static inline double hswish(double x) {
  return x * std::max(0.0, std::min(1.0, x / 6 + 0.5));
}

LogicalResult top::HardSwishOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = hswish(p.inputs[0][i]);
  }
  return success();
}

void top::HardSwishOp::shape_inference() {
  common_shape_inference(getOperation());
}
