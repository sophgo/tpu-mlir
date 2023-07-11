//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::HardSigmoidOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult top::HardSigmoidOp::init(InferenceParameter &p) {
  return success();
}
void top::HardSigmoidOp::deinit(InferenceParameter &p) {}

static inline double hsigmoid(double x, double alpha, double beta) {
  return std::max(0.0, std::min(1.0, alpha * x + beta));
}

LogicalResult top::HardSigmoidOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
  const double alpha_ = getAlpha().convertToDouble();
  const double beta_ = getBeta().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = hsigmoid(p.inputs[0][i], alpha_, beta_);
  }
  return success();
}

void top::HardSigmoidOp::shape_inference() {
  common_shape_inference(getOperation());
}
