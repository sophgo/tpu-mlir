//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::LeakyReluOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::LeakyReluOp::init(InferenceParameter &p) {
  return success();
}
void top::LeakyReluOp::deinit(InferenceParameter &p) {}

LogicalResult top::LeakyReluOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  float *dst = p.outputs[0];
  int64_t num_elements = module::getNumElements(getInput());
  float alpha = static_cast<float>(getAlpha().convertToDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
  for (int64_t i = 0; i < num_elements; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (alpha * src[i]);
  }
  return success();
}

void top::LeakyReluOp::shape_inference() {
  common_shape_inference(getOperation());
}
