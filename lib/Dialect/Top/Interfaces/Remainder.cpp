//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::RemainderOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::RemainderOp::init(InferenceParameter &p) {
  return success();
}
void top::RemainderOp::deinit(InferenceParameter &p) {}

LogicalResult top::RemainderOp::inference(InferenceParameter &p) {
  int64_t num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; ++i) {

    double quo = p.inputs[0][i] / p.inputs[1][i];
    auto quo_floor = std::floor(quo);
    p.outputs[0][i] = p.inputs[0][i] - p.inputs[1][i] * quo_floor;
  }
  return success();
}

void top::RemainderOp::shape_inference() {
  common_shape_inference(getOperation());
}
