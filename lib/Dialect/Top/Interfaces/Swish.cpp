//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SwishOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult top::SwishOp::init(InferenceParameter &p) { return success(); }
void top::SwishOp::deinit(InferenceParameter &p) {}

LogicalResult top::SwishOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
  auto beta = getBeta().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = val / (1 + std::exp(-val * beta));
  }
  return success();
}

void top::SwishOp::shape_inference() { common_shape_inference(getOperation()); }
