//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::AbsOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::AbsOp::init(InferenceParameter &p) { return success(); }
void top::AbsOp::deinit(InferenceParameter &p) {}

LogicalResult top::AbsOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::abs(val);
  }
  return success();
}

void top::AbsOp::shape_inference() { common_shape_inference(getOperation()); }
