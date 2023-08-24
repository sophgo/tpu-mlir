//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RoundOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::RoundOp::init(InferenceParameter &p) { return success(); }
void top::RoundOp::deinit(InferenceParameter &p) {}
LogicalResult top::RoundOp::inference(InferenceParameter &p) {
  int64_t num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::round(val);
  }
  return success();
}

void top::RoundOp::shape_inference() { common_shape_inference(getOperation()); }
