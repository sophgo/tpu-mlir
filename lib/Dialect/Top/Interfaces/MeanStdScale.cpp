//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MeanStdScaleOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::MeanStdScaleOp::init(InferenceParameter &p) {
  return success();
}

void top::MeanStdScaleOp::deinit(InferenceParameter &p) {}

LogicalResult top::MeanStdScaleOp::inference(InferenceParameter &p) {
  // top meanstdscale op do not need inference.
  UNREACHABLE_THIS("Not Implemented");
  return failure();
}

void top::MeanStdScaleOp::shape_inference() {
  common_shape_inference(getOperation());
}
