//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::BatchNormOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::BatchNormOp::init(InferenceParameter &p) {
  return success();
}
void top::BatchNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::BatchNormOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::BatchNormOp::shape_inference() {
  common_shape_inference(getOperation());
}
