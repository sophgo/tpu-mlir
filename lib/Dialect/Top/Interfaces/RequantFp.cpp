//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RequantFpOp::getFLOPs() { return 0; }
LogicalResult top::RequantFpOp::init(InferenceParameter &p) {
  return success();
}
void top::RequantFpOp::deinit(InferenceParameter &p) {}

LogicalResult top::RequantFpOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::RequantFpOp::shape_inference() {
  common_shape_inference(getOperation());
}
