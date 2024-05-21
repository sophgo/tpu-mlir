//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::BinaryConstShiftOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::BinaryConstShiftOp::init(InferenceParameter &p) {
  return success();
}
void top::BinaryConstShiftOp::deinit(InferenceParameter &p) {}

LogicalResult top::BinaryConstShiftOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::BinaryConstShiftOp::shape_inference() {
  common_shape_inference(getOperation());
}
