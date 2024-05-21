//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::BinaryShiftOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::BinaryShiftOp::init(InferenceParameter &p) {
  return success();
}
void top::BinaryShiftOp::deinit(InferenceParameter &p) {}

LogicalResult top::BinaryShiftOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::BinaryShiftOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  broadcast_tensor_reshape(getOutput(), getInput1());
  broadcast_tensor_reshape(getOutput(), getInput2());
}
