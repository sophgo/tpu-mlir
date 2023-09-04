//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::BatchNormBwdOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::BatchNormBwdOp::init(InferenceParameter &p) {
  return success();
}
void top::BatchNormBwdOp::deinit(InferenceParameter &p) {}

LogicalResult top::BatchNormBwdOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}

void top::BatchNormBwdOp::shape_inference() {

}
