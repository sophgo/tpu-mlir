//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::EmbDenseBwdOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::EmbDenseBwdOp::init(InferenceParameter &p) {
  return success();
}
void top::EmbDenseBwdOp::deinit(InferenceParameter &p) {}

LogicalResult top::EmbDenseBwdOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::EmbDenseBwdOp::shape_inference() {}
