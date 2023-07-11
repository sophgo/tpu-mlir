//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::ScaleLutOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::ScaleLutOp::init(InferenceParameter &p) { return success(); }
void top::ScaleLutOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScaleLutOp::inference(InferenceParameter &p) {
  // top::ScaleLutOp no need to inference
  llvm_unreachable("top::ScaleLutOp no need to inference");
  return failure();
}

void top::ScaleLutOp::shape_inference() {
  common_shape_inference(getOperation());
}
