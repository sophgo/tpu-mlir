//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::PreprocessOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::PreprocessOp::init(InferenceParameter &p) {
  return success();
}
void top::PreprocessOp::deinit(InferenceParameter &p) {}

LogicalResult top::PreprocessOp::inference(InferenceParameter &p) {
  // top::PreprocessOp no need to inference
  llvm_unreachable("top::PreprocessOp no need to inference");
  return failure();
}

void top::PreprocessOp::shape_inference() {
  common_shape_inference(getOperation());
}
