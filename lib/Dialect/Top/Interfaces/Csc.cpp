//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::CscOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::CscOp::init(InferenceParameter &p) { return success(); }
void top::CscOp::deinit(InferenceParameter &p) {}

LogicalResult top::CscOp::inference(InferenceParameter &p) {
  // top::CscOp no need to inference
  llvm_unreachable("top::CscOp no need to inference");
  return failure();
}

void top::CscOp::shape_inference() {}
