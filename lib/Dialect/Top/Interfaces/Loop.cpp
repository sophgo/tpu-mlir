//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::LoopOp::getFLOPs() { return 0; }

LogicalResult top::LoopOp::init(InferenceParameter &p) { return success(); }
void top::LoopOp::deinit(InferenceParameter &p) {}

LogicalResult top::LoopOp::inference(InferenceParameter &p) {
  return success();
}

void top::LoopOp::shape_inference() {
  return;
}
