//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::FAttentionOp::getFLOPs() {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

LogicalResult top::FAttentionOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::FAttentionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FAttentionOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::FAttentionOp::shape_inference() {
  UNREACHABLE_THIS("Not Implemented");
}
