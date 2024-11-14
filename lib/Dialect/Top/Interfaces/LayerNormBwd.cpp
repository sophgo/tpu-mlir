//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::LayerNormBwdOp::getFLOPs() { return 0; }

LogicalResult top::LayerNormBwdOp::init(InferenceParameter &p) {
  return success();
}
void top::LayerNormBwdOp::deinit(InferenceParameter &p) {}

LogicalResult top::LayerNormBwdOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::LayerNormBwdOp::shape_inference() {}
