//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RandnLikeOp::getFLOPs() { return 0; }

LogicalResult top::RandnLikeOp::init(InferenceParameter &p) {
  return success();
}
void top::RandnLikeOp::deinit(InferenceParameter &p) {}

LogicalResult top::RandnLikeOp::inference(InferenceParameter &p) {
  llvm_unreachable("Should be convert to other ops in it's canonicalize pass.");
  return success();
}

void top::RandnLikeOp::shape_inference() {
  common_shape_inference(getOperation());
}
