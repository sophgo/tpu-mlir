//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::CoreSplitOp::init(InferenceParameter &p) {
  return success();
}
void tpu::CoreSplitOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CoreSplitOp::inference(InferenceParameter &p) {
  // do nothing
  return success();
}

bool tpu::CoreSplitOp::support_multi_core() { return true; }
