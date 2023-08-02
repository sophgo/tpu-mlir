//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::LoopOp::init(InferenceParameter &p) {
  return success();
}

void tpu::LoopOp::deinit(InferenceParameter &p) {
}

LogicalResult tpu::LoopOp::inference(InferenceParameter &p) {
  return success();
}
