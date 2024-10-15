//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShapeTransposeOp::init(InferenceParameter &p) { return success(); }
void tpu::ShapeTransposeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeTransposeOp::inference(InferenceParameter &p) {
  return success();
}

bool tpu::ShapeTransposeOp::support_multi_core() { return false; }