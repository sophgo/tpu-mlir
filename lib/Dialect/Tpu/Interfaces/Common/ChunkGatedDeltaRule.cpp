//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ChunkGatedDeltaRuleOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ChunkGatedDeltaRuleOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ChunkGatedDeltaRuleOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

bool tpu::ChunkGatedDeltaRuleOp::support_multi_core() { return true; }
