//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::IfOp::init(InferenceParameter &p) { return success(); }

void tpu::IfOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::IfOp::inference(InferenceParameter &p) {
  if (p.inputs[0][0] > 0)
    return success(); // then_branch
  else
    return failure(); // else_branch
}

bool tpu::IfOp::support_multi_core() { return false; }
