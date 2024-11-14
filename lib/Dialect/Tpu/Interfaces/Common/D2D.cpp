//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::D2DOp::init(InferenceParameter &p) { return success(); }

void tpu::D2DOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::D2DOp::inference(InferenceParameter &p) {
  // assistanted op, don;t need to implement it
  return success();
}

bool tpu::D2DOp::support_multi_core() { return false; }
