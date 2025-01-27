//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Mmap2RgbmapOp::init(InferenceParameter &p) {
  return success();
}

void tpu::Mmap2RgbmapOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::Mmap2RgbmapOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

bool tpu::Mmap2RgbmapOp::support_multi_core() { return false; }
