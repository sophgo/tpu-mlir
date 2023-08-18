//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include <queue>
#include <vector>

LogicalResult tpu::ShapeArithOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeArithOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeArithOp::inference(InferenceParameter &p) {
  return success();
}
