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
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "preprocess_inference"

LogicalResult tpu::PreprocessOp::init(InferenceParameter &p) {
  return success();
}

void tpu::PreprocessOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::PreprocessOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return failure();
}

bool tpu::PreprocessOp::support_multi_core() { return false; }
