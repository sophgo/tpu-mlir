//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

static inline int align_up(int x, int n) {
  if (n == 0 || n == 1) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

static inline float UINT8(float data) {
  return static_cast<float>(to_uint8(data, ROUNDING_HALF_TO_EVEN));
}

int64_t top::CscOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::CscOp::init(InferenceParameter &p) {
  return success();
}
void top::CscOp::deinit(InferenceParameter &p) {}

LogicalResult top::CscOp::inference(InferenceParameter &p) {
  //top::CscOp no need to inference
  llvm_unreachable("top::CscOp no need to inference");
  return failure();
}
