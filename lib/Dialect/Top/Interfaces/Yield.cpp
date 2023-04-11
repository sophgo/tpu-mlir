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

int64_t top::YieldOp::getFLOPs() { return 0; }

LogicalResult top::YieldOp::init(InferenceParameter &p) { return success(); }
void top::YieldOp::deinit(InferenceParameter &p) {}

LogicalResult top::YieldOp::inference(InferenceParameter &p) {
  return success();
}

void top::YieldOp::shape_inference() {
  return;
}
