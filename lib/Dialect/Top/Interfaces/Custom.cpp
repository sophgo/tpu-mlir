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


int64_t top::CustomOp::getFLOPs() {
  // Flop of CustomOp cannot be determined
  return 0;
}

LogicalResult top::CustomOp::init(InferenceParameter &p) {
  return success();
}
void top::CustomOp::deinit(InferenceParameter &p) {}

LogicalResult top::CustomOp::inference(InferenceParameter &p) {
  //top::CustomOp no need to inference
  llvm_unreachable("top::CustomOp no need to inference");
  return failure();
}

void top::CustomOp::shape_inference() {}

