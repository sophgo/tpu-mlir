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


int64_t top::NormalizeOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::NormalizeOp::init(InferenceParameter &p) { return success(); }
void top::NormalizeOp::deinit(InferenceParameter &p) {}
LogicalResult top::NormalizeOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}
