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

int64_t top::AbsAddOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::AbsAddOp::init(InferenceParameter &p) { return success(); }
void top::AbsAddOp::deinit(InferenceParameter &p) {}

LogicalResult top::AbsAddOp::inference(InferenceParameter &p) {
  llvm_unreachable("implement top absadd inference function");
}

void top::AbsAddOp::shape_inference() {
  llvm_unreachable("implement shape inference function");
}
