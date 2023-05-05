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

int64_t top::IfOp::getFLOPs() { return 0; }

LogicalResult top::IfOp::init(InferenceParameter &p) { return success(); }
void top::IfOp::deinit(InferenceParameter &p) {}

LogicalResult top::IfOp::inference(InferenceParameter &p) {
  if (p.inputs[0][0] > 0)
    return success(); //then_branch
  else
    return failure(); //else_branch
}

void top::IfOp::shape_inference() {
  return;
}
