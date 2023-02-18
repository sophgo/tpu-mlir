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


bool top::ReluOp::isEltwise() {
  return true;
}

int64_t top::ReluOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::ReluOp::init(InferenceParameter &p) { return success(); }
void top::ReluOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReluOp::inference(InferenceParameter &p) {
  auto limit = getReluLimit().convertToDouble();
  function_relu(p.inputs[0], p.outputs[0], module::getNumElements(getInput()),
                limit);
  return success();
}
