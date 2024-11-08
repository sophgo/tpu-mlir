//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReluOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::ReluOp::init(InferenceParameter &p) { return success(); }
void top::ReluOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReluOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto limit = getReluLimit().convertToDouble();
  function_relu(p.inputs[0], p.outputs[0], module::getNumElements(getInput()),
                limit);
  return success();
}

void top::ReluOp::shape_inference() { common_shape_inference(getOperation()); }
