//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::C2CBroadcastOp::getFLOPs() { return 0; }

LogicalResult top::C2CBroadcastOp::init(InferenceParameter &p) {
  return success();
}

void top::C2CBroadcastOp::deinit(InferenceParameter &p) {}

LogicalResult top::C2CBroadcastOp::inference(InferenceParameter &p) {
  auto num = module::getNumElements(getInput());
  memcpy(p.outputs[0], p.inputs[0], num * sizeof(float));
  return success();
}

void top::C2CBroadcastOp::shape_inference() {
  module::setShapeOrVerify(getOutput(), module::getShape(getInput()));
}
