//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::C2CBroadcastOp::init(InferenceParameter &p) {
  return success();
}

void tpu::C2CBroadcastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::C2CBroadcastOp::inference(InferenceParameter &p) {
  auto num = module::getNumElements(getInput());
  memcpy(p.outputs[0], p.inputs[0], num * sizeof(float));
  return success();
}

bool tpu::C2CBroadcastOp::support_multi_core() { return module::isBM1684X2(); }
