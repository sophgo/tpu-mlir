//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Weight2ActivationOp::init(InferenceParameter &p) {
  if (!module::isWeight(getInput()) || module::isWeight(getOutput())) {
    llvm_unreachable("Weight2Activation tensor type error");
  }
  return success();
}

void tpu::Weight2ActivationOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Weight2ActivationOp::inference(InferenceParameter &p) {
  const auto bytes = module::getBytes(getInput());
  memcpy(p.outputs[0], p.inputs[0], bytes);
  return success();
}

bool tpu::Weight2ActivationOp::support_multi_core() { return false; }
