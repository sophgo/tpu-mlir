//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::PReluOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::PReluOp::init(InferenceParameter &p) {
  auto w_shape = module::getShape(getSlope());
  auto weight_shape =
      channel_expand_dim(w_shape, module::getShape(getInput()).size());
  auto prelu = new PRelu();
  (*prelu)
      .src(p.inputs[0], module::getShape(getInput()))
      .weights(p.inputs[1], weight_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .setup();

  p.handle = (void *)prelu;

  return success();
}
void top::PReluOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto prelu = (PRelu *)p.handle;
    delete prelu;
    p.handle = nullptr;
  }
}

LogicalResult top::PReluOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto prelu = (PRelu *)p.handle;
  prelu->run();
  return success();
}

void top::PReluOp::shape_inference() { common_shape_inference(getOperation()); }
