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



int64_t top::PReluOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::PReluOp::init(InferenceParameter &p) {
  auto prelu = new PRelu();
  (*prelu)
      .src(p.inputs[0], module::getShape(getInput()))
      .weights(p.inputs[1], module::getShape(getSlope()))
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
