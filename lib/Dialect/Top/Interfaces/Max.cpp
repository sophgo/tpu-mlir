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


bool top::MaxOp::isEltwise() {
  return false;
}

int64_t top::MaxOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::MaxOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  (*binary)
      .lhs(p.inputs[0], module::getShape(getInputs()[0]))
      .rhs(p.inputs[1], module::getShape(getInputs()[1]))
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(algorithm::binary_max)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::MaxOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::MaxOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}
