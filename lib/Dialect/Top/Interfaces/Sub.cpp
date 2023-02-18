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


bool top::SubOp::isEltwise() {
  return false;
}

int64_t top::SubOp::getFLOPs() {
  return module::getNumElements(getOutput()) *
         (getInputs().size() - 1 + getDoRelu() ? 1 : 0);
}

LogicalResult top::SubOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  int index0 = 0, index1 = 1;
  if (getIsReverse()) {
    index0 = 1, index1 = 0;
  }
  (*binary)
      .lhs(p.inputs[index0], module::getShape(getInputs()[index0]))
      .rhs(p.inputs[index1], module::getShape(getInputs()[index1]))
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_sub)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::SubOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::SubOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}
