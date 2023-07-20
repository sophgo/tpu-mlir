//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"



int64_t top::MinOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::MinOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto lhs_shape =  module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(algorithm::binary_min)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::MinOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::MinOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}

void top::MinOp::shape_inference() {
  auto  op = getOperation();
  auto in_shape = module::getShape(op->getOperand(0)).vec();
  for (uint32_t n = 1; n < op->getNumOperands(); n++) {
    auto _shape = module::getShape(op->getOperand(n));
    assert(in_shape.size() == _shape.size() && "Input rank mismatch.");
    for (uint32_t i = 0; i < in_shape.size(); i++) {
      if ((in_shape[i] == 1 || _shape[i] == 1) && (_shape[i] != in_shape[i])) {
        in_shape[i] = in_shape[i] > _shape[i] ? in_shape[i]: _shape[i];
      }
    }
  }
  module::setShapeOrVerify(getOutput(), in_shape);
}
