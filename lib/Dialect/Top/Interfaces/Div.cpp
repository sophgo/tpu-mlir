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


bool top::DivOp::isEltwise() {
  return false;
}

int64_t top::DivOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::DivOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto lhs_shape =  module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  auto max_ndim = std::max(lhs_shape.size(), rhs_shape.size());
  auto input0_shape = shape_expand_dim(lhs_shape, max_ndim);
  auto input1_shape = shape_expand_dim(rhs_shape, max_ndim);
  (*binary)
      .lhs(p.inputs[0], input0_shape)
      .rhs(p.inputs[1], input1_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_div)
      .setup();

  p.handle = (void *)binary;

  return success();
}
void top::DivOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::DivOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  return success();
}
