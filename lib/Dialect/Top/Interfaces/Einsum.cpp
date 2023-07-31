//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::EinsumOp::getFLOPs() {
  llvm_unreachable("GetFLOPs Not Implemented");
  return 0;
}

LogicalResult top::EinsumOp::init(InferenceParameter &p) {
  return success();
}

void top::EinsumOp::deinit(InferenceParameter &p) {}

LogicalResult top::EinsumOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::EinsumOp::shape_inference() {
  auto mode = getMode().str();
  assert(getInputs().size() == 2);
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  if (mode == "a,b->ab")  {      // outer product
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], rhs_shape[0]});
  } else if (mode == "abcd,cde->abe") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[2]});
  } else if (mode == "abcd,bed->abce") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], lhs_shape[1], lhs_shape[2], rhs_shape[1]});
  } else if (mode == "abcd,ced->abce") {
    module::setShapeOrVerify(getOutput(), {lhs_shape[0], lhs_shape[1], rhs_shape[0], rhs_shape[1]});
  } else {
    llvm_unreachable("Not support now.");
  }
}
