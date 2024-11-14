//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::DtypeCastOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::DtypeCastOp::init(InferenceParameter &p) {
  return success();
}
void top::DtypeCastOp::deinit(InferenceParameter &p) {}

LogicalResult top::DtypeCastOp::inference(InferenceParameter &p) {
  // llvm_unreachable("Not Implemented");
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);

  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());

  if (in_type.isF32() && out_type.isF16()) {
    F16(p.inputs[0], p.outputs[0], num_elem);
  };

  return success();
}

void top::DtypeCastOp::shape_inference() {
  common_shape_inference(getOperation());
}
