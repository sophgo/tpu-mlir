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

int64_t top::SelectOp::getFLOPs() { return 0; }

LogicalResult top::SelectOp::init(InferenceParameter &p) { return success(); }
void top::SelectOp::deinit(InferenceParameter &p) {}

LogicalResult top::SelectOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::SelectOp::shape_inference() {
  auto axis = getAxis();
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape(in_shape);
  out_shape[axis] = 1;
  module::setShapeOrVerify(getOutput(), out_shape);
}
