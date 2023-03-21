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

int64_t top::SliceExOp::getFLOPs() { return 0; }

LogicalResult top::SliceExOp::init(InferenceParameter &p) { return success(); }
void top::SliceExOp::deinit(InferenceParameter &p) {}

LogicalResult top::SliceExOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::SliceExOp::shape_inference() {
  auto axis = getAxis();
  auto start = getStart();
  auto end = getEnd();
  auto step = getStep();
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape(in_shape);
  start = start >= 0 ? start : start + in_shape[axis];
  end = end >= 0 ? end : end + in_shape[axis];
  end = end < in_shape[axis] ? end : in_shape[axis] - 1;
  out_shape[axis] = (end - start) / step + 1;
  module::setShapeOrVerify(getOutput(), out_shape);
}
