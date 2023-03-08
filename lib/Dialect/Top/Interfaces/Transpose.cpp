//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::TransposeOp::getFLOPs() { return 0; }

LogicalResult top::TransposeOp::init(InferenceParameter &p) {return success();}

void top::TransposeOp::deinit(InferenceParameter &p) {}

LogicalResult top::TransposeOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::TransposeOp::shape_inference() {
  auto dim0_ = getDim0();
  auto dim1_ = getDim1();
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape(in_shape);
  if (in_shape.size() >= 2) {
    out_shape[dim0_] = in_shape[dim1_];
    out_shape[dim1_] = in_shape[dim0_];
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
