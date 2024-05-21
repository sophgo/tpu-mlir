//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::TransposeOp::getFLOPs() { return 0; }

LogicalResult top::TransposeOp::init(InferenceParameter &p) {
  return success();
}

void top::TransposeOp::deinit(InferenceParameter &p) {}

LogicalResult top::TransposeOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::TransposeOp::shape_inference() {
  auto dim0_ = getDim0();
  auto dim1_ = getDim1();
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  if (dim0_ < 0) {
    dim0_ += num_dims;
  }
  if (dim1_ < 0) {
    dim1_ += num_dims;
  }
  std::vector<int64_t> out_shape(in_shape);
  if (in_shape.size() >= 2) {
    out_shape[dim0_] = in_shape[dim1_];
    out_shape[dim1_] = in_shape[dim0_];
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  std::vector<int64_t> order;
  for (int i = 0; i < num_dims; ++i) {
    if (dim0_ == i) {
      order.push_back(dim1_);
    } else if (dim1_ == i) {
      order.push_back(dim0_);
    } else {
      order.push_back(i);
    }
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  // rewrite
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("order", builder.getI64ArrayAttr(order)));
  auto new_op = builder.create<PermuteOp>(getLoc(), getOutput().getType(),
                                          ValueRange{getInput()}, attrs);
  op->replaceAllUsesWith(new_op);
}
