//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ViewOp::getFLOPs() { return 0; }

LogicalResult top::ViewOp::init(InferenceParameter &p) { return success(); }
void top::ViewOp::deinit(InferenceParameter &p) {}

LogicalResult top::ViewOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::ViewOp::shape_inference() {
  auto weight = cast<top::WeightOp>(getShape().getDefiningOp());
  auto shape = weight.read<float>();
  std::vector<int64_t> shape_(shape->begin(), shape->end());
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto out = getOutput();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      builder.getNamedAttr("shape", builder.getI64ArrayAttr(shape_)));
  auto new_op = builder.create<top::ReshapeOp>(
      getLoc(), out.getType(), ArrayRef<Value>{getInput()}, attrs);
  out.replaceAllUsesWith(new_op.getOutput());
  new_op.shape_inference();
}
