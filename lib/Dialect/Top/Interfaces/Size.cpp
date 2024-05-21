//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SizeOp::getFLOPs() { return 0; }

LogicalResult top::SizeOp::init(InferenceParameter &p) { return success(); }
void top::SizeOp::deinit(InferenceParameter &p) {}

LogicalResult top::SizeOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

// SizeOp is special, will convert to WeightOp
void top::SizeOp::shape_inference() {
  auto shape = module::getShape(getInput());
  std::vector<float> data;
  if (getAxis().has_value()) {
    auto axis = getAxis().value();
    if (axis < 0) {
      axis += shape.size();
    }
    data.push_back(shape[axis]);
  } else {
    for (auto s : shape) {
      data.push_back((float)s);
    }
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto weight_type =
      RankedTensorType::get({(int64_t)data.size()}, builder.getF32Type());
  auto new_op = top::WeightOp::create(op, "size", data, weight_type);
  getOutput().replaceAllUsesWith(new_op);
}
