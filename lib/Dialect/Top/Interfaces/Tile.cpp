//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::TileOp::getFLOPs() { return 0; }

LogicalResult top::TileOp::init(InferenceParameter &p) { return success(); }
void top::TileOp::deinit(InferenceParameter &p) {}

LogicalResult top::TileOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  tile(p.inputs[0], p.outputs[0], in_shape, getAxis(), getTile());
  return success();
}

void top::TileOp::shape_inference() {
  auto axis_ = getAxis();
  auto tile_ = getTile();
  auto in0_shape = module::getShape(getInput());
  if (axis_ < 0) {
    axis_ += in0_shape.size();
    setAxis(axis_);
  }
  std::vector<int64_t> out_shape(in0_shape);
  out_shape[axis_] *= tile_;
  module::setShapeOrVerify(getOutput(), out_shape);
}
