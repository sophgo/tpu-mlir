//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SqueezeOp::getFLOPs() { return 0; }

LogicalResult top::SqueezeOp::init(InferenceParameter &p) { return success(); }
void top::SqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult top::SqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

void top::SqueezeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> axes_ = *axes;
  for (auto &a : axes_) {
    if (a < 0) {
      a += in_dims;
    }
  }
  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (axes_.empty()) {
      if (in_shape[i] != 1) {
        out_shape.push_back(in_shape[i]);
      }
    } else {
      if (std::find(axes_.begin(), axes_.end(), i) == axes_.end()) {
        out_shape.push_back(in_shape[i]);
      } else {
        assert(in_shape[i]);
      }
    }
  }
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
