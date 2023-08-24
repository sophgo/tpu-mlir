//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::UnsqueezeOp::getFLOPs() { return 0; }

LogicalResult top::UnsqueezeOp::init(InferenceParameter &p) {
  return success();
}
void top::UnsqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult top::UnsqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

void top::UnsqueezeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> out_shape(in_shape);
  std::vector<int64_t> axes_(*axes);
  int64_t out_dims = in_shape.size() + axes_.size();
  for (int i = 0; i < axes_.size(); ++i) {
    if (axes_[i] < 0) {
      axes_[i] += out_dims;
    }
  }
  std::sort(axes_.begin(), axes_.end());
  for (auto axis : axes_) {
    out_shape.insert(out_shape.begin() + axis, 1);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    auto input_shape_v = module::getShapeTensorValue(getInput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_shape_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
