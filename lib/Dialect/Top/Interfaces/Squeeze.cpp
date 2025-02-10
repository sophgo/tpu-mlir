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
      if ((std::find(axes_.begin(), axes_.end(), i) == axes_.end()) ||
          (in_shape[i] != 1)) {
        out_shape.push_back(in_shape[i]);
      } else {
        assert(in_shape[i]);
      }
    }
  }

  if (out_shape.empty()) {
    out_shape.push_back(1);
  }

  module::setShape(getOutput(), out_shape);
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
      if ((std::find(axes_.begin(), axes_.end(), i) == axes_.end()) ||
          (in_shape[i] != 1)) {
        out_shape.push_back(in_shape[i]);
      } else {
        ASSERT_THIS(in_shape[i]);
      }
    }
  }
  if (out_shape.empty()) {
    out_shape.push_back(1);
    auto context = getContext();
    mlir::Builder builder(context);
    setIsScalarAttr(builder.getBoolAttr(true));
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput()) && out_shape.size() <= 1) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
