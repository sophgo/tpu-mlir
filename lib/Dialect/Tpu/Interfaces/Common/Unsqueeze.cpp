//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

LogicalResult tpu::UnsqueezeOp::init(InferenceParameter &p) {
  return success();
}
void tpu::UnsqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::UnsqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  auto in_shape = module::getShape(getInput());
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> out_shape(in_shape);
  auto pre_op = getInput().getDefiningOp();
  bool is_scalar = module::isScalar(pre_op);
  if (!is_scalar) {
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
  }
  module::setShape(getOutput(), out_shape);
  return success();
}

bool tpu::UnsqueezeOp::support_multi_core() { return false; }
