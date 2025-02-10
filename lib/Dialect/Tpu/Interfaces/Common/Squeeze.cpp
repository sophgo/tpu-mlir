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

LogicalResult tpu::SqueezeOp::init(InferenceParameter &p) { return success(); }
void tpu::SqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SqueezeOp::inference(InferenceParameter &p) {
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

LogicalResult tpu::SqueezeOp::LocalGenSupport() {
  if (module::isCV18xx() || module::isBM1684Family()) {
    return failure();
  }
  auto runmode = getRunMode(getOperation());
  if (runmode == RunMode::TPU_DYNAMIC)
    return failure();

  auto ishape = module::getShape(getInput());
  auto oshape = module::getShape(getOutput());
  if (ishape.size() < 2 || oshape.size() < 2 || ishape[0] != oshape[0] ||
      ishape[1] != oshape[1]) {
    return failure();
  }
  return success();
}

bool tpu::SqueezeOp::support_multi_core() { return false; }
