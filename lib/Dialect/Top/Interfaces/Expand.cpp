//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ExpandOp::getFLOPs() { return 0; }

LogicalResult top::ExpandOp::init(InferenceParameter &p) { return success(); }
void top::ExpandOp::deinit(InferenceParameter &p) {}

LogicalResult top::ExpandOp::inference(InferenceParameter &p) {
  llvm_unreachable("Should be convert to other ops in it's canonicalize pass.");
  return success();
}

void top::ExpandOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape;
  auto shape = module::getI64Array(getShape());
  auto in_dims = in_shape.size();
  auto new_dims = shape->size();
  assert(in_dims <= new_dims);
  auto sub = new_dims - in_dims;
  for (int i = 0; i < new_dims; i++) {
    auto s = shape->at(i);
    if (i < sub) {
      out_shape.push_back(s);
    } else if (s == 1) {
      out_shape.push_back(in_shape[i - sub]);
    } else if (in_shape[i - sub] == 1 || in_shape[i - sub] == s) {
      out_shape.push_back(s);
    } else {
      dump();
      llvm_unreachable("shape is illegal");
    }
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
