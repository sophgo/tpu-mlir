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
  auto num = module::getNumElements(getInput());
  std::vector<int64_t> out_shape;
  auto shape = module::getI64Array(getShape());
  int x = -1;
  for (int i = 0; i < shape->size(); i++) {
    auto s = shape->at(i);
    if (s > 0) {
      out_shape.push_back(s);
      num /= s;
    } else if (s == 0) {
      out_shape.push_back(in_shape[i]);
      num /= in_shape[i];
    } else if (s == -1) {
      out_shape.push_back(-1);
      x = i;
    } else {
      dump();
      llvm_unreachable("shape is illegal");
    }
  }
  if (x >= 0) {
    out_shape[x] = num;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  module::getModuleOp().dump();
}
