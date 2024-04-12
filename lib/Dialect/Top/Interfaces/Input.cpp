//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

void top::InputOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape(in_shape);
  // if (out_shape.size() == 1 || out_shape.size() == 0) {
  //   module::bindShapeTensorValue(getOutput(), out_shape);
  // }
  if (getIsShape()) {
    module::bindShapeTensorValue(getOutput(), out_shape);
  }
}
