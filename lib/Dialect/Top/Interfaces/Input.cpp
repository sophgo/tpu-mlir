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
  if (!getShapeTensor().has_value())
    return;
  std::vector<int64_t> shape_tensor =
      *(module::getI64Array(getShapeTensor().value()));
  if (shape_tensor.size() > 0) {
    module::bindShapeTensorValue(getOutput(), shape_tensor);
  }
  // removeShapeTensorAttr();
}
