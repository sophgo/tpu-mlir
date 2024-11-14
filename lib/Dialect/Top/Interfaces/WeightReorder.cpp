//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::WeightReorderOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::WeightReorderOp::init(InferenceParameter &p) {
  return success();
}
void top::WeightReorderOp::deinit(InferenceParameter &p) {}

LogicalResult top::WeightReorderOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::WeightReorderOp::shape_inference() {}
