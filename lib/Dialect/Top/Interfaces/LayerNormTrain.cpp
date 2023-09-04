//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"


int64_t top::LayerNormTrainOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::LayerNormTrainOp::init(InferenceParameter &p) {
  return success();
}
void top::LayerNormTrainOp::deinit(InferenceParameter &p) {}

LogicalResult top::LayerNormTrainOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}

void top::LayerNormTrainOp::shape_inference() {

}
