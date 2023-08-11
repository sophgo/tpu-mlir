//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShapeAssignOp::init(InferenceParameter &p) { return success(); }
void tpu::ShapeAssignOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeAssignOp::inference(InferenceParameter &p) {
  const int num_elem = module::getNumElements(getInput());
  std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  return success();
}
