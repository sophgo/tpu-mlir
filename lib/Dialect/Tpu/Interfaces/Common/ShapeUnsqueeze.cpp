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

LogicalResult tpu::ShapeUnsqueezeOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeUnsqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeUnsqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

bool tpu::ShapeUnsqueezeOp::support_multi_core() { return false; }
