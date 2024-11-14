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

LogicalResult tpu::ShapeSqueezeOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeSqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeSqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

mlir::Type tpu::ShapeSqueezeOp::type_verify(uint64_t opd_idx,
                                            TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::ShapeSqueezeOp::support_multi_core() { return false; }
