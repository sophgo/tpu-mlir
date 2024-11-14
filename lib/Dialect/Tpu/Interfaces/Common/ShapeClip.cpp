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

LogicalResult tpu::ShapeClipOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ShapeClipOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeClipOp::inference(InferenceParameter &p) {

  auto min_v = static_cast<float>(getMin().convertToDouble());
  auto max_v = static_cast<float>(getMax().convertToDouble());
  auto num_element = module::getNumElements(getOutput());
  assert(!module::isUniformQuantized(getOutput()) && "Not Implemented");

  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::min(max_v, std::max(min_v, val));
  }
  return success();
}

bool tpu::ShapeClipOp::support_multi_core() { return false; }