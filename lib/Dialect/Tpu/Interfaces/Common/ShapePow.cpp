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

LogicalResult tpu::ShapePowOp::init(InferenceParameter &p) { return success(); }
void tpu::ShapePowOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapePowOp::inference(InferenceParameter &p) {

  auto exponent = static_cast<float>(getExponent().convertToDouble());
  auto num_element = module::getNumElements(getOutput());

  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::pow(val, exponent);
  }
  return success();
}

bool tpu::ShapePowOp::support_multi_core() { return false; }
