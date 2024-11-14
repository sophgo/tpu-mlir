//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::LutOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::LutOp::init(InferenceParameter &p) { return success(); }
void top::LutOp::deinit(InferenceParameter &p) {}

LogicalResult top::LutOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    int offset = p.inputs[0][i];
    if (offset < 0) {
      offset += 256;
    }
    ASSERT_THIS(offset >= 0 && offset <= 255);
    p.outputs[0][i] = p.inputs[1][offset];
  }
  return success();
}

void top::LutOp::shape_inference() { common_shape_inference(getOperation()); }
