//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::Pow2Op::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::Pow2Op::init(InferenceParameter &p) { return success(); }
void top::Pow2Op::deinit(InferenceParameter &p) {}

LogicalResult top::Pow2Op::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto val = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto ex = p.inputs[0][i];
    p.outputs[0][i] = std::pow(val, ex);
  }
  return success();
}

void top::Pow2Op::shape_inference() { common_shape_inference(getOperation()); }
