//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ScaleOp::getFLOPs() {
  return module::getNumElements(getOutput()) * (2 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::ScaleOp::init(InferenceParameter &p) { return success(); }
void top::ScaleOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScaleOp::inference(InferenceParameter &p) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  const float *src = p.inputs[0];
  const float *scale = p.inputs[1];
  const float *bias = p.inputs[2];
  float *dst = p.outputs[0];
#pragma omp parallel for schedule(static, omp_schedule(c))
  for (int64_t i = 0; i < c; ++i) {
    float scale_val = scale[i];
    float bias_val = bias[i];
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t k = 0; k < h * w; ++k) {
        int64_t idx = j * c * h * w + i * h * w + k;
        dst[idx] = src[idx] * scale_val + bias_val;
      }
    }
  }

  auto num_elem = module::getNumElements(getOutput());
  if (getDoRelu()) {
    auto relu_limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, relu_limit);
  }
  return success();
}

void top::ScaleOp::shape_inference() { common_shape_inference(getOperation()); }
