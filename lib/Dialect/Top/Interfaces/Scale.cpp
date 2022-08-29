//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::ScaleOp::getFLOPs() {
  return Module::getNumElements(output()) *
         (2 + do_relu() ? 1 : 0);
}

LogicalResult top::ScaleOp::init(InferenceParameter &p) { return success(); }
void top::ScaleOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScaleOp::inference(InferenceParameter &p) {
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
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

  auto num_elem = Module::getNumElements(output());
  if (do_relu()) {
    function_relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}
