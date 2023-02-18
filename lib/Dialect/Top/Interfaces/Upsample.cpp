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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"


bool top::UpsampleOp::isEltwise() {
  return false;
}

int64_t top::UpsampleOp::getFLOPs() {
  return module::getNumElements(getOutput()) *
         (getDoRelu() ? 2 : 1);
}

LogicalResult top::UpsampleOp::init(InferenceParameter &p) { return success(); }
void top::UpsampleOp::deinit(InferenceParameter &p) {}

LogicalResult top::UpsampleOp::inference(InferenceParameter &p) {
  int64_t n, c, ih, iw;
  module::getNCHW(getInput(), n, c, ih, iw);
  int64_t oh = ih * getScaleH();
  int64_t ow = iw * getScaleW();
  auto num_elem = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t d0 = 0; d0 < n; d0++) {
    for (int64_t d1 = 0; d1 < c; d1++) {
      for (int64_t d2 = 0; d2 < oh; d2++) {
        for (int64_t d3 = 0; d3 < ow; d3++) {
          int64_t idx_o = (((d0 * c + d1) * oh) + d2) * ow + d3;
          int64_t idx_i = ((((d0 * c + d1) * ih) + d2 / getScaleH())) * iw + (d3 / getScaleW());
          p.outputs[0][idx_o] = p.inputs[0][idx_i];
        }
      }
    }
  }

  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}
