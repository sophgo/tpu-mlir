//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::UpsampleOp::init(InferenceParameter &p) { return success(); }
void tpu::UpsampleOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::UpsampleOp::inference(InferenceParameter &p) {
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
          int64_t idx_i = ((((d0 * c + d1) * ih) + d2 / getScaleH())) * iw +
                          (d3 / getScaleW());
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

LogicalResult tpu::UpsampleOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                         int64_t out_idx, int64_t out_slice) {
  auto unit = getScaleH();
  if (out_idx % unit || out_slice % unit) {
    return failure();
  }
  in_idx = out_idx / unit;
  in_slice = out_slice / unit;
  return success();
}

LogicalResult tpu::UpsampleOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                         int64_t out_idx, int64_t out_slice) {
  auto unit = getScaleW();
  if (out_idx % unit || out_slice % unit) {
    return failure();
  }
  in_idx = out_idx / unit;
  in_slice = out_slice / unit;
  return success();
}

LogicalResult tpu::UpsampleOp::LocalGenSupport() {
  if (module::isCV18xx() && (getScaleH() >= 16 || getScaleW() >= 16)) {
    return failure();
  }
  return success();
}

bool tpu::UpsampleOp::support_multi_core() { return false; }
