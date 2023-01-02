//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



LogicalResult tpu::Depth2SpaceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Depth2SpaceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Depth2SpaceOp::inference(InferenceParameter &p) {
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getInput(), in, ic, ih, iw);
  module::getNCHW(getOutput(), on, oc, oh, ow);
  assert(in == on);
  bool crd = getIs_CRD();
  bool inversed = getIsInversed();
  int64_t bh = getBlockH();
  int64_t bw = getBlockW();
  if (inversed) {
    std::swap(in, on);
    std::swap(ic, oc);
    std::swap(ih, oh);
    std::swap(iw, ow);
  }
  int64_t nc = in * ic;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int64_t i = 0; i < nc; i++) {
    int64_t n = i / ic;
    int64_t c = i % ic;
    int64_t new_c, new_h, new_w, left;
    if (crd == true) {
      new_c = c / (bh * bw);
      left = c % (bh * bw);
    } else {
      new_c = c % oc;
      left = c / oc;
    }
    for (int64_t h = 0; h < ih; h++) {
      for (int64_t w = 0; w < iw; w++) {
        new_h = h * bh + left / bw;
        new_w = w * bw + left % bw;
        int64_t i_index = n * ic * ih * iw + c * ih * iw + h * iw + w;
        int64_t o_index =
            n * oc * oh * ow + new_c * oh * ow + new_h * ow + new_w;
        if (inversed == true) {
          p.outputs[0][i_index] = p.inputs[0][o_index];
        } else {
          p.outputs[0][o_index] = p.inputs[0][i_index];
        }
      }
    }
  }

  return success();
}
