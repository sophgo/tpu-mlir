//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::Depth2SpaceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Depth2SpaceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Depth2SpaceOp::inference(InferenceParameter &p) {
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  Module::getNCHW(input(), in, ic, ih, iw);
  Module::getNCHW(output(), on, oc, oh, ow);
  assert(in == on);
  bool crd = is_CRD();
  bool inversed = is_inversed();
  int64_t bh = block_h();
  int64_t bw = block_w();
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
