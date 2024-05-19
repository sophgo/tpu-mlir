//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::Depth2SpaceOp::getFLOPs() { return 0; }

LogicalResult top::Depth2SpaceOp::init(InferenceParameter &p) {
  return success();
}
void top::Depth2SpaceOp::deinit(InferenceParameter &p) {}

LogicalResult top::Depth2SpaceOp::inference(InferenceParameter &p) {
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  if (getInIs_NCHW()) {
    module::getNCHW(getInput(), in, ic, ih, iw, false);
  } else {
    module::getNCHW(getInput(), in, ih, iw, ic, false);
  }
  if (getOutIs_NCHW()) {
    module::getNCHW(getOutput(), on, oc, oh, ow, false);
  } else {
    module::getNCHW(getOutput(), on, oh, ow, oc, false);
  }
  ASSERT_THIS(in == on);
  bool crd = getIs_CRD();
  bool inversed = getIsInversed();
  int64_t bh = getBlockH();
  int64_t bw = getBlockW();
  int64_t instride = ic * ih * iw;
  int64_t icstride = ih * iw;
  int64_t ihstride = iw;
  int64_t iwstride = 1;
  int64_t onstride = oc * oh * ow;
  int64_t ocstride = oh * ow;
  int64_t ohstride = ow;
  int64_t owstride = 1;
  if (!getInIs_NCHW()) {
    icstride = 1;
    ihstride = iw * ic;
    iwstride = ic;
  }
  if (!getOutIs_NCHW()) {
    ocstride = 1;
    ohstride = ow * oc;
    owstride = oc;
  }
  if (inversed) {
    std::swap(in, on);
    std::swap(ic, oc);
    std::swap(ih, oh);
    std::swap(iw, ow);
    std::swap(instride, onstride);
    std::swap(icstride, ocstride);
    std::swap(ihstride, ohstride);
    std::swap(iwstride, owstride);
  }
  int64_t nc = in * ic;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int64_t i = 0; i < nc; i++) {
    int64_t n = i / ic;
    int64_t c = i % ic;
    int64_t new_c, left;
    if (crd) { // oc, bh, bw
      new_c = c / (bh * bw);
      left = c % (bh * bw);
    } else { // bh, bw, oc
      new_c = c % oc;
      left = c / oc;
    }
    if (getSwapCr()) {
      int64_t c1 = left / bw;
      int64_t c2 = left % bw;
      int64_t rleft = c2 * bh + c1;
      if (crd) {
        c = new_c * (bh * bw) + rleft;
      } else {
        c = rleft * oc + new_c;
      }
    }
    for (int64_t h = 0; h < ih; h++) {
      for (int64_t w = 0; w < iw; w++) {
        int64_t new_h = h * bh + left / bw;
        int64_t new_w = w * bw + left % bw;
        int64_t i_index =
            n * instride + c * icstride + h * ihstride + w * iwstride;
        int64_t o_index = n * onstride + new_c * ocstride + new_h * ohstride +
                          new_w * owstride;
        if (inversed) {
          p.outputs[0][i_index] = p.inputs[0][o_index];
        } else {
          p.outputs[0][o_index] = p.inputs[0][i_index];
        }
      }
    }
  }
  return success();
}

void top::Depth2SpaceOp::shape_inference() {
  int64_t in, ic, ih, iw, oc, oh, ow;
  if (getInIs_NCHW()) {
    module::getNCHW(getInput(), in, ic, ih, iw, false);
  } else {
    module::getNCHW(getInput(), in, ih, iw, ic, false);
  }
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  std::vector<int64_t> out_shape = in_shape;
  auto block_h = getBlockH();
  auto block_w = getBlockW();
  if (getIsInversed()) {
    oc = ic * block_h * block_w;
    oh = ih / block_h;
    ow = iw / block_w;
  } else {
    oc = ic / (block_h * block_w);
    oh = ih * block_h;
    ow = iw * block_w;
  }
  if (getOutIs_NCHW()) {
    out_shape[num_dims - 3] = oc;
    out_shape[num_dims - 2] = oh;
    out_shape[num_dims - 1] = ow;
  } else {
    out_shape[num_dims - 3] = oh;
    out_shape[num_dims - 2] = ow;
    out_shape[num_dims - 1] = oc;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
