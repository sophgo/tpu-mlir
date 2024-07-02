//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaDepth2SpaceOp(tpu::Depth2SpaceOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());

  int64_t in, ic, ih, iw, on, oc, oh, ow;
  if (op.getInIs_NCHW()) {
    module::getNCHW(op.getInput(), in, ic, ih, iw, false);
  } else {
    module::getNCHW(op.getInput(), in, ih, iw, ic, false);
  }
  if (op.getOutIs_NCHW()) {
    module::getNCHW(op.getOutput(), on, oc, oh, ow, false);
  } else {
    module::getNCHW(op.getOutput(), on, oh, ow, oc, false);
  }
  assert(in == on);
  bool crd = op.getIs_CRD();
  bool inversed = op.getIsInversed();
  int64_t instride = ic * ih * iw;
  int64_t icstride = ih * iw;
  int64_t ihstride = iw;
  int64_t iwstride = 1;
  int64_t onstride = oc * oh * ow;
  int64_t ocstride = oh * ow;
  int64_t ohstride = ow;
  int64_t owstride = 1;
  if (!op.getInIs_NCHW()) {
    icstride = 1;
    ihstride = iw * ic;
    iwstride = ic;
  }
  if (!op.getOutIs_NCHW()) {
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
  cuda::depth2Space(input, output, in, ic, ih, iw, on, oc, oh, ow, instride,
                    icstride, ihstride, iwstride, onstride, ocstride, ohstride,
                    owstride, op.getBlockH(), op.getBlockW(), crd,
                    op.getSwapCr(), inversed,
                    module::getDtypeSize(op.getInput()));
}
