//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::ConvOp::parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw,
                             int64_t &oc, int64_t &oh, int64_t &ow, int64_t &g,
                             int64_t &kh, int64_t &kw, int64_t &ins_h,
                             int64_t &ins_w, int64_t &sh, int64_t &sw,
                             int64_t &pt, int64_t &pb, int64_t &pl, int64_t &pr,
                             int64_t &dh, int64_t &dw, bool &is_dw,
                             bool &with_bias, bool &do_relu) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  do_relu = this->do_relu();
  with_bias = !bias().getType().isa<NoneType>();
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  auto kernel = Module::getI64Array(kernel_shape());
  kh = kernel->at(0);
  kw = kernel->at(1);
  auto pads_v = Module::getI64Array(pads());
  pt = pads_v->at(0);
  pl = pads_v->at(1);
  pb = pads_v->at(2);
  pr = pads_v->at(3);
  auto strides_v = Module::getI64Array(strides());
  sh = strides_v->at(0);
  sw = strides_v->at(1);
  auto dhdw = Module::getI64Array(dilations(), 2, 1);
  dh = dhdw->at(0);
  dw = dhdw->at(1);
  auto ins = Module::getI64Array(inserts(), 2, 0);
  ins_h = ins->at(0);
  ins_w = ins->at(1);
  g = group();
  is_dw = (oc == ic && oc == g && g > 1);
  return;
}

int64_t top::ConvOp::getFLOPs() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, has_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, has_relu);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return Module::getNumElements(output()) * (kw * kw * ic / g * 2 + extra);
}

LogicalResult top::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu());
  p.handle = (void *)conv;
  return success();
}

void top::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult top::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  return success();
}
