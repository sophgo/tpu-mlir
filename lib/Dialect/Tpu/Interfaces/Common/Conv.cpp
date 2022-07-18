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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::ConvOp::parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw,
                             int64_t &oc, int64_t &oh, int64_t &ow, int64_t &g,
                             int64_t &kh, int64_t &kw, int64_t &ins_h,
                             int64_t &ins_w, int64_t &sh, int64_t &sw,
                             int64_t &pt, int64_t &pb, int64_t &pl, int64_t &pr,
                             int64_t &dh, int64_t &dw, bool &is_dw,
                             bool &has_bias, bool &do_relu) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  do_relu = this->do_relu();
  has_bias = with_bias();
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

LogicalResult tpu::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  int izp = 0;
  if (Quant::isUniformQuantized(input())) {
    izp = Quant::getUniformQuantizedType(input()).getZeroPoint();
  }
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu(), izp);
  p.handle = (void *)conv;
  return success();
}

void tpu::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  // requant
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (Quant::isUniformQuantized(output())) {
    int64_t n, c, h, w;
    auto sType = Module::getStorageType(output());
    Module::getNCHW(output(), n, c, h, w);
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto rshift_v = Module::getI64Array(rshift().getValue());
    auto multiplier_v = Module::getI64Array(multiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          auto v =
              MultiplyByQuantizedMultiplier((int32_t)p.outputs[0][offset],
                                            (int32_t)multi, (int32_t)shift) +
              o_qtype.getZeroPoint();
          p.outputs[0][offset] = sType.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                            : Quant::to_int8(v);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::ConvOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                     int64_t out_idx, int64_t out_slice) {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, do_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, do_relu);
  int kh_with_dh = (kh - 1) * dh + 1;
  in_slice = (out_slice - 1) * sh + (kh_with_dh >= sh ? kh_with_dh : sh);
  in_idx = out_idx * sh - pt;
  LocalGenInterface::fixSlice(in_idx, in_slice, ih);
  return success();
}
