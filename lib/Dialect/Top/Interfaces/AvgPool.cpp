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
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::AvgPoolOp::getFLOPs() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool has_relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             has_relu, is_global, count_include_pad);
  return Module::getNumElements(output()) * (kh * kw + has_relu ? 1 : 0);
}

void top::AvgPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih,
                                int64_t &iw, int64_t &oh, int64_t &ow,
                                int64_t &kh, int64_t &kw, int64_t &sh,
                                int64_t &sw, int64_t &pt, int64_t &pb,
                                int64_t &pl, int64_t &pr, int64_t &pad_value,
                                bool &relu, bool &is_global,
                                bool &count_include_pad) {
  Module::getNCHW(input(), n, c, ih, iw);
  int64_t on,oc;
  Module::getNCHW(output(), on,oc,oh,ow);
  assert(on == n && oc == c);
  auto kernel = Module::getI64Array(kernel_shape());
  kh = kernel->at(0);
  kw = kernel->at(1);
  auto stride = Module::getI64Array(strides());
  sh = stride->at(0);
  sw = stride->at(1);
  relu = do_relu();
  auto pad = Module::getI64Array(pads());
  pt = pad->at(0);
  pl = pad->at(1);
  pb = pad->at(2);
  pr = pad->at(3);
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
  if (pt == 0 && pb == 0 && pl == 0 && pr == 0) {
    // no pad
    count_include_pad = true;
  }
}

LogicalResult top::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad, relu;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  pooling->setup(p.inputs[0], p.outputs[0], n, c, ih, iw, oh, ow, kh, kw, sh,
                 sw, pt, pb, pl, pr, true, count_include_pad, pad_value);
  p.handle = (void *)pooling;
  return success();
}

void top::AvgPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::AvgPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    function_relu(p.outputs[0], p.outputs[0], Module::getNumElements(output()));
  }
  return success();
}
