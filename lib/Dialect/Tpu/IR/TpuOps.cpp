//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include <numeric>

using namespace mlir;
using namespace sophgo::tpu;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "sophgo/Dialect/Tpu/IR/TpuOpsDialect.cpp.inc"

void TpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sophgo/Dialect/Tpu/IR/TpuOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tpu Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sophgo/Dialect/Tpu/IR/TpuOps.cpp.inc"

void ConvOp::parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw,
                              int64_t &oc, int64_t &oh, int64_t &ow, int64_t &g,
                              int64_t &kh, int64_t &kw, int64_t &ins_h,
                              int64_t &ins_w, int64_t &sh, int64_t &sw,
                              int64_t &pt, int64_t &pb, int64_t &pl,
                              int64_t &pr, int64_t &dh, int64_t &dw,
                              bool &is_dw, bool &with_bias, bool &do_relu) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto k_s = filter().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  do_relu = this->do_relu();
  with_bias = !bias().getType().isa<mlir::NoneType>();
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();
  dh = dilations().getValue()[0].cast<IntegerAttr>().getInt();
  dw = dilations().getValue()[1].cast<IntegerAttr>().getInt();
  g = group();
  is_dw = (oc == ic && oc == g);
  return;
}

void MaxPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih,
                                 int64_t &iw, int64_t &oh, int64_t &ow,
                                 int64_t &kh, int64_t &kw, int64_t &sh,
                                 int64_t &sw, int64_t &pt, int64_t &pb,
                                 int64_t &pl, int64_t &pr, int64_t &pad_value,
                                 bool &is_global, bool &count_include_pad) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();

  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();

  size_t num_dims = i_s.size();
  assert(num_dims == 4); // 4 dims now
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
}

void AvgPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih,
                                 int64_t &iw, int64_t &oh, int64_t &ow,
                                 int64_t &kh, int64_t &kw, int64_t &sh,
                                 int64_t &sw, int64_t &pt, int64_t &pb,
                                 int64_t &pl, int64_t &pr, int64_t &pad_value,
                                 bool &is_global, bool &count_include_pad) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();

  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();

  size_t num_dims = i_s.size();
  assert(num_dims == 4); // 4 dims now
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
}

void MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                                int64_t &N) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto r_s = right().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  auto r_dims = r_s.size();
  auto i_dims = i_s.size();
  N = r_s[r_dims - 1];
  K = r_s[r_dims - 2];
  if (r_dims > 2) {
    M = i_s[i_dims - 2];
    assert(i_s[i_dims - 1] == K);
    batch = std::accumulate(r_s.begin(), r_s.begin() + r_dims - 2, 1,
                            std::multiplies<int64_t>());
  } else {
    batch = 1;
    M = std::accumulate(i_s.begin(), i_s.begin() + i_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}
