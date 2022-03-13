//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/QuantizeInterface.h"
#include "sophgo/Support/DnnlConv.h"
#include "sophgo/Support/DnnlPool.h"
#include "sophgo/Support/DnnlMatMul.h"
#include "sophgo/Support/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"

using namespace sophgo;
using namespace mlir;

void tops::ConvOp::quantize_int8() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  // auto filterOp = cast<tops::WeightOp>(filter().getDefiningOp());
  // auto biasOp = cast<tops::WeightOp>(bias().getDefiningOp());
  // auto filter = filterOp.read<float>();
  // auto bias = biasOp.read<float>();
  // auto type = output().getType().cast<RankedTensorType>();
  // auto fmax = findMaxabs(filter->data(), filter->size());
  // to be continue (pengchao.hu)
}

void tops::ReluOp::quantize_int8() {
  llvm_unreachable("Unsupport now");
}

void tops::AddOp::quantize_int8() {
}

void tops::MaxPoolOp::quantize_int8() {

}

void tops::AvgPoolOp::quantize_int8() {
}

void tops::ReshapeOp::quantize_int8() {
}

void tops::MatMulOp::quantize_int8() {
}

