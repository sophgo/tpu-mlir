//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::AvgPoolOp::lowering_int8_bm1684x(bool asymmetric) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  auto op = getOperation();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(input());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);
  assert(in_zp == 0 && out_zp == 0);
  double scale = in_scale / (out_scale * kh * kw);
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getI64IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  builder.setInsertionPointAfter(op);
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp = builder.create<tpu::AvgPoolOp>(getLoc(), newType,
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::AvgPoolOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::AvgPoolOp>(getOperation());
}

Value top::AvgPoolOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::AvgPoolOp, BFloat16Type>(getOperation());
}

Value top::AvgPoolOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::AvgPoolOp, Float16Type>(getOperation());
}

Value top::AvgPoolOp::lowering_quant_bm1684x() {
  if (false == Quant::isUniformQuantized(input(), output())) {
    llvm_unreachable("input output should be quantized");
  }
  // input to f32
  Builder builder(getContext());
  auto in_f32 = do_cast(input(), builder.getF32Type(), false);
  auto op = getOperation();
  op->setOperand(0, in_f32);
  auto type = output().getType();
  auto v = lowering_common_float<tpu::AvgPoolOp>(getOperation());
  return do_cast(v, type, true);
}
