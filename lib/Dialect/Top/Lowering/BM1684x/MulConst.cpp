//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::MulConstOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                            bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  double scale_i, scale_o;
  int64_t zp_i, zp_o;
  Quant::getScaleAndZeroPoint(input(), scale_i, zp_i, asymmetric);
  Quant::getScaleAndZeroPoint(output(), scale_o, zp_o, asymmetric);
  auto scale = scale_i / scale_o * const_val().convertToDouble();
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType, ValueRange{input()},
                                               attrs);
}

void top::MulConstOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulConstOp, Float32Type>(rewriter, getOperation());
}

void top::MulConstOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulConstOp, BFloat16Type>(rewriter,
                                                       getOperation());
}

void top::MulConstOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulConstOp, Float16Type>(rewriter, getOperation());
}

void top::MulConstOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}
