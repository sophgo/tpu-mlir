//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::LeakyReluOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                             bool asymmetric) {
  if (!asymmetric) {
    auto op = getOperation();
    Value input = op->getOperand(0);

    int multiplier, rshift;
    get_scale_and_shift(alpha().convertToDouble(), multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));

    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType, Value(input),
                                                  attrs);
  } else {
    lowering_f32_bm1684x(rewriter);
  }
}

void top::LeakyReluOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LeakyReluOp, Float32Type>(rewriter,
                                                       getOperation());
}

void top::LeakyReluOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LeakyReluOp, BFloat16Type>(rewriter,
                                                        getOperation());
}

void top::LeakyReluOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LeakyReluOp, Float16Type>(rewriter,
                                                       getOperation());
}

void top::LeakyReluOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not supported now");
}
