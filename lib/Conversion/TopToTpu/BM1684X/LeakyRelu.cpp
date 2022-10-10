//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void LeakyReluLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::LeakyReluOp op) const {
  lowering_common_float<tpu::LeakyReluOp>(rewriter, op);
}

void LeakyReluLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {
  if (asymmetric) {
    lowering_common_float<tpu::LeakyReluOp>(rewriter, op);
  } else {
    int multiplier, rshift;
    get_scale_and_shift(op.alpha().convertToDouble(), multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));

    auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType,
                                                  Value(op.input()), attrs);
  }
}

void LeakyReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LeakyReluOp op) const {
  lowering_common_float<tpu::LeakyReluOp, BFloat16Type>(rewriter, op);
}

void LeakyReluLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::LeakyReluOp op) const {
  lowering_common_float<tpu::LeakyReluOp, Float16Type>(rewriter, op);
}

void LeakyReluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::LeakyReluOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
