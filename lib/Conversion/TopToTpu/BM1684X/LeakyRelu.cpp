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
  lowering_common_f32<tpu::LeakyReluOp>(rewriter, op);
}
void LeakyReluLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void LeakyReluLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {
  if (asymmetric) {
    LoweringF16(rewriter, op);
  } else {
    int multiplier, rshift;
    get_scale_and_shift(op.getAlpha().convertToDouble(), multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));

    auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType,
                                                  Value(op.getInput()), attrs);
  }
}

void LeakyReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LeakyReluOp op) const {
  lowering_common_bf16<tpu::LeakyReluOp>(rewriter, op);
}

void LeakyReluLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::LeakyReluOp op) const {
  lowering_common_f16<tpu::LeakyReluOp>(rewriter, op);
}

void LeakyReluLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::LeakyReluOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LeakyReluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::LeakyReluOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  int multiplier, rshift;
  get_scale_and_shift(op.getAlpha().convertToDouble(), multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  auto round_mode = op.getRoundModeAttr().str();
  attrs.push_back(rewriter.getNamedAttr(
      "round_mode",
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode))));

  rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, op.getOutput().getType(),
                                                Value(op.getInput()), attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
