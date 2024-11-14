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

void CastLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CastOp op) const {
  auto round_mode = op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}
void CastLowering::LoweringINT4(PatternRewriter &rewriter, top::CastOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void CastLowering::LoweringINT8(PatternRewriter &rewriter, top::CastOp op,
                                bool asymmetric) const {
  auto round_mode = op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void CastLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CastOp op) const {
  auto round_mode = op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void CastLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CastOp op) const {
  std::string round_mode =
      module::isTrain() ? "HalfAwayFromZero" : op.getRoundModeAttr().str();
  auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                            op.getOutput().getType());
  newOp.setRoundModeAttr(
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void CastLowering::LoweringF8(PatternRewriter &rewriter, top::CastOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void CastLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CastOp op) const {
  auto round_mode = op.getRoundModeAttr().str();
  if (module::isUniformQuantized(op.getInput(), op.getOutput()) == false) {
    auto newOp = lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                              op.getOutput().getType());
    newOp.setRoundModeAttr(
        tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
    return;
  }
  int64_t i_zeropoint, o_zeropoint;
  double i_scale, o_scale;
  module::getScaleAndZeroPoint(op.getInput(), i_scale, i_zeropoint, true);
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zeropoint, true);
  std::vector<NamedAttribute> attrs;
  if (i_scale == o_scale) {
    int zero_diff = i_zeropoint - o_zeropoint;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(-zero_diff)));
    rewriter.replaceOpWithNewOp<tpu::AddConstOp>(
        op.getOperation(), op.getOutput().getType(), op.getInput(), attrs);
    return;
  }

  std::vector<Value> operands;
  if (i_zeropoint != 0) {
    auto sub_zp = do_binary_saclar<tpu::AddConstOp>(
        op.getInput(), rewriter.getI32Type(), -i_zeropoint);
    operands.push_back(sub_zp);
  } else {
    operands.push_back(op.getInput());
  }
  o_scale = i_scale / o_scale;
  int64_t multiplier;
  int64_t shift;
  QuantizeMultiplier(o_scale, &multiplier, &shift);
  auto ctx = op.getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(-shift)));
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFLite_LShift)));
  attrs.push_back(rewriter.getNamedAttr(
      "round_mode",
      tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode))));

  rewriter.replaceOpWithNewOp<tpu::RequantIntOp>(
      op.getOperation(), op.getOutput().getType(), operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
