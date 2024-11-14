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

void DtypeCastLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::DtypeCastOp op) const {
  // auto round_mode = op.getRoundModeAttr().str();
  lowering_common<tpu::DtypeCastOp>(rewriter, op.getOperation(),
                                    op.getOutput().getType());
  //  newOp.setRoundModeAttr(
  //     tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}
void DtypeCastLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::DtypeCastOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void DtypeCastLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::DtypeCastOp op,
                                     bool asymmetric) const {
  // auto round_mode = op.getRoundModeAttr().str();
  lowering_common<tpu::DtypeCastOp>(rewriter, op.getOperation(),
                                    op.getOutput().getType());
  //  newOp.setRoundModeAttr(
  //     tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void DtypeCastLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::DtypeCastOp op) const {
  // auto round_mode = op.getRoundModeAttr().str();
  lowering_common<tpu::DtypeCastOp>(rewriter, op.getOperation(),
                                    op.getOutput().getType());
  //  newOp.setRoundModeAttr(
  //     tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void DtypeCastLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::DtypeCastOp op) const {
  // auto round_mode = op.getRoundModeAttr().str();
  lowering_common<tpu::DtypeCastOp>(rewriter, op.getOperation(),
                                    op.getOutput().getType());
  //  newOp.setRoundModeAttr(
  //     tpu::RoundModeAttr::get(op.getContext(), get_round_mode(round_mode)));
}

void DtypeCastLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::DtypeCastOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void DtypeCastLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::DtypeCastOp op) const {
  //   auto round_mode = op.getRoundModeAttr().str();
  //   if (module::isUniformQuantized(op.getInput(), op.getOutput()) == false) {
  //     auto newOp = lowering_common<tpu::DtypeCastOp>(rewriter,
  //     op.getOperation(),
  //                                  op.getOutput().getType());
  //     newOp.setRoundModeAttr(
  //       tpu::RoundModeAttr::get(op.getContext(),
  //       get_round_mode(round_mode)));
  //     return;
  //   }
  //   int64_t i_zeropoint, o_zeropoint;
  //   double i_scale, o_scale;
  //   module::getScaleAndZeroPoint(op.getInput(), i_scale, i_zeropoint, true);
  //   module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zeropoint, true);
  //   std::vector<NamedAttribute> attrs;
  //   if (i_scale == o_scale) {
  //     int zero_diff = i_zeropoint - o_zeropoint;
  //     attrs.push_back(rewriter.getNamedAttr(
  //         "const_val", rewriter.getF64FloatAttr(-zero_diff)));
  //     rewriter.replaceOpWithNewOp<tpu::AddConstOp>(
  //         op.getOperation(), op.getOutput().getType(), op.getInput(), attrs);
  //     return;
  //   }

  //   std::vector<Value> operands;
  //   if (i_zeropoint != 0) {
  //     auto sub_zp = do_binary_saclar<tpu::AddConstOp>(
  //         op.getInput(), rewriter.getI32Type(), -i_zeropoint);
  //     operands.push_back(sub_zp);
  //   } else {
  //     operands.push_back(op.getInput());
  //   }
  //   o_scale = i_scale / o_scale;
  //   int64_t multiplier;
  //   int64_t shift;
  //   QuantizeMultiplier(o_scale, &multiplier, &shift);
  //   auto ctx = op.getContext();
  //   attrs.push_back(rewriter.getNamedAttr(
  //       "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  //   attrs.push_back(
  //       rewriter.getNamedAttr("rshift",
  //       rewriter.getSI32IntegerAttr(-shift)));
  //   attrs.push_back(rewriter.getNamedAttr(
  //       "quant_mode",
  //       tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFLite_LShift)));
  //   attrs.push_back(rewriter.getNamedAttr("round_mode",
  //       tpu::RoundModeAttr::get(op.getContext(),
  //       get_round_mode(round_mode))));

  //   rewriter.replaceOpWithNewOp<tpu::RequantIntOp>(
  //       op.getOperation(), op.getOutput().getType(), operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
