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

static inline Operation *set_mode(top::SiLUOp op) {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SILU));
  return op_;
}

void SiLULowering::LoweringF32(PatternRewriter &rewriter,
                               top::SiLUOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SiLULowering::LoweringINT4(PatternRewriter &rewriter, top::SiLUOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op,
                                bool asymmetric) const {
  bool output_asym = op->hasAttr("output_asym");
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [](double val) { return val / (1 + std::exp(-val)); }, 8,
      tpu_mlir::ROUNDING_HALF_AWAY_FROM_ZERO, output_asym || asymmetric);
  auto newType = getQuantInt8Type(op.getOutput(), output_asym || asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SiLULowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SiLUOp op) const {
  if (module::isBM1690Family() || module::isMARS3() || module::isSGTPUV8()) {
    auto op_ = set_mode(op);
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  } else
    LoweringF32(rewriter, op);
}

void SiLULowering::LoweringF16(PatternRewriter &rewriter,
                               top::SiLUOp op) const {
  if (module::isBM1690Family()) {
    auto op_ = set_mode(op);
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  } else
    LoweringF32(rewriter, op);
}

void SiLULowering::LoweringF8(PatternRewriter &rewriter, top::SiLUOp op) const {
  if (module::isBM1690Family()) {
    auto op_ = set_mode(op);
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  } else
    LoweringF32(rewriter, op);
}

void SiLULowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SiLUOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir
