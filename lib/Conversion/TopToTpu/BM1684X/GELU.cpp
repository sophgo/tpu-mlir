//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/ActiveUtils.h"

namespace tpu_mlir {
namespace bm1684x {

static inline tpu::ActiveMode get_active_mode(top::GELUOp op) {
  auto active_mode = tpu::ActiveMode::GELU;
  auto approx_mode = op.getApproxMode();
  if (approx_mode == "tanh")
    active_mode = tpu::ActiveMode::TGELU;
  else if (approx_mode == "sigm")
    active_mode = tpu::ActiveMode::QGELU;
  return active_mode;
}

static inline Operation* update_attr(top::GELUOp op) {
  auto active_mode = get_active_mode(op);
  op->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), active_mode));
  op->removeAttr("approx_mode");
  return op.getOperation();
}

void GELULowering::LoweringF32(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  auto op_ = update_attr(op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void GELULowering::LoweringINT4(PatternRewriter &rewriter, top::GELUOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void GELULowering::LoweringINT8(PatternRewriter &rewriter, top::GELUOp op,
                                bool asymmetric) const {
  bool output_asym = op->hasAttr("output_asym");
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      getActivateFunc(get_active_mode(op), nullptr), 8,
      tpu_mlir::ROUNDING_HALF_AWAY_FROM_ZERO,
      output_asym);
  auto newType = getQuantInt8Type(op.getOutput(), output_asym);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void GELULowering::LoweringBF16(PatternRewriter &rewriter,
                                top::GELUOp op) const {
  if (module::isBM1684X()) {
    LoweringF32(rewriter, op);
  } else {
    auto op_ = update_attr(op);
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  }
}

void GELULowering::LoweringF16(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  if (module::isBM1684X()) {
    LoweringF32(rewriter, op);
  } else {
    auto op_ = update_attr(op);
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  }
}

void GELULowering::LoweringF8(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void GELULowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::GELUOp op) const {
  auto round_mode = round_mode_convert(get_round_mode(op.getRoundModeAttr().str()));
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), true, getActivateFunc(get_active_mode(op), nullptr), 8, round_mode);
  auto newType = getQuantInt8Type(op.getOutput(), true);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
