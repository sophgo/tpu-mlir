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

static void set_tanh_attr(PatternRewriter &rewriter, top::TanhOp op) {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::TANH));
}

void TanhLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TanhOp op) const {
  set_tanh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanhLowering::LoweringINT8(PatternRewriter &rewriter, top::TanhOp op,
                                bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::tanh(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void TanhLowering::LoweringINT4(PatternRewriter &rewriter, top::TanhOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void TanhLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TanhOp op) const {
  set_tanh_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void TanhLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TanhOp op) const {
  set_tanh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanhLowering::LoweringF8(PatternRewriter &rewriter, top::TanhOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void TanhLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TanhOp op) const {
  auto round_mode =
      round_mode_convert(get_round_mode(op.getRoundModeAttr().str()));
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), true,
      [](double val) { return std::tanh(val); }, 8, round_mode);
  auto newType = getQuantInt8Type(op.getOutput(), true);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
