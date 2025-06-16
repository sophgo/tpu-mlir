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

static void set_swish_attr(PatternRewriter &rewriter, top::SwishOp op) {
  auto op_ = op.getOperation();
  const double beta_ = op.getBeta().convertToDouble();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::SWISH));
  op_->setAttr("coeffs", rewriter.getF64ArrayAttr(ArrayRef<double>{beta_}));
  op_->removeAttr("beta");
  op_->removeAttr("round_mode");
}

void SwishLowering::LoweringF32(PatternRewriter &rewriter,
                                top::SwishOp op) const {
  set_swish_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}
void SwishLowering::LoweringINT4(PatternRewriter &rewriter, top::SwishOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SwishLowering::LoweringINT8(PatternRewriter &rewriter, top::SwishOp op,
                                 bool asymmetric) const {
  auto beta = op.getBeta().convertToDouble();
  auto round_mode =
      round_mode_convert(get_round_mode(op.getRoundModeAttr().str()));
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [beta](double val) { return val / (1 + std::exp(-val * beta)); }, 8,
      round_mode);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SwishLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::SwishOp op) const {
  LoweringF32(rewriter, op);
}

void SwishLowering::LoweringF16(PatternRewriter &rewriter,
                                top::SwishOp op) const {
  LoweringF32(rewriter, op);
}

void SwishLowering::LoweringF8(PatternRewriter &rewriter,
                               top::SwishOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SwishLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::SwishOp op) const {
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
