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

void SoftsignLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::SoftsignOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::SOFT_SIGN));

  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SoftsignLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::SoftsignOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SoftsignLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::SoftsignOp op, bool asymmetric) const {
  Value table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [](double val) { return val / (1 + std::abs(val)); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SoftsignLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::SoftsignOp op) const {
  LoweringF32(rewriter, op);
}

void SoftsignLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::SoftsignOp op) const {
  LoweringF32(rewriter, op);
}

void SoftsignLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::SoftsignOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SoftsignLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::SoftsignOp op) const {
  Value table =
      create_lookup_table(op.getInput(), op.getOutput(), true,
                          [](double val) { return val / (1 + std::abs(val)); });
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
