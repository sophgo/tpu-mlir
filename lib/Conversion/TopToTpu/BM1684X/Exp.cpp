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

void ExpLowering::LoweringF32(PatternRewriter &rewriter, top::ExpOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void ExpLowering::LoweringINT4(PatternRewriter &rewriter, top::ExpOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ExpLowering::LoweringINT8(PatternRewriter &rewriter, top::ExpOp op,
                               bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::exp(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void ExpLowering::LoweringBF16(PatternRewriter &rewriter, top::ExpOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP));
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
}

void ExpLowering::LoweringF16(PatternRewriter &rewriter, top::ExpOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op);
}

void ExpLowering::LoweringF8(PatternRewriter &rewriter, top::ExpOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ExpLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ExpOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
