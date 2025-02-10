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

static inline Operation *set_mode(top::SqrtOp op) {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SQRT));
  return op_;
}

void SqrtLowering::LoweringF32(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SqrtLowering::LoweringINT4(PatternRewriter &rewriter, top::SqrtOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SqrtLowering::LoweringINT8(PatternRewriter &rewriter, top::SqrtOp op,
                                bool asymmetric) const {
  auto table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                   [](double val) { return std::sqrt(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SqrtLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SqrtOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SqrtLowering::LoweringF16(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SqrtLowering::LoweringF8(PatternRewriter &rewriter, top::SqrtOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SqrtLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SqrtOp op) const {
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir
