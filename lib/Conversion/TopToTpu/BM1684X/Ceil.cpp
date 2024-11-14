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

void CeilLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CeilOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::CEIL));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void CeilLowering::LoweringINT8(PatternRewriter &rewriter, top::CeilOp op,
                                bool asymmetric) const {

  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::ceil(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void CeilLowering::LoweringINT4(PatternRewriter &rewriter, top::CeilOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CeilLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CeilOp ceilOp) const {
  auto op = ceilOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::CEIL));
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
}

void CeilLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CeilOp ceilOp) const {
  auto op = ceilOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::CEIL));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op);
}

void CeilLowering::LoweringF8(PatternRewriter &rewriter,
                              top::CeilOp ceilOp) const {
  llvm_unreachable("FIXME: not implement");
}

void CeilLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CeilOp ceilOp) const {
  LoweringINT8(rewriter, ceilOp, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
