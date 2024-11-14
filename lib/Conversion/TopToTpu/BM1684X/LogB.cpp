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

void LogBLowering::LoweringF32(PatternRewriter &rewriter,
                               top::LogBOp op) const {
  auto op_ = op.getOperation();
  int base = op.getBase();
  if (base == 2) {
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::LOG2));
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
  }
}
void LogBLowering::LoweringINT4(PatternRewriter &rewriter, top::LogBOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void LogBLowering::LoweringINT8(PatternRewriter &rewriter, top::LogBOp op,
                                bool asymmetric) const {
  int base = op.getBase();
  if (base == 2) {
    Value table =
        create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                            [](double val) { return std::log2(val); });
    auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                            ValueRange{op.getInput(), table});
  }
}

void LogBLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::LogBOp op) const {
  if (!(module::isBM1688() || module::isSG2380())) {
    LoweringF32(rewriter, op);
    return;
  }
  auto op_ = op.getOperation();
  int base = op.getBase();
  if (base == 2) {
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::LOG2));
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  }
}

void LogBLowering::LoweringF16(PatternRewriter &rewriter,
                               top::LogBOp op) const {
  if (!(module::isBM1688() || module::isSG2380())) {
    LoweringF32(rewriter, op);
    return;
  }
  auto op_ = op.getOperation();
  int base = op.getBase();
  if (base == 2) {
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::LOG2));
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  }
}

void LogBLowering::LoweringF8(PatternRewriter &rewriter, top::LogBOp op) const {
  llvm_unreachable("not implement");
}

void LogBLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::LogBOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
