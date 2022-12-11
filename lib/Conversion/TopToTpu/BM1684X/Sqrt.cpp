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

void SqrtLowering::LoweringF32(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SQRT));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SqrtLowering::LoweringINT8(PatternRewriter &rewriter, top::SqrtOp op,
                                bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  auto table = create_lookup_table(op.input(), op.output(), asymmetric,
                                   [](double val) { return std::sqrt(val); });
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
}

void SqrtLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SqrtOp op) const {
  LoweringF32(rewriter, op);
}

void SqrtLowering::LoweringF16(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  LoweringF32(rewriter, op);
}

void SqrtLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SqrtOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
