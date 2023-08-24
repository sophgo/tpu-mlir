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

void GELULowering::LoweringF32(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::GELU));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void GELULowering::LoweringINT4(PatternRewriter &rewriter, top::GELUOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void GELULowering::LoweringINT8(PatternRewriter &rewriter, top::GELUOp op,
                                bool asymmetric) const {
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric, [](double val) {
        return 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
      });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void GELULowering::LoweringBF16(PatternRewriter &rewriter,
                                top::GELUOp op) const {
  if (module::isBM1686()) {
    auto op_ = op.getOperation();
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::GELU));
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  } else {
    LoweringF32(rewriter, op);
  }
}

void GELULowering::LoweringF16(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  if (module::isBM1686()) {
    auto op_ = op.getOperation();
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::GELU));
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  } else {
    LoweringF32(rewriter, op);
  }
}

void GELULowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::GELUOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
