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

void SiLULowering::LoweringF32(PatternRewriter &rewriter,
                               top::SiLUOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SILU));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SiLULowering::LoweringINT4(PatternRewriter &rewriter, top::SiLUOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op,
                                bool asymmetric) const {
  auto stype = module::getStorageType(op.getOutput());
  auto table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric, [](double val) {
        return val / (1 + std::exp(-val));
      });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SiLULowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SiLUOp op) const {
  LoweringF32(rewriter, op);
}

void SiLULowering::LoweringF16(PatternRewriter &rewriter,
                               top::SiLUOp op) const {
  LoweringF32(rewriter, op);
}

void SiLULowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SiLUOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
