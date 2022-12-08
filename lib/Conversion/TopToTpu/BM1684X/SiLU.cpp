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

void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op,
                                bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  auto table =
      create_lookup_table(op.input(), op.output(), asymmetric, [](double val) {
        return val / (1 + std::exp(-val));
      });
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table}, attrs);
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
