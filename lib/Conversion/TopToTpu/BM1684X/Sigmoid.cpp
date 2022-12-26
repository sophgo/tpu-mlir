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

void SigmoidLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::SIGMOID));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SigmoidLowering::LoweringINT4(PatternRewriter &rewriter, top::SigmoidOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SigmoidLowering::LoweringINT8(PatternRewriter &rewriter, top::SigmoidOp op,
                                   bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  Value table =
      create_lookup_table(op.input(), op.output(), asymmetric,
                          [](double val) { return 1 / (1 + std::exp(-val)); });
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
}

void SigmoidLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SigmoidOp op) const {
  LoweringF32(rewriter, op);
}

void SigmoidLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  LoweringF32(rewriter, op);
}

void SigmoidLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SigmoidOp op) const {
  auto stype = Module::getStorageType(op.output());
  Value table =
      create_lookup_table(op.input(), op.output(), true,
                          [](double val) { return 1 / (1 + std::exp(-val)); });
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.output().getType(),
                                          ValueRange{op.input(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
