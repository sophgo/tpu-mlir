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

void AbsLowering::LoweringF32(PatternRewriter &rewriter, top::AbsOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::ABSVAL));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void AbsLowering::LoweringINT8(PatternRewriter &rewriter, top::AbsOp absOp,
                               bool asymmetric) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common_int8<tpu::ActiveOp>(rewriter, op, asymmetric);
}

void AbsLowering::LoweringINT4(PatternRewriter &rewriter, top::AbsOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void AbsLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::AbsOp absOp) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
}

void AbsLowering::LoweringF16(PatternRewriter &rewriter,
                              top::AbsOp absOp) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op);
}

void AbsLowering::LoweringF8(PatternRewriter &rewriter,
                             top::AbsOp absOp) const {
  // llvm_unreachable("FIXME: not implement");
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  if (module::getMode() == module::Mode::F8E4M3) {
    lowering_common_f8<tpu::ActiveOp>(rewriter, op, isE4);
  } else if (module::getMode() == module::Mode::F8E5M2) {
    lowering_common_f8<tpu::ActiveOp>(rewriter, op, isE4);
  }
}
void AbsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::AbsOp absOp) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common<tpu::ActiveOp>(rewriter, op, op->getResult(0).getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
