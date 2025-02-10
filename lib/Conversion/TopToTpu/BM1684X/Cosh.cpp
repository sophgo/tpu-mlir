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

static void set_cosh_attr(PatternRewriter &rewriter, top::CoshOp op) {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::COSH));
}

void CoshLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CoshOp op) const {
  set_cosh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void CoshLowering::LoweringINT8(PatternRewriter &rewriter, top::CoshOp op,
                                bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::cosh(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void CoshLowering::LoweringINT4(PatternRewriter &rewriter, top::CoshOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CoshLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CoshOp op) const {
  set_cosh_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void CoshLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CoshOp op) const {
  set_cosh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void CoshLowering::LoweringF8(PatternRewriter &rewriter, top::CoshOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void CoshLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CoshOp op) const {
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
