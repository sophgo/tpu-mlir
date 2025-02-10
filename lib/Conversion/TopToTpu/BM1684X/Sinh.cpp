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

static void set_sinh_attr(PatternRewriter &rewriter, top::SinhOp op) {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SINH));
}

void SinhLowering::LoweringF32(PatternRewriter &rewriter,
                               top::SinhOp op) const {
  set_sinh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void SinhLowering::LoweringINT8(PatternRewriter &rewriter, top::SinhOp op,
                                bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::sinh(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SinhLowering::LoweringINT4(PatternRewriter &rewriter, top::SinhOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void SinhLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SinhOp op) const {
  set_sinh_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void SinhLowering::LoweringF16(PatternRewriter &rewriter,
                               top::SinhOp op) const {
  set_sinh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void SinhLowering::LoweringF8(PatternRewriter &rewriter, top::SinhOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SinhLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SinhOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir
