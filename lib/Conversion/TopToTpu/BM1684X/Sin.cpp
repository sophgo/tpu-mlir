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

static void set_sin_attr(PatternRewriter &rewriter, top::SinOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SIN));
}

void SinLowering::LoweringF32(PatternRewriter &rewriter, top::SinOp op) const {
  set_sin_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void SinLowering::LoweringINT8(PatternRewriter &rewriter, top::SinOp op,
                               bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::sin(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SinLowering::LoweringINT4(PatternRewriter &rewriter, top::SinOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void SinLowering::LoweringBF16(PatternRewriter &rewriter, top::SinOp op) const {
  set_sin_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void SinLowering::LoweringF16(PatternRewriter &rewriter, top::SinOp op) const {
  set_sin_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void SinLowering::LoweringF8(PatternRewriter &rewriter, top::SinOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SinLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::SinOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir
