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

static void set_elu_attr(PatternRewriter &rewriter, top::EluOp op) {
  const double alpha_ = op.getAlpha().convertToDouble();
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::ELU));
  op_->setAttr("coeffs", rewriter.getF64ArrayAttr(ArrayRef<double>{alpha_}));
  op_->removeAttr("alpha");
}

void EluLowering::LoweringF32(PatternRewriter &rewriter, top::EluOp op) const {
  set_elu_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

static inline double elu(double x, double alpha) {
  return x > 0 ? x : alpha * (std::exp(x) - 1);
}
void EluLowering::LoweringINT4(PatternRewriter &rewriter, top::EluOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void EluLowering::LoweringINT8(PatternRewriter &rewriter, top::EluOp op,
                               bool asymmetric) const {
  const double alpha_ = op.getAlpha().convertToDouble();
  Value table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [alpha_](double val) { return elu(val, alpha_); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}
void EluLowering::LoweringBF16(PatternRewriter &rewriter, top::EluOp op) const {
  set_elu_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}
void EluLowering::LoweringF16(PatternRewriter &rewriter, top::EluOp op) const {
  set_elu_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}
void EluLowering::LoweringF8(PatternRewriter &rewriter, top::EluOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void EluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::EluOp op) const {
  LoweringINT8(rewriter, op, true);
}
} // namespace bm1684x
} // namespace tpu_mlir
