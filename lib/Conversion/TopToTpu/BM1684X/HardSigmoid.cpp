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

static void set_hsigmoid_attr(PatternRewriter &rewriter,
                              top::HardSigmoidOp op) {
  const double beta_ = op.getBeta().convertToDouble();
  const double alpha_ = op.getAlpha().convertToDouble();
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::HSIGMOID));
  op_->setAttr("coeffs",
               rewriter.getF64ArrayAttr(ArrayRef<double>{beta_, alpha_}));
  op_->removeAttr("alpha");
  op_->removeAttr("beta");
}

void HardSigmoidLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::HardSigmoidOp op) const {
  set_hsigmoid_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

static inline double hsigmoid(double x, double alpha, double beta) {
  return std::max(0.0, std::min(1.0, alpha * x + beta));
}
void HardSigmoidLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::HardSigmoidOp op,
                                       bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void HardSigmoidLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::HardSigmoidOp op,
                                       bool asymmetric) const {
  const double beta_ = op.getBeta().convertToDouble();
  const double alpha_ = op.getAlpha().convertToDouble();
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [alpha_, beta_](double val) { return hsigmoid(val, alpha_, beta_); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void HardSigmoidLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::HardSigmoidOp op) const {
  set_hsigmoid_attr(rewriter, op); // TODO: supports bf16
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
}

void HardSigmoidLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::HardSigmoidOp op) const {
  set_hsigmoid_attr(rewriter, op); // TODO: supports f16
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void HardSigmoidLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::HardSigmoidOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void HardSigmoidLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::HardSigmoidOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
