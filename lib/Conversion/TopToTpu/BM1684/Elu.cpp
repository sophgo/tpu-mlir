//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void EluLowering::LoweringF32(PatternRewriter &rewriter, top::EluOp op) const {
  const double alpha = op.getAlpha().convertToDouble();
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::ELU));
  op_->setAttr("coeffs", rewriter.getF64ArrayAttr(ArrayRef<double>{alpha}));
  op_->removeAttr("alpha");
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void EluLowering::LoweringINT8(PatternRewriter &rewriter, top::EluOp op,
                               bool asymmetric) const {
  float alpha = op.getAlpha().convertToDouble();
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [alpha](double val) {
        return val > 0. ? val : alpha * (std::exp(val) - 1.);
      },
      32);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684
} // namespace tpu_mlir
