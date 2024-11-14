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

void GELULowering::LoweringF32(PatternRewriter &rewriter,
                               top::GELUOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::GELU));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

// 0.5∗x∗(1+Tanh(sqrt(2/π)∗(x+0.044715∗x^3)))
void GELULowering::LoweringINT8(PatternRewriter &rewriter, top::GELUOp op,
                                bool asymmetric) const {
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [](double x) {
        return 0.5 * x *
               (1 + std::tanh(std::sqrt(2.0 / M_PI) *
                              (x + 0.044715 * std::pow(x, 3))));
      },
      32);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684
} // namespace tpu_mlir
