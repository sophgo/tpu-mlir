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

void SigmoidLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  auto op_ = op.getOperation();
  bool log = op.getLog();
  if (log) {
    op_->setAttr("mode", tpu::ActiveModeAttr::get(
                             op.getContext(), tpu::ActiveMode::LOG_SIGMOID));
  } else {
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::SIGMOID));
  }
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SigmoidLowering::LoweringINT8(PatternRewriter &rewriter, top::SigmoidOp op,
                                   bool asymmetric) const {
  bool log = op.getLog();
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [&](double val) {
        return log ? std::log(1 / (1 + std::exp(-val)))
                   : 1 / (1 + std::exp(-val));
      },
      32);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684
} // namespace tpu_mlir
