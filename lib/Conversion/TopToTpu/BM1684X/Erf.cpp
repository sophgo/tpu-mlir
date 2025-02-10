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

void ErfLowering::LoweringF32(PatternRewriter &rewriter, top::ErfOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::ERF));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void ErfLowering::LoweringINT4(PatternRewriter &rewriter, top::ErfOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ErfLowering::LoweringINT8(PatternRewriter &rewriter, top::ErfOp op,
                               bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::erf(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void ErfLowering::LoweringBF16(PatternRewriter &rewriter, top::ErfOp op) const {
  if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
      module::isSGTPUV8()) {
    auto op_ = op.getOperation();
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::ERF));
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  } else {
    LoweringF32(rewriter, op);
  }
}

void ErfLowering::LoweringF16(PatternRewriter &rewriter, top::ErfOp op) const {
  if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
      module::isSGTPUV8()) {
    auto op_ = op.getOperation();
    op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::ERF));
    lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
  } else {
    LoweringF32(rewriter, op);
  }
}

void ErfLowering::LoweringF8(PatternRewriter &rewriter, top::ErfOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ErfLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ErfOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
