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

static inline Operation *set_mode(top::SigmoidOp op) {
  auto op_ = op.getOperation();
  bool log = op.getLog();
  auto active_mode =
      log ? tpu::ActiveMode::LOG_SIGMOID : tpu::ActiveMode::SIGMOID;
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(), active_mode));
  return op_;
}

void SigmoidLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void SigmoidLowering::LoweringINT4(PatternRewriter &rewriter, top::SigmoidOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SigmoidLowering::LoweringINT8(PatternRewriter &rewriter, top::SigmoidOp op,
                                   bool asymmetric) const {
  bool log = op.getLog();
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric, [&](double val) {
        return log ? std::log(1 / (1 + std::exp(-val)))
                   : 1 / (1 + std::exp(-val));
      });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SigmoidLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SigmoidOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SigmoidLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  auto op_ = set_mode(op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SigmoidLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::SigmoidOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SigmoidLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SigmoidOp op) const {
  bool log = op.getLog();
  auto round_mode =
      round_mode_convert(get_round_mode(op.getRoundModeAttr().str()));
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), true,
      [&](double val) {
        return log ? std::log(1 / (1 + std::exp(-val)))
                   : 1 / (1 + std::exp(-val));
      },
      8, round_mode);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
