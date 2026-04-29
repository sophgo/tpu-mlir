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

void RecurrentGatedDeltaRuleLowering::LoweringF32(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op) const {
  lowering_common_f32<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op);
}

void RecurrentGatedDeltaRuleLowering::LoweringINT8(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op,
    bool asymmetric) const {
  lowering_common_f16<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op);
}

void RecurrentGatedDeltaRuleLowering::LoweringINT4(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op,
    bool asymmetric) const {
  lowering_common_f16<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op);
}

void RecurrentGatedDeltaRuleLowering::LoweringBF16(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op) const {
  lowering_common_bf16<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op);
}

void RecurrentGatedDeltaRuleLowering::LoweringF16(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op) const {
  lowering_common_f16<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op);
}

void RecurrentGatedDeltaRuleLowering::LoweringF8(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RecurrentGatedDeltaRuleLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::RecurrentGatedDeltaRuleOp op) const {
  lowering_common<tpu::RecurrentGatedDeltaRuleOp>(rewriter, op.getOperation(),
                                                  op.getAttnOut().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
