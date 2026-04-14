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

void ChunkGatedDeltaRuleLowering::LoweringF32(
    PatternRewriter &rewriter, top::ChunkGatedDeltaRuleOp op) const {
  lowering_common_f32<tpu::ChunkGatedDeltaRuleOp>(rewriter, op);
}

void ChunkGatedDeltaRuleLowering::LoweringINT8(PatternRewriter &rewriter,
                                               top::ChunkGatedDeltaRuleOp op,
                                               bool asymmetric) const {
  lowering_common_f16<tpu::ChunkGatedDeltaRuleOp>(rewriter, op);
}

void ChunkGatedDeltaRuleLowering::LoweringINT4(PatternRewriter &rewriter,
                                               top::ChunkGatedDeltaRuleOp op,
                                               bool asymmetric) const {
  lowering_common_f16<tpu::ChunkGatedDeltaRuleOp>(rewriter, op);
}

void ChunkGatedDeltaRuleLowering::LoweringBF16(
    PatternRewriter &rewriter, top::ChunkGatedDeltaRuleOp op) const {
  lowering_common_bf16<tpu::ChunkGatedDeltaRuleOp>(rewriter, op);
}

void ChunkGatedDeltaRuleLowering::LoweringF16(
    PatternRewriter &rewriter, top::ChunkGatedDeltaRuleOp op) const {
  lowering_common_f16<tpu::ChunkGatedDeltaRuleOp>(rewriter, op);
}

void ChunkGatedDeltaRuleLowering::LoweringF8(
    PatternRewriter &rewriter, top::ChunkGatedDeltaRuleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ChunkGatedDeltaRuleLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::ChunkGatedDeltaRuleOp op) const {
  lowering_common<tpu::ChunkGatedDeltaRuleOp>(rewriter, op.getOperation(),
                                              op.getAttnOut().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
