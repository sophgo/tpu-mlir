//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

// Custom lowering: the attention output follows the quant dtype (bf16/f16/f32),
// but lse is ALWAYS fp32 (the kernel stores fp32 logsumexp; sink scaling is
// numerically sensitive). lowering_common handles multi-result ops via a
// std::vector<Type> of result types; the weight-clone path keys off
// newTypes[0] (output) which is fine since FAttentionLseOp has no weight
// operands.
static void LoweringFAttentionLse(PatternRewriter &rewriter,
                                  top::FAttentionLseOp op, Type out_type) {
  Type lse_type = op.getLse().getType(); // fp32
  lowering_common<tpu::FAttentionLseOp>(rewriter, op.getOperation(),
                                        std::vector<Type>{out_type, lse_type});
}

void FAttentionLseLowering::LoweringF32(PatternRewriter &rewriter,
                                        top::FAttentionLseOp op) const {
  LoweringFAttentionLse(rewriter, op, op.getOutput().getType());
}

void FAttentionLseLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::FAttentionLseOp op,
                                         bool asymmetric) const {
  LoweringFAttentionLse(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FAttentionLseLowering::LoweringINT4(PatternRewriter &rewriter,
                                         top::FAttentionLseOp op,
                                         bool asymmetric) const {
  LoweringFAttentionLse(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FAttentionLseLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::FAttentionLseOp op) const {
  LoweringFAttentionLse(rewriter, op, getQuantBF16Type(op.getOutput()));
}

void FAttentionLseLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::FAttentionLseOp op) const {
  LoweringFAttentionLse(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FAttentionLseLowering::LoweringF8(PatternRewriter &rewriter,
                                       top::FAttentionLseOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void FAttentionLseLowering::LoweringQuantized(PatternRewriter &rewriter,
                                              top::FAttentionLseOp op) const {
  LoweringFAttentionLse(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
