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

// Custom lowering (same shape as FAttentionLse): the attention output follows
// the quant dtype (bf16/f16/f32), but lse is ALWAYS fp32 (the kernel stores
// fp32 logsumexp; sink scaling is numerically sensitive). lowering_common
// handles multi-result ops via a std::vector<Type> of result types; the
// weight-clone path keys off newTypes[0] (output) which is fine since
// FlexAttentionOp has no weight operands.
static void LoweringFlexAttention(PatternRewriter &rewriter,
                                  top::FlexAttentionOp op, Type out_type) {
  Type lse_type = op.getLse().getType(); // fp32
  lowering_common<tpu::FlexAttentionOp>(rewriter, op.getOperation(),
                                        std::vector<Type>{out_type, lse_type});
}

void FlexAttentionLowering::LoweringF32(PatternRewriter &rewriter,
                                        top::FlexAttentionOp op) const {
  LoweringFlexAttention(rewriter, op, op.getOutput().getType());
}

void FlexAttentionLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::FlexAttentionOp op,
                                         bool asymmetric) const {
  LoweringFlexAttention(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FlexAttentionLowering::LoweringINT4(PatternRewriter &rewriter,
                                         top::FlexAttentionOp op,
                                         bool asymmetric) const {
  LoweringFlexAttention(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FlexAttentionLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::FlexAttentionOp op) const {
  LoweringFlexAttention(rewriter, op, getQuantBF16Type(op.getOutput()));
}

void FlexAttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::FlexAttentionOp op) const {
  LoweringFlexAttention(rewriter, op, getQuantF16Type(op.getOutput()));
}

void FlexAttentionLowering::LoweringF8(PatternRewriter &rewriter,
                                       top::FlexAttentionOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void FlexAttentionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                              top::FlexAttentionOp op) const {
  LoweringFlexAttention(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
