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

void FAttentionLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::FAttentionOp op) const {
  lowering_common_f32<tpu::FAttentionOp>(rewriter, op);
}

void FAttentionLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::FAttentionOp op,
                                      bool asymmetric) const {
  lowering_common_f16<tpu::FAttentionOp>(rewriter, op);
}

void FAttentionLowering::LoweringINT4(PatternRewriter &rewriter,
                                      top::FAttentionOp op,
                                      bool asymmetric) const {
  lowering_common_f16<tpu::FAttentionOp>(rewriter, op);
}

void FAttentionLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::FAttentionOp op) const {
  lowering_common_bf16<tpu::FAttentionOp>(rewriter, op);
}

void FAttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::FAttentionOp op) const {
  lowering_common_f16<tpu::FAttentionOp>(rewriter, op);
}

void FAttentionLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::FAttentionOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void FAttentionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::FAttentionOp op) const {
  lowering_common<tpu::FAttentionOp>(rewriter, op.getOperation(),
                                     op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
