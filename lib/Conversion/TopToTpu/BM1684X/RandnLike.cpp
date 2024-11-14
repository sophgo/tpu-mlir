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

void RandnLikeLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::RandnLikeOp op) const {
  lowering_common_f32<tpu::RandnLikeOp>(rewriter, op);
}
void RandnLikeLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::RandnLikeOp op,
                                     bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RandnLikeLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::RandnLikeOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RandnLikeLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::RandnLikeOp op) const {
  LoweringF32(rewriter, op);
}

void RandnLikeLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::RandnLikeOp op) const {
  LoweringF32(rewriter, op);
}

void RandnLikeLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::RandnLikeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RandnLikeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::RandnLikeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
