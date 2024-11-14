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

void SoftmaxBwdLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::SoftmaxBwdOp op) const {
  lowering_common_f32<tpu::SoftmaxBwdOp>(rewriter, op);
}

void SoftmaxBwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::SoftmaxBwdOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxBwdLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::SoftmaxBwdOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxBwdLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::SoftmaxBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SoftmaxBwdLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::SoftmaxBwdOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void SoftmaxBwdLowering::LoweringINT4(PatternRewriter &rewriter,
                                      top::SoftmaxBwdOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SoftmaxBwdLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::SoftmaxBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
