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

void BatchNormBwdLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::BatchNormBwdOp op) const {
  rewriter.replaceOpWithNewOp<tpu::BatchNormBwdOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}

void BatchNormBwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op) const {
  LoweringF32(rewriter, op);
}

void BatchNormBwdLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::BatchNormBwdOp op) const {
  LoweringF32(rewriter, op);
}

void BatchNormBwdLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::BatchNormBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::BatchNormBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
