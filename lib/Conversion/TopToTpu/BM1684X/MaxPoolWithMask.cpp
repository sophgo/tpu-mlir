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

void MaxPoolWithMaskLowering::LoweringF32(PatternRewriter &rewriter,
                                          top::MaxPoolWithMaskOp op) const {
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}
void MaxPoolWithMaskLowering::LoweringINT4(PatternRewriter &rewriter, top::MaxPoolWithMaskOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MaxPoolWithMaskLowering::LoweringINT8(PatternRewriter &rewriter,
                                           top::MaxPoolWithMaskOp op,
                                           bool asymmetric) const {
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}

void MaxPoolWithMaskLowering::LoweringBF16(PatternRewriter &rewriter,
                                           top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolWithMaskLowering::LoweringF16(PatternRewriter &rewriter,
                                          top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolWithMaskLowering::LoweringF8(PatternRewriter &rewriter,
                                          top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolWithMaskLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
