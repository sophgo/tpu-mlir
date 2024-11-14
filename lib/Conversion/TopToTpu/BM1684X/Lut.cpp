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

void LutLowering::LoweringINT8(PatternRewriter &rewriter, top::LutOp op,
                               bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringINT4(PatternRewriter &rewriter, top::LutOp op,
                               bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringF32(PatternRewriter &rewriter, top::LutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringBF16(PatternRewriter &rewriter, top::LutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringF16(PatternRewriter &rewriter, top::LutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringF8(PatternRewriter &rewriter, top::LutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LutLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::LutOp op) const {
  rewriter.replaceOpWithNewOp<tpu::LutOp>(
      op, op.getOutput().getType(), ValueRange{op.getInput(), op.getTable()});
}

} // namespace bm1684x
} // namespace tpu_mlir
