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

void SplitLowering::LoweringINT8(PatternRewriter &rewriter, top::SplitOp unpackOp,
                               bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void SplitLowering::LoweringF32(PatternRewriter &rewriter,
                              top::SplitOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void SplitLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::SplitOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void SplitLowering::LoweringF16(PatternRewriter &rewriter,
                              top::SplitOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void SplitLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::SplitOp op) const {
  lowering_common<tpu::SplitOp>(rewriter, op, op.input().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
