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

void ReverseLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  lowering_common_f32<tpu::ReverseOp>(rewriter, op);
}

void ReverseLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::ReverseOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ReverseLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::ReverseOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ReverseLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::ReverseOp op) const {
  LoweringF32(rewriter, op);
}

void ReverseLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  LoweringF32(rewriter, op);
}

void ReverseLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  llvm_unreachable("Not Implemented");
}

void ReverseLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::ReverseOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
