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

void CompareLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::CompareOp compareOp) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringINT8(PatternRewriter &rewriter, top::CompareOp compareOp,
                                   bool asymmetric) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::CompareOp compareOp) const {
  lowering_common_bf16<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::CompareOp compareOp) const {
  lowering_common_f16<tpu::CompareOp>(rewriter, compareOp.getOperation());
}

void CompareLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::CompareOp compareOp) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
