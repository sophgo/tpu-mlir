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

void PermuteLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_float<tpu::PermuteOp>(rewriter, op);
}

void PermuteLowering::LoweringINT8(PatternRewriter &rewriter,
                                   top::PermuteOp op, bool asymmetric) const {
  lowering_common_int8<tpu::PermuteOp>(rewriter, op, asymmetric);
}

void PermuteLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::PermuteOp op) const {
  lowering_common_float<tpu::PermuteOp, BFloat16Type>(rewriter, op);
}

void PermuteLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_float<tpu::PermuteOp, Float16Type>(rewriter, op);
}

void PermuteLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::PermuteOp op) const {
  lowering_common<tpu::PermuteOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
