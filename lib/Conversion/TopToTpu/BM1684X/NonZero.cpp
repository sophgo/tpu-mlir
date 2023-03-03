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

void NonZeroLowering::LoweringF32(PatternRewriter &rewriter, top::NonZeroOp op) const {
  lowering_common_f32<tpu::NonZeroOp>(rewriter, op, 2);
}

void NonZeroLowering::LoweringINT8(PatternRewriter &rewriter, top::NonZeroOp op,
                                   bool asymmetric) const {
  lowering_common_f32<tpu::NonZeroOp>(rewriter, op, 2);
}

void NonZeroLowering::LoweringINT4(PatternRewriter &rewriter, top::NonZeroOp op,
                                   bool asymmetric) const {
  lowering_common_f32<tpu::NonZeroOp>(rewriter, op, 2);
}

void NonZeroLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::NonZeroOp op) const {
  lowering_common_f32<tpu::NonZeroOp>(rewriter, op, 2);
}

void NonZeroLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::NonZeroOp op) const {
  lowering_common_f32<tpu::NonZeroOp>(rewriter, op, 2);
}

void NonZeroLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::NonZeroOp nonzeroOp) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
