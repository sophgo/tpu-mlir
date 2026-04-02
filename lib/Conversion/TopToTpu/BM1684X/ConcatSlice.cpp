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

void ConcatSliceLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::ConcatSliceOp op) const {
  lowering_common_f32<tpu::ConcatSliceOp>(rewriter, op);
}

void ConcatSliceLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::ConcatSliceOp op,
                                       bool asymmetric) const {
  lowering_common_f16<tpu::ConcatSliceOp>(rewriter, op);
}

void ConcatSliceLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::ConcatSliceOp op,
                                       bool asymmetric) const {
  lowering_common_f16<tpu::ConcatSliceOp>(rewriter, op);
}

void ConcatSliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::ConcatSliceOp op) const {
  lowering_common_bf16<tpu::ConcatSliceOp>(rewriter, op);
}

void ConcatSliceLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::ConcatSliceOp op) const {
  lowering_common_f16<tpu::ConcatSliceOp>(rewriter, op);
}

void ConcatSliceLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::ConcatSliceOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ConcatSliceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::ConcatSliceOp op) const {
  lowering_common<tpu::ConcatSliceOp>(rewriter, op.getOperation(),
                                      op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
