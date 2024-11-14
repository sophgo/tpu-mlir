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

void StridedSliceLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::StridedSliceOp op) const {
  lowering_common_f32<tpu::StridedSliceOp>(rewriter, op);
}

void StridedSliceLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::StridedSliceOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::StridedSliceOp>(rewriter, op, asymmetric);
}
void StridedSliceLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::StridedSliceOp op,
                                        bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void StridedSliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::StridedSliceOp op) const {
  lowering_common_bf16<tpu::StridedSliceOp>(rewriter, op);
}

void StridedSliceLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::StridedSliceOp op) const {
  lowering_common_f16<tpu::StridedSliceOp>(rewriter, op);
}

void StridedSliceLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::StridedSliceOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void StridedSliceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::StridedSliceOp op) const {
  lowering_common<tpu::StridedSliceOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
