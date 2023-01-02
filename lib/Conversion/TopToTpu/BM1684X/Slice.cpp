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

void SliceLowering::LoweringF32(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  lowering_common_f32<tpu::SliceOp>(rewriter, op);
}
void SliceLowering::LoweringINT4(PatternRewriter &rewriter, top::SliceOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SliceLowering::LoweringINT8(PatternRewriter &rewriter, top::SliceOp op,
                                 bool asymmetric) const {
  lowering_common_int8<tpu::SliceOp>(rewriter, op, asymmetric);
}

void SliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::SliceOp op) const {
  lowering_common_bf16<tpu::SliceOp>(rewriter, op);
}

void SliceLowering::LoweringF16(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  lowering_common_f16<tpu::SliceOp>(rewriter, op);
}

void SliceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::SliceOp op) const {
  lowering_common<tpu::SliceOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
