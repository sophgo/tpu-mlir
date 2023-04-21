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

void GatherElementsLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::GatherElementsOp op) const {
  lowering_common_f32<tpu::GatherElementsOp>(rewriter, op, 0, 1);
}

void GatherElementsLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherElementsOp op,
                                  bool asymmetric) const {
  lowering_common_f32<tpu::GatherElementsOp>(rewriter, op, 0, 1);
}
void GatherElementsLowering::LoweringINT4(PatternRewriter &rewriter, top::GatherElementsOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void GatherElementsLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::GatherElementsOp op) const {
  lowering_common_bf16<tpu::GatherElementsOp>(rewriter, op, 0, 1);
}

void GatherElementsLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::GatherElementsOp op) const {
  lowering_common_f16<tpu::GatherElementsOp>(rewriter, op, 0, 1);
}

void GatherElementsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::GatherElementsOp op) const {
  lowering_common<tpu::GatherElementsOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
