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

void ScatterElementsLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::ScatterElementsOp op) const {
  lowering_common_f32<tpu::ScatterElementsOp>(rewriter, op);
}

void ScatterElementsLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::ScatterElementsOp op,
                                       bool asymmetric) const {
  lowering_common_int8<tpu::ScatterElementsOp>(rewriter, op.getOperation(),
                                           asymmetric);
}
void ScatterElementsLowering::LoweringINT4(PatternRewriter &rewriter, top::ScatterElementsOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ScatterElementsLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::ScatterElementsOp op) const {
  lowering_common_bf16<tpu::ScatterElementsOp>(rewriter, op);
}

void ScatterElementsLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::ScatterElementsOp op) const {
  lowering_common_f16<tpu::ScatterElementsOp>(rewriter, op);
}

void ScatterElementsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::ScatterElementsOp op) const {
  lowering_common<tpu::ScatterElementsOp>(rewriter, op.getOperation(),
                                      op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
