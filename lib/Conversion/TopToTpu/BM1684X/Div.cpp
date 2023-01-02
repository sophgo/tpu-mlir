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

void DivLowering::LoweringF32(PatternRewriter &rewriter, top::DivOp op) const {
  lowering_common_f32<tpu::DivOp>(rewriter, op);
}

void DivLowering::LoweringINT8(PatternRewriter &rewriter, top::DivOp op,
                               bool asymmetric) const {
  lowering_common_f32<tpu::DivOp>(rewriter, op);
}
void DivLowering::LoweringINT4(PatternRewriter &rewriter, top::DivOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void DivLowering::LoweringBF16(PatternRewriter &rewriter, top::DivOp op) const {
  lowering_common_f32<tpu::DivOp>(rewriter, op);
}

void DivLowering::LoweringF16(PatternRewriter &rewriter, top::DivOp op) const {
  lowering_common_f32<tpu::DivOp>(rewriter, op);
}

void DivLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::DivOp op) const {
  lowering_common<tpu::DivOp>(rewriter, op.getOperation(),
                              op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
