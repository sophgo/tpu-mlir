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

void TransformerLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::TransformerOp op) const {
  lowering_common_f32<tpu::TransformerOp>(rewriter, op);
}
void TransformerLowering::LoweringINT4(PatternRewriter &rewriter, top::TransformerOp op,
                                       bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void TransformerLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::TransformerOp op, bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void TransformerLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::TransformerOp op) const {
  lowering_common_bf16<tpu::TransformerOp>(rewriter, op);
}

void TransformerLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::TransformerOp op) const {
  lowering_common_f16<tpu::TransformerOp>(rewriter, op);
}

void TransformerLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::TransformerOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
