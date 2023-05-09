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

void CustomLowering::LoweringF32(PatternRewriter &rewriter, top::CustomOp op) const {
  lowering_common_f32<tpu::CustomOp>(rewriter, op);
}

void CustomLowering::LoweringINT8(PatternRewriter &rewriter, top::CustomOp op,
                                bool asymmetric) const {
  lowering_common_f16<tpu::CustomOp>(rewriter, op);
}

void CustomLowering::LoweringINT4(PatternRewriter &rewriter, top::CustomOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CustomLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CustomOp op) const {
  lowering_common_bf16<tpu::CustomOp>(rewriter, op);
}

void CustomLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CustomOp op) const {
  lowering_common_f16<tpu::CustomOp>(rewriter, op);
}

void CustomLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CustomOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
