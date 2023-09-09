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

void BatchNormBwdLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::BatchNormBwdOp op) const {
  lowering_common_f32<tpu::BatchNormBwdOp>(rewriter, op, 6);
}

void BatchNormBwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::BatchNormBwdOp op) const {
  LoweringF32(rewriter, op);
}

void BatchNormBwdLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::BatchNormBwdOp op) const {
  LoweringF32(rewriter, op);
}

void BatchNormBwdLowering::LoweringINT8(PatternRewriter &rewriter, top::BatchNormBwdOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void BatchNormBwdLowering::LoweringINT4(PatternRewriter &rewriter, top::BatchNormBwdOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void BatchNormBwdLowering::LoweringQuantized(PatternRewriter &rewriter, top::BatchNormBwdOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
