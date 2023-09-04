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

void EmbDenseBwdLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::EmbDenseBwdOp op) const {
  lowering_common_f32<tpu::EmbDenseBwdOp>(rewriter, op);
}

void EmbDenseBwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::EmbDenseBwdOp op) const {
  LoweringF32(rewriter, op);
}

void EmbDenseBwdLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::EmbDenseBwdOp op) const {
  LoweringF32(rewriter, op);
}

void EmbDenseBwdLowering::LoweringINT8(PatternRewriter &rewriter, top::EmbDenseBwdOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void EmbDenseBwdLowering::LoweringINT4(PatternRewriter &rewriter, top::EmbDenseBwdOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void EmbDenseBwdLowering::LoweringQuantized(PatternRewriter &rewriter, top::EmbDenseBwdOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
