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

void MaskRCNNMaskPoolerLowering::LoweringF32(
    PatternRewriter &rewriter, top::MaskRCNNMaskPoolerOp op) const {
  lowering_common_f32<tpu::MaskRCNNMaskPoolerOp>(rewriter, op, 12);
}

void MaskRCNNMaskPoolerLowering::LoweringINT8(PatternRewriter &rewriter,
                                              top::MaskRCNNMaskPoolerOp op,
                                              bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNMaskPoolerLowering::LoweringINT4(PatternRewriter &rewriter,
                                              top::MaskRCNNMaskPoolerOp op,
                                              bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNMaskPoolerLowering::LoweringBF16(
    PatternRewriter &rewriter, top::MaskRCNNMaskPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNMaskPoolerLowering::LoweringF16(
    PatternRewriter &rewriter, top::MaskRCNNMaskPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNMaskPoolerLowering::LoweringF8(
    PatternRewriter &rewriter, top::MaskRCNNMaskPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNMaskPoolerLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaskRCNNMaskPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
