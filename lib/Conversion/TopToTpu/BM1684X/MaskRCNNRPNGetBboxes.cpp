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

void MaskRCNNRPNGetBboxesLowering::LoweringF32(
    PatternRewriter &rewriter, top::MaskRCNNRPNGetBboxesOp op) const {
  lowering_common_f32<tpu::MaskRCNNRPNGetBboxesOp>(rewriter, op, 42);
}

void MaskRCNNRPNGetBboxesLowering::LoweringINT8(PatternRewriter &rewriter,
                                                top::MaskRCNNRPNGetBboxesOp op,
                                                bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNRPNGetBboxesLowering::LoweringINT4(PatternRewriter &rewriter,
                                                top::MaskRCNNRPNGetBboxesOp op,
                                                bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNRPNGetBboxesLowering::LoweringBF16(
    PatternRewriter &rewriter, top::MaskRCNNRPNGetBboxesOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNRPNGetBboxesLowering::LoweringF16(
    PatternRewriter &rewriter, top::MaskRCNNRPNGetBboxesOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNRPNGetBboxesLowering::LoweringF8(
    PatternRewriter &rewriter, top::MaskRCNNRPNGetBboxesOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNRPNGetBboxesLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaskRCNNRPNGetBboxesOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
