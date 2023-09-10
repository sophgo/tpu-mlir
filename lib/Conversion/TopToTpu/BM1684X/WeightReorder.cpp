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

void WeightReorderLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::WeightReorderOp op) const {
  lowering_common_f32<tpu::WeightReorderOp>(rewriter, op);
}

void WeightReorderLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::WeightReorderOp op) const {
  LoweringF32(rewriter, op);
}

void WeightReorderLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::WeightReorderOp op) const {
  LoweringF32(rewriter, op);
}

void WeightReorderLowering::LoweringINT8(PatternRewriter &rewriter, top::WeightReorderOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void WeightReorderLowering::LoweringINT4(PatternRewriter &rewriter, top::WeightReorderOp op,
                          bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void WeightReorderLowering::LoweringQuantized(PatternRewriter &rewriter, top::WeightReorderOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
