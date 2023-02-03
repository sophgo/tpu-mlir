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

void ScatterNDLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  lowering_common_f32<tpu::ScatterNDOp>(rewriter, op);
}

void ScatterNDLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ScatterNDLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ScatterNDLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

void ScatterNDLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

void ScatterNDLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
