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

void AbsLowering::LoweringF32(PatternRewriter &rewriter,
                              top::AbsOp absOp) const {
  lowering_common_float<tpu::AbsOp>(rewriter, absOp.getOperation());
}

void AbsLowering::LoweringINT8(PatternRewriter &rewriter, top::AbsOp absOp,
                               bool asymmetric) const {
  lowering_common_int8<tpu::AbsOp>(rewriter, absOp.getOperation(), asymmetric);
}

void AbsLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::AbsOp absOp) const {
  lowering_common_float<tpu::AbsOp, BFloat16Type>(rewriter,
                                                  absOp.getOperation());
}

void AbsLowering::LoweringF16(PatternRewriter &rewriter,
                              top::AbsOp absOp) const {
  lowering_common_float<tpu::AbsOp, Float16Type>(rewriter,
                                                 absOp.getOperation());
}

void AbsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::AbsOp absOp) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
