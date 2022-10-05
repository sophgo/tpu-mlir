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

void GatherLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  lowering_common_float<tpu::GatherOp>(rewriter, op.getOperation());
}

void GatherLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::GatherOp op, bool asymmetric) const {
  lowering_common_float<tpu::GatherOp>(rewriter, op.getOperation());
}

void GatherLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::GatherOp op) const {
  llvm_unreachable("Not Implemented");
}

void GatherLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  llvm_unreachable("Not Implemented");
}

void GatherLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::GatherOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
