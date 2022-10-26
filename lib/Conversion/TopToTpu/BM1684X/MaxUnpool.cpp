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

void MaxUnpoolLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::MaxUnpoolOp op) const {
  lowering_common_f32<tpu::MaxUnpoolOp>(rewriter, op);
}

void MaxUnpoolLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::MaxUnpoolOp op,
                                     bool asymmetric) const {
  lowering_common_f32<tpu::MaxUnpoolOp>(rewriter, op);
}

void MaxUnpoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::MaxUnpoolOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaxUnpoolLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::MaxUnpoolOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaxUnpoolLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::MaxUnpoolOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
