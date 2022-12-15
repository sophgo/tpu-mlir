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

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  lowering_common_f32<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  LoweringF16(rewriter, op);
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ReduceOp op) const {
  lowering_common_bf16<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  lowering_common_f16<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ReduceOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
