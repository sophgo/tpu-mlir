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
  lowering_common_f32<tpu::ReduceOp>(rewriter, op);
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::ReduceOp op, bool asymmetric) const {
  lowering_common_f16<tpu::ReduceOp>(rewriter, op);
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::ReduceOp op) const {
  lowering_common_bf16<tpu::ReduceOp>(rewriter, op);
}

void ReduceLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::ReduceOp op) const {
  lowering_common_f16<tpu::ReduceOp>(rewriter, op);
}

void ReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::ReduceOp op) const {
  lowering_common<tpu::ReduceOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
