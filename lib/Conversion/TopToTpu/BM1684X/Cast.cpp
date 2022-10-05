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

void CastLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringINT8(PatternRewriter &rewriter,
                                top::CastOp op, bool asymmetric) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
