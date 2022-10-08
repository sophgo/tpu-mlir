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

void ReshapeLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  lowering_common_float<tpu::ReshapeOp>(rewriter, op);
}

void ReshapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::ReshapeOp>(rewriter, op, asymmetric);
}

void ReshapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ReshapeOp op) const {
  lowering_common_float<tpu::ReshapeOp, BFloat16Type>(rewriter, op);
}

void ReshapeLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  lowering_common_float<tpu::ReshapeOp, Float16Type>(rewriter, op);
}

void ReshapeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::ReshapeOp op) const {
  lowering_common<tpu::ReshapeOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
