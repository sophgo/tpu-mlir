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

void Depth2SpaceLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::Depth2SpaceOp op) const {
  lowering_common_float<tpu::Depth2SpaceOp>(rewriter, op.getOperation());
}

void Depth2SpaceLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::Depth2SpaceOp op,
                                       bool asymmetric) const {
  lowering_common_int8<tpu::Depth2SpaceOp>(rewriter, op.getOperation(),
                                           asymmetric);
}

void Depth2SpaceLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::Depth2SpaceOp op) const {
  lowering_common_float<tpu::Depth2SpaceOp, BFloat16Type>(rewriter,
                                                          op.getOperation());
}

void Depth2SpaceLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::Depth2SpaceOp op) const {
  lowering_common_float<tpu::Depth2SpaceOp, Float16Type>(rewriter,
                                                         op.getOperation());
}

void Depth2SpaceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::Depth2SpaceOp op) const {
  lowering_common<tpu::Depth2SpaceOp>(rewriter, op.getOperation(),
                                      op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
