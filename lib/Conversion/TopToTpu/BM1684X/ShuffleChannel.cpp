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

void ShuffleChannelLowering::LoweringF32(PatternRewriter &rewriter,
                                         top::ShuffleChannelOp op) const {
  lowering_common_f32<tpu::ShuffleChannelOp>(rewriter, op);
}

void ShuffleChannelLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::ShuffleChannelOp op,
                                          bool asymmetric) const {
  lowering_common_int8<tpu::ShuffleChannelOp>(rewriter, op.getOperation(),
                                              asymmetric);
}
void ShuffleChannelLowering::LoweringINT4(PatternRewriter &rewriter,
                                          top::ShuffleChannelOp op,
                                          bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ShuffleChannelLowering::LoweringBF16(PatternRewriter &rewriter,
                                          top::ShuffleChannelOp op) const {
  lowering_common_bf16<tpu::ShuffleChannelOp>(rewriter, op);
}

void ShuffleChannelLowering::LoweringF16(PatternRewriter &rewriter,
                                         top::ShuffleChannelOp op) const {
  lowering_common_f16<tpu::ShuffleChannelOp>(rewriter, op);
}

void ShuffleChannelLowering::LoweringF8(PatternRewriter &rewriter,
                                        top::ShuffleChannelOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ShuffleChannelLowering::LoweringQuantized(PatternRewriter &rewriter,
                                               top::ShuffleChannelOp op) const {
  lowering_common<tpu::ShuffleChannelOp>(rewriter, op.getOperation(),
                                         op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
