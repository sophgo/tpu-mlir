//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

#define DEBUG_TYPE "lowering-SwapChannel"
namespace tpu_mlir {
namespace bm1684x {
void SwapChannelLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::SwapChannelOp op) const {
  lowering_common_f32<tpu::SwapChannelOp>(rewriter, op);
}

void SwapChannelLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::SwapChannelOp op,
                                       bool asymmetric) const {
  lowering_common_int8<tpu::SwapChannelOp>(rewriter, op, asymmetric);
}
void SwapChannelLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::SwapChannelOp op,
                                       bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SwapChannelLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::SwapChannelOp op) const {
  lowering_common_bf16<tpu::SwapChannelOp>(rewriter, op);
}

void SwapChannelLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::SwapChannelOp op) const {
  lowering_common_f16<tpu::SwapChannelOp>(rewriter, op);
}

void SwapChannelLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::SwapChannelOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SwapChannelLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::SwapChannelOp op) const {
  lowering_common<tpu::SwapChannelOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
