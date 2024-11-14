//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-SwapChannel"
namespace tpu_mlir {
namespace cv18xx {
void SwapChannelLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::SwapChannelOp op,
                                       bool asymmetric) const {
  lowering_common_int8<tpu::SwapChannelOp>(rewriter, op, asymmetric);
}

void SwapChannelLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::SwapChannelOp op) const {
  lowering_common_bf16<tpu::SwapChannelOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
