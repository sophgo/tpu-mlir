//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

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

} // namespace bm1684
} // namespace tpu_mlir
