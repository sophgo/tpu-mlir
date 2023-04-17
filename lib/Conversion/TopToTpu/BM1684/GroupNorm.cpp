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

void GroupNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::GroupNormOp op) const {
  lowering_common_f32<tpu::GroupNormOp>(rewriter, op, 5);
}

void GroupNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::GroupNormOp op,
                                     bool asymmetric) const {
  // only support FP32
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
