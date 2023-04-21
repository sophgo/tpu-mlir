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

void Depth2SpaceLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::Depth2SpaceOp op) const {
  lowering_common_f32<tpu::Depth2SpaceOp>(rewriter, op);
}

void Depth2SpaceLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::Depth2SpaceOp op,
                                       bool asymmetric) const {
  lowering_common_int8<tpu::Depth2SpaceOp>(rewriter, op, false);
}

} // namespace bm1684
} // namespace tpu_mlir