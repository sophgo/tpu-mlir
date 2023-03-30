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

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::ReduceOp op) const {
  lowering_common_f32<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::ReduceOp op, bool asymmetric) const {
  // lowering_common_int8<tpu::ReduceOp>(rewriter, op, false, 3);
  lowering_common_f32<tpu::ReduceOp>(rewriter, op, 3);
}

} // namespace bm1684
} // namespace tpu_mlir
