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

void UpsampleLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::UpsampleOp op) const {
  lowering_common_f32<tpu::UpsampleOp>(rewriter, op);
}

void UpsampleLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::UpsampleOp op, bool asymmetric) const {
  lowering_common_int8<tpu::UpsampleOp>(rewriter, op, false);
}

} // namespace bm1684
} // namespace tpu_mlir
