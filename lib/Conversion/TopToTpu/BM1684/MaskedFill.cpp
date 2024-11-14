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

void MaskedFillLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::MaskedFillOp op) const {
  lowering_common_f32<tpu::MaskedFillOp>(rewriter, op, 2);
}

void MaskedFillLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::MaskedFillOp op,
                                      bool asymmetric) const {
  lowering_common_f32<tpu::MaskedFillOp>(rewriter, op, 2);
}

} // namespace bm1684
} // namespace tpu_mlir
