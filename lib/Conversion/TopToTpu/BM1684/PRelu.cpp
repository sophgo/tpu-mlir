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

void PReluLowering::LoweringF32(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  lowering_common_f32<tpu::PReluOp>(rewriter,op);
}

void PReluLowering::LoweringINT8(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  // todo
  lowering_common_f32<tpu::PReluOp>(rewriter,op);
}

} // namespace bm1684
} // namespace tpu_mlir