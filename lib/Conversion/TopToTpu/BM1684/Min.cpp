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

void MinLowering::LoweringF32(PatternRewriter &rewriter, top::MinOp op) const {
  lowering_common_f32<tpu::MinOp>(rewriter, op, 2);
}

void MinLowering::LoweringINT8(PatternRewriter &rewriter, top::MinOp op,
                               bool asymmetric) const {
  lowering_common_int8<tpu::MinOp>(rewriter, op, 2);
}

} // namespace bm1684
} // namespace tpu_mlir
