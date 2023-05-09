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

void CompareLowering::LoweringINT8(PatternRewriter &rewriter, top::CompareOp op,
                                   bool asymmetric) const {
  auto newType = getQuantBoolType(op.getOutput());
  lowering_common<tpu::CompareOp>(rewriter, op.getOperation(), newType);
}

void CompareLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::CompareOp op) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
