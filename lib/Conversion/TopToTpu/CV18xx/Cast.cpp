//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {
void CastLowering::LoweringINT8(PatternRewriter &rewriter, top::CastOp op,
                                bool asymmetric) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.getOutput().getType());
}

void CastLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.getOutput().getType());
}
} // namespace cv18xx
} // namespace tpu_mlir
