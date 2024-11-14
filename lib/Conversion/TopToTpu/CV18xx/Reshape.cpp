//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-add"

namespace tpu_mlir {
namespace cv18xx {
void ReshapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::ReshapeOp>(rewriter, op, asymmetric);
}

void ReshapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ReshapeOp op) const {
  auto out = op.getOutput();
  if (module::isCalibratedType(out)) {
    // For align-input(CscOp sometimes maybe convert to ReshapeOp) use, it
    // should be lowered to uint8.
    auto qtype = module::getCalibratedType(out);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    if (min == 0 && max == 255) {
      lowering_common_int8<tpu::ReshapeOp>(rewriter, op, false);
      return;
    }
  }
  lowering_common_bf16<tpu::ReshapeOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
