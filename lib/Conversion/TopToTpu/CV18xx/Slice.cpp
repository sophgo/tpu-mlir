//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-slice"
namespace tpu_mlir {
namespace cv18xx {
void SliceLowering::LoweringINT8(PatternRewriter &rewriter, top::SliceOp op,
                                 bool asymmetric) const {
  lowering_common_int8<tpu::SliceOp>(rewriter, op, asymmetric, 5);
}

void SliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::SliceOp op) const {
  auto out = op.getOutput();
  if (module::isCalibratedType(out)) {
    // For fuse_preprocess(crop image) and aligned use, it should be lowered to
    // uint8.
    auto qtype = module::getCalibratedType(out);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    if (min == 0 && max == 255) {
      lowering_common_int8<tpu::SliceOp>(rewriter, op, false, 5);
      return;
    }
  }
  lowering_common_bf16<tpu::SliceOp>(rewriter, op, 5);
}

} // namespace cv18xx
} // namespace tpu_mlir
