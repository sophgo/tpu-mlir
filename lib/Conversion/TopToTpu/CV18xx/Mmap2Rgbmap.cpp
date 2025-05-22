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

static void LoweringMmap2Rgbmap(PatternRewriter &rewriter,
                                top::Mmap2RgbmapOp op) {
  std::vector<Value> operands;
  operands.emplace_back(op.getOperand());
  mlir::Type new_type = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tpu::Mmap2RgbmapOp>(
      op, new_type, operands, op.getOperation()->getAttrs());
  return;
}

void Mmap2RgbmapLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::Mmap2RgbmapOp op,
                                       bool asymmetric) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

void Mmap2RgbmapLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::Mmap2RgbmapOp op) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
