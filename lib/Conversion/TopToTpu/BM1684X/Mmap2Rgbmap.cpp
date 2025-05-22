//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

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

void Mmap2RgbmapLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::Mmap2RgbmapOp op,
                                       bool asymmetric) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

void Mmap2RgbmapLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::Mmap2RgbmapOp op) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

void Mmap2RgbmapLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::Mmap2RgbmapOp op) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

void Mmap2RgbmapLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::Mmap2RgbmapOp op) const {
  LoweringMmap2Rgbmap(rewriter, op);
}

void Mmap2RgbmapLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::Mmap2RgbmapOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void Mmap2RgbmapLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::Mmap2RgbmapOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
