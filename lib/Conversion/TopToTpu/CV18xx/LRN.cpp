//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#define DEBUG_TYPE "lowering-lrn"
namespace tpu_mlir {
namespace cv18xx {

void LRNLowering::LoweringINT8(PatternRewriter &rewriter, top::LRNOp op,
                               bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void LRNLowering::LoweringBF16(PatternRewriter &rewriter, top::LRNOp op) const {
  Value table_weight, mantissa_weight;
  float range_start = -62, range_end = 63;
  createBf16LutOp(op, "pow", TableMode::Mantissa,
                  -1 * op.getBeta().convertToDouble(), 0.0, range_start,
                  range_end, nullptr, table_weight, mantissa_weight);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutBF16ModeAttr::get(op->getContext(), tpu::LutBF16Mode::Mantissa)));
  attrs.push_back(rewriter.getNamedAttr("min_range",
                                        rewriter.getF64FloatAttr(range_start)));
  attrs.push_back(
      rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::LRNOp>(
      op, newType, ValueRange{op.getInput(), table_weight, mantissa_weight},
      attrs);
  return;
}
} // namespace cv18xx
} // namespace tpu_mlir
