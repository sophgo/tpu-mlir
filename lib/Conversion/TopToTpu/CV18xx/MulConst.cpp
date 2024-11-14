//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-mulconst"
namespace tpu_mlir {
namespace cv18xx {
void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  double const_val = op.getConstVal().convertToDouble();
  auto active_mulconst = [const_val](double val) { return val * const_val; };
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    active_mulconst);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void MulConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MulConstOp op) const {
  double const_val = op.getConstVal().convertToDouble();
  auto active_mulconst = [const_val](double val) { return val * const_val; };
  Value table_weight, slope_weight;
  float range_start = -15, range_end = 15;
  createBf16LutOp(op, "slope", TableMode::Slope, 0.0, 0.0, range_start,
                  range_end, active_mulconst, table_weight, slope_weight);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutBF16ModeAttr::get(op->getContext(), tpu::LutBF16Mode::Slope)));
  attrs.push_back(rewriter.getNamedAttr("min_range",
                                        rewriter.getF64FloatAttr(range_start)));
  attrs.push_back(
      rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::LutBF16Op>(
      op, newType, ValueRange{op.getInput(), table_weight, slope_weight},
      attrs);
  return;
}
} // namespace cv18xx
} // namespace tpu_mlir
