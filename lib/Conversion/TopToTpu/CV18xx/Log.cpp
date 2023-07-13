//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#define DEBUG_TYPE "lowering-log"
namespace tpu_mlir {
namespace cv18xx {
static double active_log(double val) { return std::log(val); }

void LogLowering::LoweringINT8(PatternRewriter &rewriter, top::LogOp op,
                               bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::log(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void LogLowering::LoweringBF16(PatternRewriter &rewriter, top::LogOp op) const {
  Value table_weight, mantissa_weight;
  float range_start = -62, range_end = 63;
  createBf16LutOp(op, "log", TableMode::Mantissa, 0.0, 0.0, range_start,
                  range_end, nullptr, table_weight, mantissa_weight);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutBF16ModeAttr::get(op->getContext(), tpu::LutBF16Mode::Log)));
  attrs.push_back(rewriter.getNamedAttr("min_range",
                                        rewriter.getF64FloatAttr(range_start)));
  attrs.push_back(
      rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::LutBF16Op>(
      op, newType, ValueRange{op.getInput(), table_weight, mantissa_weight},
      attrs);
  return;
}
} // namespace cv18xx
} // namespace tpu_mlir
