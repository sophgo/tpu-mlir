//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "lowering-pow"
namespace tpu_mlir {
namespace cv18xx {

static double g_ex, g_max;
void PowLowering::LoweringINT8(PatternRewriter &rewriter, top::PowOp op,
                               bool asymmetric) const {
  auto stype = module::getStorageType(op.output());
  auto qtype = module::getCalibratedType(op.output());
  g_ex = op.exponent().convertToDouble();
  g_max = qtype.getMax();
  auto fn = [](double val) {
    if (g_ex < 0 && val == 0) {
      return g_max;
    } else if (g_ex != (int)(g_ex) && val < 0) {
      return (double)(0.0);
    } else {
      return std::pow(val, g_ex);
    }
  };
  Value table = create_lookup_table(op.input(), op.output(), asymmetric, fn);
                                    // , ROUNDING_HALF_UP);
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
}

void PowLowering::LoweringBF16(PatternRewriter &rewriter, top::PowOp op) const {
  Value table_weight, mantissa_weight;
  float range_start = -62, range_end = 63;
  createBf16LutOp(op, "pow", TableMode::Mantissa,
                  op.exponent().convertToDouble(), 0.0, range_start, range_end,
                  nullptr, table_weight, mantissa_weight);
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
  auto newType = getQuantBF16Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::LutBF16Op>(
      op, newType, ValueRange{op.input(), table_weight, mantissa_weight},
      attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
