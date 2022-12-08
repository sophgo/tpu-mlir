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

#define DEBUG_TYPE "lowering-reciprocal"
namespace tpu_mlir {
namespace cv18xx {

void ReciprocalLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::ReciprocalOp op,
                                      bool asymmetric) const {
  // for convert from DivOp
  if (!Quant::isCalibratedType(op.output()) &&
      !Quant::isUniformQuantized(op.output())) {
    LoweringBF16(rewriter, op);
    return;
  }

  double const_s = op.const_val().convertToDouble();
  Value table =
      create_lookup_table(op.input(), op.output(), asymmetric,
                          [const_s](double val) { return const_s / val; });
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table}, attrs);
}

void ReciprocalLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::ReciprocalOp op) const {
  Value table_weight, mantissa_weight;
  float range_start = -62, range_end = 63;
  createBf16LutOp(op, "pow", TableMode::Mantissa, -1.0, 0.0, range_start, range_end,
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
      op, newType,
      ValueRange{op.input(), table_weight, mantissa_weight},
      attrs);
  return;
}
} // namespace cv18xx
} // namespace tpu_mlir
