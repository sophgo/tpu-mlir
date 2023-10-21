//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-leakyrelu"
namespace tpu_mlir {
namespace cv18xx {
void LeakyReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LeakyReluOp op) const {
  lowering_common_bf16<tpu::LeakyReluOp>(rewriter, op);
}
void LeakyReluLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {
  auto threshold_x = module::getThreshold(op.getInput());
  auto threshold_y = module::getThreshold(op.getOutput());
  double negative_slope = op.getAlpha().convertToDouble();
  int64_t multiplier_pos, multiplier_neg;
  int64_t rshift_pos, rshift_neg;

  double qscale_pos = threshold_x / threshold_y;
  if (std::fabs(threshold_x - threshold_y) <
      1e-5 * std::min(threshold_x, threshold_y)) {
    rshift_pos = 0;
    multiplier_pos = 1;
  } else {
    getRShiftAndMultiplierFromQScale(qscale_pos, &multiplier_pos, &rshift_pos,
                                     false);
  }
  float qscale_neg = std::fabs(qscale_pos * negative_slope);
  getRShiftAndMultiplierFromQScale(qscale_neg, &multiplier_neg, &rshift_neg,
                                   false);
  if (negative_slope < 0) {
    multiplier_neg *= -1;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("alpha", op.getAlphaAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier_pos)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier_neg", rewriter.getSI32IntegerAttr(multiplier_neg)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift_pos)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift_neg", rewriter.getSI32IntegerAttr(rshift_neg)));

  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType,
                                                Value(op.getInput()), attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
