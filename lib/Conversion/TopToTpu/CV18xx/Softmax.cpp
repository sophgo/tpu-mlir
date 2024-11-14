//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-Softmax"
namespace tpu_mlir {
namespace cv18xx {
static double active_exp(double val) { return std::exp(val); }
void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void SoftmaxLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SoftmaxOp op) const {
  Value table_weight, slope_table_weight, reciprocal_table_weight,
      reciprocal_mantissa_table_weight;
  createBf16LutOp(op, "slope", TableMode::Slope, 0.0, 0.0, -EXP_BF16_LUT_RANGE,
                  EXP_BF16_LUT_RANGE, active_exp, table_weight,
                  slope_table_weight);
  if (op.getLog()) {
    createBf16LutOp(op, "log", TableMode::Mantissa, 0.0, 0.0, -62, 63, nullptr,
                    reciprocal_table_weight, reciprocal_mantissa_table_weight);
  } else {
    createBf16LutOp(op, "pow", TableMode::Mantissa, -1.0, 0.0, -62, 63, nullptr,
                    reciprocal_table_weight, reciprocal_mantissa_table_weight);
  }
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
      op, newType,
      ValueRange{op.getInput(), table_weight, slope_table_weight,
                 reciprocal_table_weight, reciprocal_mantissa_table_weight,
                 module::getNoneOp(op.getOperation())},
      op->getAttrs());
  return;
}
} // namespace cv18xx
} // namespace tpu_mlir
