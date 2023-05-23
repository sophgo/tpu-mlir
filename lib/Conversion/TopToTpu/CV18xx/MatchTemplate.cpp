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

void MatchTemplateLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::MatchTemplateOp op,
                                         bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void MatchTemplateLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::MatchTemplateOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (auto v : op.getOperands())
    operands.push_back(v);
  // creat table
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  std::vector<float> exp_table(table_hw);
  std::vector<float> mantissa_table(table_hw);
  bf16_gen_exponent_mantissa_table("pow", exp_table.data(),
                                   mantissa_table.data(), -0.5f, 0.f);
  auto table_type =
      RankedTensorType::get({1, 1, table_h, table_w}, rewriter.getF32Type());
  auto tableOp = top::WeightOp::create(op, "table", exp_table, table_type);
  auto mantissaOp =
      top::WeightOp::create(op, "mantissa_table", mantissa_table, table_type);
  operands.push_back(
      dyn_cast<top::WeightOp>(tableOp.getDefiningOp()).clone_bf16(op));
  operands.push_back(
      dyn_cast<top::WeightOp>(mantissaOp.getDefiningOp()).clone_bf16(op));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::MatchTemplateOp>(op, newType, operands,
                                                    op->getAttrs());
}
} // namespace cv18xx
} // namespace tpu_mlir
