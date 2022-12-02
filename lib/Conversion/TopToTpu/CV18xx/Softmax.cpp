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

#define DEBUG_TYPE "lowering-Softmax"
namespace tpu_mlir {
namespace cv18xx {
static double active_exp(double val) {
    return std::exp(val);
}
void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void SoftmaxLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SoftmaxOp op) const {
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  std::vector<float> table(table_hw);
  std::vector<float> slope_table(table_hw);
  std::vector<float> reciprocal_table(table_hw);
  std::vector<float> reciprocal_mantissa_table(table_hw);
  float range_start = -15;
  float range_end = 15;
  bf16_gen_base_slope_table(table.data(), slope_table.data(), range_start, range_end, active_exp);
  bf16_gen_exponent_mantissa_table("pow", reciprocal_table.data(), reciprocal_mantissa_table.data(), -1.0f, 0);
  auto shape = std::vector<int64_t>{1, 1, table_h, table_w};
  OpBuilder builder(op->getContext());
  auto table_type = RankedTensorType::get(shape, builder.getF32Type());
  auto table_op = top::WeightOp::create(op, "table", table, table_type);
  auto slope_table_op = top::WeightOp::create(op, "slope_table", slope_table, table_type);
  auto reciprocal_table_op = top::WeightOp::create(op, "reciprocal_table", reciprocal_table, table_type);
  auto reciprocal_mantissa_table_op = top::WeightOp::create(op, "reciprocal_mantissa_table", reciprocal_mantissa_table, table_type);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  // attrs.push_back(rewriter.getNamedAttr("min_range",
  //                                       rewriter.getF64FloatAttr(range_start)));
  // attrs.push_back(
  //     rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto newType = getQuantBF16Type(op.output());
  auto table_weight_op = dyn_cast<top::WeightOp>(table_op.getDefiningOp());
  auto slope_table_weight_op = dyn_cast<top::WeightOp>(slope_table_op.getDefiningOp());
  auto reciprocal_table_weight_op = dyn_cast<top::WeightOp>(reciprocal_table_op.getDefiningOp());
  auto reciprocal_mantissa_tableweight_op = dyn_cast<top::WeightOp>(reciprocal_mantissa_table_op.getDefiningOp());
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
      op, newType,
      ValueRange{op.input(), table_weight_op.clone_bf16(op), slope_table_weight_op.clone_bf16(op),
                 reciprocal_table_weight_op.clone_bf16(op), reciprocal_mantissa_tableweight_op.clone_bf16(op)}, attrs);
  return;
}
}
}
