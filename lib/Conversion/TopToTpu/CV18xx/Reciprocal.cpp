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

  double const_s = op.const_val().convertToDouble();
  Value table =
      create_lookup_table(op.input(), op.output(), asymmetric,
                          [const_s](double val) { return const_s / val; });
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(
      op, newType,
      ValueRange{op.input(), table, Module::getNoneOp(op.getOperation())},
      attrs);
}

void ReciprocalLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::ReciprocalOp op) const {
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  std::vector<float> reciprocal_table(table_hw);
  std::vector<float> reciprocal_mantissa_table(table_hw);
  float range_start = -62;
  float range_end = 63;
  bf16_gen_exponent_mantissa_table("pow", reciprocal_table.data(), reciprocal_mantissa_table.data(), -1.0f, 0);
  auto shape = std::vector<int64_t>{1, 1, table_h, table_w};

  OpBuilder builder(op->getContext());
  auto table_type = RankedTensorType::get(shape, builder.getF32Type());
  auto table_op = top::WeightOp::create(op, "table", reciprocal_table, table_type);
  auto mantissa_op =
      top::WeightOp::create(op, "mantissa", reciprocal_mantissa_table, table_type);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutModeAttr::get(op->getContext(), tpu::LutMode::Mantissa)));
  attrs.push_back(rewriter.getNamedAttr("min_range",
                                        rewriter.getF64FloatAttr(range_start)));
  attrs.push_back(
      rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto tensor_type = op.output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), rewriter.getBF16Type());
  auto table_weight_op = dyn_cast<top::WeightOp>(table_op.getDefiningOp());
  auto mantissa_weight_op =
      dyn_cast<top::WeightOp>(mantissa_op.getDefiningOp());
  rewriter.replaceOpWithNewOp<tpu::LutOp>(
      op, newType,
      ValueRange{op.input(), table_weight_op.clone_bf16(op),
                 mantissa_weight_op.clone_bf16(op)},
      attrs);
}
}
}
