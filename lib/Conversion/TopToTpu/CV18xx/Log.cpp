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
#define DEBUG_TYPE "lowering-log"
namespace tpu_mlir {
namespace cv18xx {
static double active_log(double val) { return std::log(val); }

void LogLowering::LoweringINT8(PatternRewriter &rewriter, top::LogOp op,
                                   bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  Value table = create_lookup_table(op.input(), op.output(), asymmetric,
                                    [](double val) { return std::log(val); });
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

void LogLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::LogOp op) const {
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  std::vector<float> table(table_hw);
  std::vector<float> mantissa(table_hw);

  bf16_gen_exponent_mantissa_table("log", table.data(), mantissa.data(), 0.0, 0);
  auto shape = std::vector<int64_t>{1, 1, table_h, table_w};

  OpBuilder builder(op->getContext());
  auto table_type = RankedTensorType::get(shape, builder.getF32Type());
  auto table_op = top::WeightOp::create(op, "table", table, table_type);
  auto mantissa_op =
      top::WeightOp::create(op, "mantissa", mantissa, table_type);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutModeAttr::get(op->getContext(), tpu::LutMode::Log)));
  auto newType = getQuantBF16Type(op.output());
  auto table_weight_op = dyn_cast<top::WeightOp>(table_op.getDefiningOp());
  auto mantissa_weight_op =
      dyn_cast<top::WeightOp>(mantissa_op.getDefiningOp());
  rewriter.replaceOpWithNewOp<tpu::LutOp>(
      op, newType,
      ValueRange{op.input(), table_weight_op.clone_bf16(op),
                 mantissa_weight_op.clone_bf16(op)},
      attrs);
  return;
}
}
}
