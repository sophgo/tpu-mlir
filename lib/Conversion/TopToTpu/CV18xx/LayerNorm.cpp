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

#define DEBUG_TYPE "lowering-layernorm"

namespace tpu_mlir {
namespace cv18xx {

static bool is_unused_weight(Operation *op, float v) {
  if (auto castOp = dyn_cast<top::WeightOp>(op)) {
    auto data = castOp.read<float>();
    for (int i = 0; i < data->size(); i++) {
      if (data->at(i) != v) {
        return false;
      }
    }
    return true;
  } else {
    op->dump();
    llvm_unreachable("Is not weightOp.");
  }
}

void LayerNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LayerNormOp op) const {

  // TODO support output mean std
  assert(op.getMean().getType().isa<mlir::NoneType>() &&
         op.getRstd().getType().isa<mlir::NoneType>());
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  // lowering weight
  auto none = module::getNoneOp(op);
  auto filterOp = op.getWeight().getDefiningOp();
  auto biasOp = op.getBias().getDefiningOp();
  bool w_unused = is_unused_weight(filterOp, 1);
  bool b_unused = is_unused_weight(biasOp, 0);
  if (w_unused && b_unused) {
    operands.push_back(none);
    operands.push_back(none);
  } else {
    operands.push_back(dyn_cast<top::WeightOp>(filterOp).clone_bf16(op));
    operands.push_back(dyn_cast<top::WeightOp>(biasOp).clone_bf16(op));
  }
  // add extra lut table
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

  std::vector<Type> new_types;
  for (auto out : op.getResults()) {
    new_types.push_back(getQuantBF16Type(out));
  }
  rewriter.replaceOpWithNewOp<tpu::LayerNormOp>(op, new_types, operands,
                                             op->getAttrs());
}

void LayerNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
