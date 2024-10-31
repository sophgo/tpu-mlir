//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

static void LoweringScatterElements(PatternRewriter &rewriter,
                                   top::ScatterElementsOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  if (module::isWeight(op.getInput())) {
    auto wOp = op.getInput().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getInput());
    }
  } else {
    operands.push_back(op.getInput());
  }

  if (module::isWeight(op.getIndices())) {
    auto wOp = op.getIndices().getDefiningOp<top::WeightOp>();
    operands.push_back(wOp.clone_int(op));
  } else {
    operands.push_back(op.getIndices());
  }

 if (module::isWeight(op.getUpdates())) {
    auto wOp = op.getUpdates().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getUpdates());
    }
  } else {
    operands.push_back(op.getUpdates());
  }
  // operands.push_back(op.getUpdates());

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp); // indices_coeff
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::ScatterElementsOp>(op, type, operands,
                                                     op->getAttrs());
  return;
}

void ScatterElementsLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::ScatterElementsOp op) const {
  // lowering_common_f32<tpu::ScatterElementsOp>(rewriter, op, 4);
  auto new_type = getQuantFloatType(op.getOutput());
  LoweringScatterElements(rewriter, op, new_type);
}


static void RequantizeInt8(PatternRewriter &rewriter,
                                   top::ScatterElementsOp op, Type type, bool asymmetric) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  auto data = op.getInput();
  auto updates = op.getUpdates();
  auto output = op.getOutput();
  auto new_data = do_transfer(data, output, asymmetric);
  operands.push_back(new_data);
  operands.push_back(op.getIndices());
  auto new_updates = do_transfer(updates, output, asymmetric);
  operands.push_back(new_updates);

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp); // indices_coeff
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::ScatterElementsOp>(op, type, operands,
                                                     op->getAttrs());
  return;
}


void ScatterElementsLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::ScatterElementsOp op,
                                       bool asymmetric) const {
  // lowering_common_int8<tpu::ScatterElementsOp>(rewriter, op.getOperation(),
  //                                          asymmetric);
  // Please implent lowering quant for weight if necessary
  if(module::isWeight(op.getInput()) || module::isWeight(op.getIndices()) || module::isWeight(op.getUpdates())){
    if(module::isMARS3())
      LoweringBF16(rewriter, op);
    else
      LoweringF32(rewriter, op);
    return;
  }
  auto new_type = getQuantInt8Type(op.getOutput());
  RequantizeInt8(rewriter, op, new_type, asymmetric);
}
void ScatterElementsLowering::LoweringINT4(PatternRewriter &rewriter, top::ScatterElementsOp op,
                                   bool asymmetric) const {
  // LoweringINT8(rewriter, op, asymmetric);
  auto new_type = getQuantInt4Type(op.getOutput());
  LoweringScatterElements(rewriter, op, new_type);
}
void ScatterElementsLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::ScatterElementsOp op) const {
  // lowering_common_bf16<tpu::ScatterElementsOp>(rewriter, op);
  auto new_type = getQuantFloatType<mlir::BFloat16Type>(op.getOutput());
  LoweringScatterElements(rewriter, op, new_type);
}

void ScatterElementsLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::ScatterElementsOp op) const {
  // lowering_common_f16<tpu::ScatterElementsOp>(rewriter, op);
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  LoweringScatterElements(rewriter, op, new_type);
}

void ScatterElementsLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::ScatterElementsOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ScatterElementsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::ScatterElementsOp op) const {
  // lowering_common<tpu::ScatterElementsOp>(rewriter, op.getOperation(),
  //                                     op.getOutput().getType());
  auto new_type = op.getOutput().getType();
  LoweringScatterElements(rewriter, op, new_type);
}

} // namespace bm1684x
} // namespace tpu_mlir
