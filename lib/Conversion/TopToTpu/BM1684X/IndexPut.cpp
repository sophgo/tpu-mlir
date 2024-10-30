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

static void LoweringIndexPut(PatternRewriter &rewriter, top::IndexPutOp op,
                           Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  auto input = op.getInput();
  auto indices = op.getIndices();
  auto values = op.getValues();
  auto stype = module::getStorageType(type);
  if (module::isWeight(input)) {
    auto wOp = input.getDefiningOp<top::WeightOp>();
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else{
      operands.push_back(input);
    }
  } else{
      operands.push_back(input);
  }

  if (module::isWeight(indices)) {
    auto wOp = indices.getDefiningOp<top::WeightOp>();
    operands.push_back(wOp.clone_int(op));
  } else{
    operands.push_back(indices);
  }

  if (module::isWeight(values)) {
    auto wOp = values.getDefiningOp<top::WeightOp>();
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else{
      operands.push_back(values);
    }
  } else{
      operands.push_back(values);
  }

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  rewriter.replaceOpWithNewOp<tpu::IndexPutOp>(op, type, operands,
                                             op->getAttrs());
  return;
}

void IndexPutLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::IndexPutOp op) const {
  auto new_type = getQuantFloatType(op.getOutput());
  LoweringIndexPut(rewriter, op, new_type);
}

void IndexPutLowering::LoweringINT8(PatternRewriter &rewriter, top::IndexPutOp op,
                                  bool asymmetric) const {
  if (module::isMARS3()) {
    LoweringBF16(rewriter, op);
  } else {
    LoweringF16(rewriter, op);
  }
}

void IndexPutLowering::LoweringINT4(PatternRewriter &rewriter, top::IndexPutOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void IndexPutLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::IndexPutOp op) const {
  auto new_type = getQuantFloatType<mlir::BFloat16Type>(op.getOutput());
  LoweringIndexPut(rewriter, op, new_type);
}

void IndexPutLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::IndexPutOp op) const {
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  LoweringIndexPut(rewriter, op, new_type);
}

void IndexPutLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::IndexPutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void IndexPutLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::IndexPutOp op) const {
  auto new_type = op.getOutput().getType();
  LoweringIndexPut(rewriter, op, new_type);
}

} // namespace bm1684x
} // namespace tpu_mlir
