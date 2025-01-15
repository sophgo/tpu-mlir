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

static void LoweringGatherElements(PatternRewriter &rewriter,
                                   top::GatherElementsOp op, Type type) {
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

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp); // indices_coeff
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::GatherElementsOp>(op, type, operands,
                                                     op->getAttrs());
  return;
}

void GatherElementsLowering::LoweringF32(PatternRewriter &rewriter,
                                         top::GatherElementsOp op) const {
  auto new_type = getQuantFloatType(op.getOutput());
  LoweringGatherElements(rewriter, op, new_type);
}

void GatherElementsLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::GatherElementsOp op,
                                          bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8())
    LoweringBF16(rewriter, op);
  else
    LoweringF16(rewriter, op);
}

void GatherElementsLowering::LoweringINT4(PatternRewriter &rewriter,
                                          top::GatherElementsOp op,
                                          bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void GatherElementsLowering::LoweringBF16(PatternRewriter &rewriter,
                                          top::GatherElementsOp op) const {
  auto new_type = getQuantFloatType<mlir::BFloat16Type>(op.getOutput());
  LoweringGatherElements(rewriter, op, new_type);
}

void GatherElementsLowering::LoweringF16(PatternRewriter &rewriter,
                                         top::GatherElementsOp op) const {
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  LoweringGatherElements(rewriter, op, new_type);
}

void GatherElementsLowering::LoweringF8(PatternRewriter &rewriter,
                                        top::GatherElementsOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void GatherElementsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                               top::GatherElementsOp op) const {
  auto new_type = op.getOutput().getType();
  LoweringGatherElements(rewriter, op, new_type);
}

} // namespace bm1684x
} // namespace tpu_mlir
