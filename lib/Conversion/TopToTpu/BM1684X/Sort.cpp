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

static void loweringSort(PatternRewriter &rewriter, top::SortOp op) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(module::getNoneOp(op));
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.push_back(op.getValues().getType());
  new_types.push_back(op.getIndices().getType());
  rewriter.replaceOpWithNewOp<tpu::SortOp>(op, new_types, operands, attrs);
}

void SortLowering::LoweringF32(PatternRewriter &rewriter,
                               top::SortOp op) const {
  loweringSort(rewriter, op);
}

void SortLowering::LoweringINT8(PatternRewriter &rewriter, top::SortOp op,
                                bool asymmetric) const {
  loweringSort(rewriter, op);
}
void SortLowering::LoweringINT4(PatternRewriter &rewriter, top::SortOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SortLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SortOp op) const {
  loweringSort(rewriter, op);
}

void SortLowering::LoweringF16(PatternRewriter &rewriter,
                               top::SortOp op) const {
  loweringSort(rewriter, op);
}

void SortLowering::LoweringF8(PatternRewriter &rewriter, top::SortOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SortLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SortOp op) const {
  loweringSort(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
