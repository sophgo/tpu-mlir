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

static void LoweringTopK(PatternRewriter &rewriter, top::TopKOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::TopKOp>(op, op.getResultTypes(), operands,
                                             attrs);
    return;
  }
  std::vector<Type> new_types;
  for (auto out : op.getResults()) {
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }
  rewriter.replaceOpWithNewOp<tpu::TopKOp>(op, new_types, operands, attrs);
  return;
}

void TopKLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}
void TopKLowering::LoweringINT4(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void TopKLowering::LoweringINT8(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}

void TopKLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getBF16Type());
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}

void TopKLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getF16Type());
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}

void TopKLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TopKOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
