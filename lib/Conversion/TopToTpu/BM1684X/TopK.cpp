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
  if (op.getKT())
    operands.push_back(op.getKT());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.push_back(op.getValues().getType());
  if (!module::isNone(op.getIndices())) {
    auto shape = module::getShape(op.getIndices());
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_types.push_back(new_type);
  } else {
    new_types.push_back(op.getIndices().getType());
  }
  rewriter.replaceOpWithNewOp<tpu::TopKOp>(op, new_types, operands, attrs);
  return;
}

void TopKTryLowering::Lowering(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  if (!op.getKT() ||
      !op.getKT().getDefiningOp()->hasTrait<trait::ShapeProducer>())
    return;

  LoweringTopK(rewriter, op, rewriter.getF32Type());
  return;
}

void TopKLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}
void TopKLowering::LoweringINT4(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void TopKLowering::LoweringINT8(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getBF16Type());
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getF16Type());
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TopKOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
