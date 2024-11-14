//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

static void LoweringGRU(PatternRewriter &rewriter, top::GRUOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    operands.push_back(opd);
  }
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::GRUOp>(op, op.getResultTypes(), operands,
                                            attrs);
    return;
  }
  std::vector<Type> new_types;
  for (auto out : op.getResults()) {
    new_types.push_back(out.getType());
  }
  rewriter.replaceOpWithNewOp<tpu::GRUOp>(op, new_types, operands, attrs);
  return;
}

void GRULowering::LoweringF32(PatternRewriter &rewriter, top::GRUOp op) const {
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}

void GRULowering::LoweringINT8(PatternRewriter &rewriter, top::GRUOp op,
                               bool asymmetric) const {
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}

} // namespace bm1684
} // namespace tpu_mlir
