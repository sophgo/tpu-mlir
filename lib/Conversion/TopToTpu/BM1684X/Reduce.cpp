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

static void LoweringReduce(PatternRewriter &rewriter, top::ReduceOp op, Type type) {
  auto ctx = rewriter.getContext();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    operands.push_back(opd);
  }
  // add buffer
  operands.push_back(Module::getNoneOp(op));
  auto output = op->getResult(0);
  Type newType;
  if (type.isF32()) {
   newType = output.getType();
  } else if (type.isF16()) {
    newType = getQuantF16Type(output);
  } else if (type.isBF16()) {
    newType = getQuantBF16Type(output);
  }

  rewriter.replaceOpWithNewOp<tpu::ReduceOp>(op, newType, operands, op->getAttrs());
  return;
}

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getF32Type());
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                bool asymmetric) const {
  LoweringReduce(rewriter, op, rewriter.getF16Type());
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getBF16Type());
}

void ReduceLowering::LoweringF16(PatternRewriter &rewriter,
                               top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getF16Type());
}

void ReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::ReduceOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
