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

static void LoweringLSTM(PatternRewriter &rewriter, top::LSTMOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    if (isa<top::WeightOp>(opd.getDefiningOp())) {
      auto weightOp = opd.getDefiningOp<top::WeightOp>();
      if (type.isBF16()) {
        operands.push_back(weightOp.clone_bf16(op));
      } else if (type.isF16()) {
        operands.push_back(weightOp.clone_f16(op));
      } else {
        operands.push_back(opd);
      }
    } else {
      operands.push_back(opd);
    }
  }
  // add buffer
  operands.push_back(Module::getNoneOp(op));
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::LSTMOp>(op, op.getResultTypes(), operands,
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
  rewriter.replaceOpWithNewOp<tpu::LSTMOp>(op, new_types, operands, attrs);
  return;
}

void LSTMLowering::LoweringF32(PatternRewriter &rewriter,
                               top::LSTMOp op) const {
  LoweringLSTM(rewriter, op, rewriter.getF32Type());
}

void LSTMLowering::LoweringINT8(PatternRewriter &rewriter, top::LSTMOp op,
                                bool asymmetric) const {
  LoweringLSTM(rewriter, op, rewriter.getF32Type());
}

void LSTMLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::LSTMOp op) const {
  // LoweringLSTM(rewriter, op, rewriter.getBF16Type());
  LoweringLSTM(rewriter, op, rewriter.getF32Type());
}

void LSTMLowering::LoweringF16(PatternRewriter &rewriter,
                               top::LSTMOp op) const {
  // LoweringLSTM(rewriter, op, rewriter.getF16Type());
  LoweringLSTM(rewriter, op, rewriter.getF32Type());
}

void LSTMLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::LSTMOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
