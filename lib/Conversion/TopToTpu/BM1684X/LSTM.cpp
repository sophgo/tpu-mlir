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
    if (module::isWeight(opd)) {
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
  operands.push_back(module::getNoneOp(op));
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto noneOp = module::getNoneOp(op);
  for (int32_t i = 0; i < 4; i++) {
    operands.push_back(noneOp);
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
void LSTMLowering::LoweringINT4(PatternRewriter &rewriter, top::LSTMOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
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

void LSTMLowering::LoweringF8(PatternRewriter &rewriter, top::LSTMOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void LSTMLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::LSTMOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
