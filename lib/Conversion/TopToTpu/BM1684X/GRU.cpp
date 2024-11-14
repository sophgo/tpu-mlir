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

static void LoweringGRU(PatternRewriter &rewriter, top::GRUOp op, Type type) {
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
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::GRUOp>(op, op.getResultTypes(), operands,
                                            op->getAttrs());
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
  rewriter.replaceOpWithNewOp<tpu::GRUOp>(op, new_types, operands,
                                          op->getAttrs());
  return;
}

void GRULowering::LoweringF32(PatternRewriter &rewriter, top::GRUOp op) const {
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}

void GRULowering::LoweringINT8(PatternRewriter &rewriter, top::GRUOp op,
                               bool asymmetric) const {
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}
void GRULowering::LoweringINT4(PatternRewriter &rewriter, top::GRUOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void GRULowering::LoweringBF16(PatternRewriter &rewriter, top::GRUOp op) const {
  // LoweringGRU(rewriter, op, rewriter.getBF16Type());
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}

void GRULowering::LoweringF16(PatternRewriter &rewriter, top::GRUOp op) const {
  // LoweringGRU(rewriter, op, rewriter.getF16Type());
  LoweringGRU(rewriter, op, rewriter.getF32Type());
}

void GRULowering::LoweringF8(PatternRewriter &rewriter, top::GRUOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void GRULowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::GRUOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
