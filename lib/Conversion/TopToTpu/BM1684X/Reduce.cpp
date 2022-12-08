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

static void LoweringReduce(PatternRewriter &rewriter, top::ReduceOp op,
                           Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.input());
  operands.push_back(Module::getNoneOp(op));
  operands.push_back(Module::getNoneOp(op));
  mlir::Type out_type = op.output().getType();
  if (type.isF16()) {
    out_type = getQuantF16Type(op.output());
  } else if (type.isBF16()) {
    out_type = getQuantBF16Type(op.output());
  }
  rewriter.replaceOpWithNewOp<tpu::ReduceOp>(op, out_type, operands,
                                             op.getOperation()->getAttrs());
}

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getF32Type());
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  LoweringF16(rewriter, op);
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
