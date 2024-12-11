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

static void LoweringBatchNormBwd(PatternRewriter &rewriter,
                                 top::BatchNormBwdOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.reserve(3);
  for (int i = 0; i < 3; i++) {
    auto out = op.getResult(i);
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }
  rewriter.replaceOpWithNewOp<tpu::BatchNormBwdOp>(op, new_types, operands,
                                                   attrs);
  return;
}

void BatchNormBwdLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::BatchNormBwdOp op) const {
  // rewriter.replaceOpWithNewOp<tpu::BatchNormBwdOp>(
  //     op, op->getResultTypes(), op->getOperands(), op->getAttrs());
  LoweringBatchNormBwd(rewriter, op, rewriter.getF32Type());
}

void BatchNormBwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op) const {
  LoweringF32(rewriter, op);
}

void BatchNormBwdLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::BatchNormBwdOp op) const {
  // LoweringF32(rewriter, op);
  LoweringBatchNormBwd(rewriter, op, rewriter.getF16Type());
}

void BatchNormBwdLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::BatchNormBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::BatchNormBwdOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormBwdLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::BatchNormBwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
