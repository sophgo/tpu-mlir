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

static void LoweringBatchNorm(PatternRewriter &rewriter,
                              top::BatchNormTrainOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> opds;
  opds.reserve(5);
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    if (module::isWeight(opd)) {
      auto weightOp = opd.getDefiningOp<top::WeightOp>();
      if (type.isBF16()) {
        opds.push_back(weightOp.clone_bf16(op));
      } else if (type.isF16()) {
        opds.push_back(weightOp.clone_f16(op));
      } else {
        opds.push_back(opd);
      }
    } else {
      opds.push_back(opd);
    }
  }

  // l2 buffer for running mean and var
  auto noneOp = module::getNoneOp(op);
  opds.emplace_back(noneOp);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  std::vector<Type> new_types;
  new_types.reserve(5);
  for (int i = 0; i < 5; i++) {
    auto out = op.getResult(i);
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }
  printf("xxxx2, len:%d\n", (int)opds.size());
  rewriter.replaceOpWithNewOp<tpu::BatchNormTrainOp>(op, new_types, opds,
                                                     attrs);
  return;
}

void BatchNormTrainLowering::LoweringF32(PatternRewriter &rewriter,
                                         top::BatchNormTrainOp op) const {
  LoweringBatchNorm(rewriter, op, rewriter.getF32Type());
}

void BatchNormTrainLowering::LoweringBF16(PatternRewriter &rewriter,
                                          top::BatchNormTrainOp op) const {
  LoweringBatchNorm(rewriter, op, rewriter.getBF16Type());
}

void BatchNormTrainLowering::LoweringF16(PatternRewriter &rewriter,
                                         top::BatchNormTrainOp op) const {
  LoweringBatchNorm(rewriter, op, rewriter.getF16Type());
}

void BatchNormTrainLowering::LoweringF8(PatternRewriter &rewriter,
                                        top::BatchNormTrainOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormTrainLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::BatchNormTrainOp op,
                                          bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormTrainLowering::LoweringINT4(PatternRewriter &rewriter,
                                          top::BatchNormTrainOp op,
                                          bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void BatchNormTrainLowering::LoweringQuantized(PatternRewriter &rewriter,
                                               top::BatchNormTrainOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
