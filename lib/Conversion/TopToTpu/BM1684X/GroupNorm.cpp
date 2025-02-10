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

static void LoweringGroupNorm(PatternRewriter &rewriter, top::GroupNormOp op,
                              Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> opds;
  opds.reserve(3);
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
  auto none = module::getNoneOp(op);
  opds.push_back(none);
  opds.push_back(none);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  Type new_type;
  if (type.isF16()) {
    new_type = getQuantF16Type(op.getResult());
  } else if (type.isBF16()) {
    new_type = getQuantBF16Type(op.getResult());
  } else {
    new_type = op.getResult().getType();
  }
  rewriter.replaceOpWithNewOp<tpu::GroupNormOp>(op, new_type, opds, attrs);
}

void GroupNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::GroupNormOp op) const {
  LoweringGroupNorm(rewriter, op, rewriter.getF32Type());
}

void GroupNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::GroupNormOp op,
                                     bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8())
    LoweringGroupNorm(rewriter, op, rewriter.getBF16Type());
  else
    LoweringGroupNorm(rewriter, op, rewriter.getF32Type());
}

void GroupNormLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::GroupNormOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void GroupNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::GroupNormOp op) const {
  if (module::isMARS3() || module::isSGTPUV8())
    LoweringGroupNorm(rewriter, op, rewriter.getBF16Type());
  else
    LoweringGroupNorm(rewriter, op, rewriter.getF32Type());
}

void GroupNormLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::GroupNormOp op) const {
  LoweringGroupNorm(rewriter, op, rewriter.getF16Type());
}

void GroupNormLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::GroupNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void GroupNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::GroupNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
