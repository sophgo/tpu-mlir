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

static void LoweringRMSNorm(PatternRewriter &rewriter, top::RMSNormOp op,
                            Type type) {
  rewriter.setInsertionPointAfter(op);
  const int nInputs = op->getNumOperands();
  std::vector<Value> opds;
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    if (isa<top::WeightOp>(opd.getDefiningOp())) {
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

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  auto out = op.getResult();
  Type new_type;
  if (type.isF16()) {
    new_type = getQuantF16Type(out);
  } else if (type.isBF16()) {
    new_type = getQuantBF16Type(out);
  } else {
    new_type = out.getType();
  }
  rewriter.replaceOpWithNewOp<tpu::RMSNormOp>(op, new_type, opds, attrs);
  return;
}

void RMSNormLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::RMSNormOp op) const {
  LoweringRMSNorm(rewriter, op, rewriter.getF32Type());
}

void RMSNormLowering::LoweringINT8(PatternRewriter &rewriter, top::RMSNormOp op,
                                   bool asymmetric) const {
  if(module::isMARS3())
    LoweringBF16(rewriter, op);
  else
    LoweringF16(rewriter, op);
}

void RMSNormLowering::LoweringINT4(PatternRewriter &rewriter, top::RMSNormOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void RMSNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::RMSNormOp op) const {
  LoweringRMSNorm(rewriter, op, rewriter.getBF16Type());
}

void RMSNormLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::RMSNormOp op) const {
  LoweringRMSNorm(rewriter, op, rewriter.getF16Type());
}

void RMSNormLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::RMSNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RMSNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::RMSNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
