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

static void LoweringLayerNorm(PatternRewriter &rewriter, top::LayerNormOp op,
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

  std::vector<Type> new_types;
  new_types.reserve(3);
  auto out = op.getResult();
  if (type.isF16()) {
    new_types.push_back(getQuantF16Type(out));
  } else if (type.isBF16()) {
    new_types.push_back(getQuantBF16Type(out));
  } else {
    new_types.push_back(out.getType());
  }
  rewriter.replaceOpWithNewOp<tpu::LayerNormOp>(op, new_types, opds, attrs);
  return;
}

void LayerNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
}

void LayerNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  if (!module::isMARS3() && !module::isSGTPUV8()) {
    LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
  } else {
    LoweringLayerNorm(rewriter, op, rewriter.getBF16Type());
  }
}

void LayerNormLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void LayerNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LayerNormOp op) const {
  if (module::isBM1688() || module::isSG2380()) {
    LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
  } else {
    LoweringLayerNorm(rewriter, op, rewriter.getBF16Type());
  }
}

void LayerNormLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
}

void LayerNormLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
}

void LayerNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::LayerNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
