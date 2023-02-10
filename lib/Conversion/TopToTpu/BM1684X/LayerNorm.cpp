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
  auto none = module::getNoneOp(op);
  opds.push_back(none);
  opds.push_back(none);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::LayerNormOp>(op, op.getResultTypes(), opds,
                                                  attrs);
    return;
  }
  std::vector<Type> new_types;
  new_types.reserve(3);
  for (auto out : op.getResults()) {
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
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
  LoweringF16(rewriter, op);
}

void LayerNormLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void LayerNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LayerNormOp op) const {
  if (module::isBM1686()) {
    LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
  } else {
    LoweringLayerNorm(rewriter, op, rewriter.getBF16Type());
  }
}

void LayerNormLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  if (module::isCalibratedType(op.getInput()) == false) {
    LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
    return;
  }
  auto cali_type = module::getCalibratedType(op.getInput());
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  auto shape = module::getShape(op.getInput());
  auto axis = op.getAxis();
  auto inner_size = std::accumulate(shape.begin() + axis, shape.end(), 1,
                                    std::multiplies<int64_t>());
  // assume half distance is (max - min)
  auto limit = std::pow((max - min) / 2, 2) * inner_size / 2;
  if (limit > 65504.0) { // F16 Max
    LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
    return;
  }
  LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
}

void LayerNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::LayerNormOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
