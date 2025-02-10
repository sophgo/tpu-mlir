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

static void LoweringPixelNorm_FP(PatternRewriter &rewriter, top::PixelNormOp op,
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
  rewriter.replaceOpWithNewOp<tpu::PixelNormOp>(op, new_type, opds, attrs);
  return;
}

static void LoweringPixelNorm_INT8(PatternRewriter &rewriter,
                                   top::PixelNormOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> opds;
  opds.reserve(3);
  const int nInputs = op->getNumOperands();
  auto none = module::getNoneOp(op);
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    if (opd.getDefiningOp() == none) {
      opds.push_back(none);
    } else if (module::isWeight(opd)) {
      auto weightOp = opd.getDefiningOp<top::WeightOp>();
      if (type.isBF16()) {
        opds.push_back(weightOp.clone_bf16(op));
      } else if (type.isF16()) {
        opds.push_back(weightOp.clone_f16(op));
      } else {
        opds.push_back(opd);
      }
    } else {
      auto ctx = opd.getContext();
      OpBuilder builder(ctx);
      builder.setInsertionPointAfterValue(opd);
      auto name = module::getName(opd).str();
      auto newType = getQuantInt8Type(opd, module::isAsymmetric());
      name += "_" + type_string(newType);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      auto castOp = builder.create<tpu::CastOp>(loc, newType, opd);
      opds.push_back(castOp.getOutput());
    }
  }
  // auto none = module::getNoneOp(op);
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
  rewriter.replaceOpWithNewOp<tpu::PixelNormOp>(op, new_type, opds, attrs);
  return;
}

void PixelNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::PixelNormOp op) const {
  LoweringPixelNorm_FP(rewriter, op, rewriter.getF32Type());
}

void PixelNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::PixelNormOp op) const {
  LoweringPixelNorm_FP(rewriter, op, rewriter.getBF16Type());
}

void PixelNormLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::PixelNormOp op) const {
  if (module::isCalibratedType(op.getInput()) == false) {
    LoweringPixelNorm_FP(rewriter, op, rewriter.getF32Type());
    return;
  }
  auto cali_type = module::getCalibratedType(op.getInput());
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  auto shape = module::getShape(op.getInput());
  auto channel = shape[1];
  // assume half distance is (max - min)
  auto limit = std::pow((max - min) / 2, 2) * channel / 2;
  if (limit > 65504.0) { // F16 Max
    LoweringPixelNorm_FP(rewriter, op, rewriter.getF32Type());
    return;
  }
  LoweringPixelNorm_FP(rewriter, op, rewriter.getF16Type());
}

void PixelNormLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::PixelNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PixelNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::PixelNormOp op,
                                     bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    LoweringPixelNorm_INT8(rewriter, op, rewriter.getBF16Type());
  } else {
    LoweringPixelNorm_INT8(rewriter, op, rewriter.getF16Type());
  }
}

void PixelNormLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::PixelNormOp op,
                                     bool asymmetric) const {
  LoweringPixelNorm_INT8(rewriter, op, rewriter.getF16Type());
}

void PixelNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::PixelNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
