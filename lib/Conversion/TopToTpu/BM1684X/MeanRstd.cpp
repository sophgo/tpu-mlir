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
static void LoweringMeanRstd(PatternRewriter &rewriter, top::MeanRstdOp op,
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

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  std::vector<Type> new_types;
  new_types.reserve(6);
  for (int i = 0; i < 6; i++) {
    auto out = op.getResult(i);
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }

  rewriter.replaceOpWithNewOp<tpu::MeanRstdOp>(op, new_types, opds, attrs);
  return;
}

void MeanRstdLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MeanRstdOp op) const {
  // lowering_common_f32<tpu::MeanRstdOp>(rewriter, op);
  LoweringMeanRstd(rewriter, op, rewriter.getF32Type());
}
void MeanRstdLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::MeanRstdOp op, bool asymmetric) const {
  // LoweringINT8(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}
void MeanRstdLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MeanRstdOp op, bool asymmetric) const {
  // lowering_common_int8<tpu::MeanRstdOp>(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanRstdLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MeanRstdOp op) const {
  LoweringF32(rewriter, op);
}

void MeanRstdLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MeanRstdOp op) const {
  std::vector<Value> operands;
  std::vector<Type> new_types;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  for (auto in : op.getOperands()) {
    operands.push_back(in);
  }

  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);

  for (int i = 0; i < 6; i++) {
    auto out = op.getResult(i);
    new_types.push_back(getQuantF16Type(out));
  }
  rewriter.replaceOpWithNewOp<tpu::MeanRstdOp>(op, new_types, operands, attrs);
  // LoweringF32(rewriter, op);
}

void MeanRstdLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::MeanRstdOp op) const {
  LoweringF32(rewriter, op);
}

void MeanRstdLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MeanRstdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
