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
static void LoweringConvBwdWeight(PatternRewriter &rewriter,
                                  top::ConvBwdWeightOp op, Type type) {
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
  new_types.reserve(3);
  auto out = op.getResult();
  if (type.isF16()) {
    new_types.push_back(getQuantF16Type(out));
  } else if (type.isBF16()) {
    new_types.push_back(getQuantBF16Type(out));
  } else {
    new_types.push_back(out.getType());
  }

  printf("xxxx2, len:%d\n", (int)opds.size());
  rewriter.replaceOpWithNewOp<tpu::ConvBwdWeightOp>(op, new_types, opds, attrs);
  return;
}

void ConvBwdWeightLowering::LoweringF32(PatternRewriter &rewriter,
                                        top::ConvBwdWeightOp op) const {
  // lowering_common_f32<tpu::ConvBwdWeightOp>(rewriter, op);
  LoweringConvBwdWeight(rewriter, op, rewriter.getF32Type());
}
void ConvBwdWeightLowering::LoweringINT4(PatternRewriter &rewriter,
                                         top::ConvBwdWeightOp op,
                                         bool asymmetric) const {
  // LoweringINT8(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}
void ConvBwdWeightLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::ConvBwdWeightOp op,
                                         bool asymmetric) const {
  // lowering_common_int8<tpu::ConvBwdWeightOp>(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}

void ConvBwdWeightLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::ConvBwdWeightOp op) const {
  LoweringF32(rewriter, op);
}

void ConvBwdWeightLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::ConvBwdWeightOp op) const {
  LoweringF32(rewriter, op);
}

void ConvBwdWeightLowering::LoweringF8(PatternRewriter &rewriter,
                                       top::ConvBwdWeightOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void ConvBwdWeightLowering::LoweringQuantized(PatternRewriter &rewriter,
                                              top::ConvBwdWeightOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
