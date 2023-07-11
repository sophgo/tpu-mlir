//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-relu"
namespace tpu_mlir {
namespace cv18xx {

void ReluLowering::LoweringINT8(PatternRewriter &rewriter, top::ReluOp op,
                                bool asymmetric) const {
  assert(!asymmetric && "CV18xx not support asymmetric quantify");
  if (op.getReluLimit().convertToDouble() != -1) {
    LoweringBF16(rewriter, op);
  } else {
    lowering_common_int8<tpu::ReluOp>(rewriter, op, asymmetric);
  }
}

void ReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ReluOp op) const {
  if (op.getReluLimit().convertToDouble() != -1) {
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("min", rewriter.getF64FloatAttr(0.)));
    attrs.push_back(rewriter.getNamedAttr("max", op.getReluLimitAttr()));
    auto newType = getQuantBF16Type(op.getOutput());
    rewriter.replaceOpWithNewOp<tpu::ClipOp>(op, newType, op->getOperands(),
                                             attrs);
  } else {
    lowering_common_bf16<tpu::ReluOp>(rewriter, op);
  }
}
} // namespace cv18xx
} // namespace tpu_mlir
