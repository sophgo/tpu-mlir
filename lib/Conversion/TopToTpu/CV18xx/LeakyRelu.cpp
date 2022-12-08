//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-leakyrelu"
namespace tpu_mlir {
namespace cv18xx {
void LeakyReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LeakyReluOp op) const {
  lowering_common_bf16<tpu::LeakyReluOp>(rewriter, op);
}
void LeakyReluLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {

  int multiplier, rshift;
  get_scale_and_shift(op.alpha().convertToDouble(), multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));

  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType,
                                                Value(op.input()), attrs);
}
}
}
