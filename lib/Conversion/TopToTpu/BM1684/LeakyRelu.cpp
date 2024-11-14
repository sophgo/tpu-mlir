//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void LeakyReluLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::LeakyReluOp op) const {
  lowering_common_f32<tpu::LeakyReluOp>(rewriter, op);
}

void LeakyReluLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LeakyReluOp op,
                                     bool asymmetric) const {
  int multiplier, rshift;
  get_scale_and_shift(op.getAlpha().convertToDouble(), multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LeakyReluOp>(op, newType,
                                                Value(op.getInput()), attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
