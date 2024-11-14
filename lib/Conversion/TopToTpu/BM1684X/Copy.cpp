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

void CopyLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CopyOp op) const {
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0)));
  auto name = module::getName(op.getOutput());
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  auto mulconst_op = rewriter.create<tpu::MulConstOp>(
      mul_loc, op.getOutput().getType(), ValueRange{op.getInput()}, attrs);
  op.replaceAllUsesWith(mulconst_op.getOperation());
  rewriter.eraseOp(op);
}

void CopyLowering::LoweringINT8(PatternRewriter &rewriter, top::CopyOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void CopyLowering::LoweringINT4(PatternRewriter &rewriter, top::CopyOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void CopyLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CopyOp op) const {
  LoweringF32(rewriter, op);
}

void CopyLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CopyOp op) const {
  LoweringF32(rewriter, op);
}

void CopyLowering::LoweringF8(PatternRewriter &rewriter, top::CopyOp op) const {
  LoweringF32(rewriter, op);
}

void CopyLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CopyOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
