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

void CastLowering::LoweringF32(PatternRewriter &rewriter,
                               top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringINT8(PatternRewriter &rewriter, top::CastOp op,
                                bool asymmetric) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringF16(PatternRewriter &rewriter,
                               top::CastOp op) const {
  lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                               op.output().getType());
}

void CastLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::CastOp op) const {
  if (Quant::isUniformQuantized(op.input(), op.output()) == false) {
    lowering_common<tpu::CastOp>(rewriter, op.getOperation(),
                                 op.output().getType());
    return;
  }
  int64_t i_zeropoint, o_zeropoint;
  double i_scale, o_scale;
  Quant::getScaleAndZeroPoint(op.input(), i_scale, i_zeropoint, true);
  Quant::getScaleAndZeroPoint(op.output(), o_scale, o_zeropoint, true);
  std::vector<NamedAttribute> attrs;
  if (i_scale == o_scale) {
    int zero_diff = i_zeropoint - o_zeropoint;
    attrs.push_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(-zero_diff)));
    rewriter.replaceOpWithNewOp<tpu::AddConstOp>(op.getOperation(), op.output().getType(),
                                                 op.input(), attrs);
    return;
  }

  std::vector<Value> operands;
  if (i_zeropoint != 0) {
    auto sub_zp = do_add_zp(op.input(), rewriter.getI32Type(), -i_zeropoint);
    operands.push_back(sub_zp);
  } else {
    operands.push_back(op.input());
  }
  o_scale = i_scale / o_scale;
  int64_t multiplier;
  int64_t shift;
  QuantizeMultiplier(o_scale, &multiplier, &shift);
  auto ctx = op.getContext();
  attrs.push_back(rewriter.getNamedAttr("multiplier",
                                        rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(-shift)));
  attrs.push_back(
      rewriter.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFlite_Lshift)));

  rewriter.replaceOpWithNewOp<tpu::RequantIntOp>(op.getOperation(), op.output().getType(),
                                                 operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
