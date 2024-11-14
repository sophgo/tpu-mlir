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

void MaxConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MaxConstOp op) const {
  lowering_common_f32<tpu::MaxConstOp>(rewriter, op);
}

void MaxConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MaxConstOp op, bool asymmetric) const {
  float coeff_v = 1.0;
  int64_t b0_zp, o_zp;
  double b0_scale, o_scale;
  float b1_value = op.getConstVal().convertToDouble();
  int16_t b1_value_fixed = 0;
  int8_t multiplier;

  // cal rshift, scale and fixed const max val
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);
  module::getScaleAndZeroPoint(op.getInput(), b0_scale, b0_zp, asymmetric);
  auto b0_rshift =
      calRightShiftNumUseCblas(coeff_v, b0_scale, o_scale, BITS_INT8);
  auto const_scale = std::pow(2.0, float(b0_rshift)) / o_scale;
  float overflow_ratio =
      quantizeToInt16(&b1_value, &b1_value_fixed, 1, const_scale);
  int count = 5;
  while (overflow_ratio > 0.03 && count > 0) {
    b0_rshift -= 1;
    count -= 1;
    const_scale = std::pow(2.0, float(b0_rshift)) / o_scale;
    overflow_ratio =
        quantizeToInt16(&b1_value, &b1_value_fixed, 1, const_scale);
  }
  if (b0_rshift <= 0) {
    lowering_common_f32<tpu::MaxConstOp>(rewriter, op);
    return;
  }
  b0_rshift = std::min(b0_rshift, 31);

  // input tensor
  float tenosr_scale = 1.0 * (1 << b0_rshift) * b0_scale / o_scale;
  quantizeToInt8(&coeff_v, &multiplier, 1, tenosr_scale);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "const_val", rewriter.getF64FloatAttr(b1_value_fixed)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(b0_rshift)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::MaxConstOp>(
      op, newType, ValueRange{op.getInput()}, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
