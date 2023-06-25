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

void AddConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::AddConstOp op) const {
  lowering_common_f32<tpu::AddConstOp>(rewriter, op);
}

void AddConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::AddConstOp op, bool asymmetric) const {
  std::vector<Value> operands;
  float coeff_v = 1.0;
  int64_t b0_zp, o_zp;
  double b0_scale, o_scale;

  auto b0 = op.getOperand();
  operands.push_back(b0);

  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);
  module::getScaleAndZeroPoint(b0, b0_scale, b0_zp, asymmetric);
  auto b0_rshift =
      calRightShiftNumUseCblas(coeff_v, b0_scale, o_scale, BITS_INT8);
  b0_rshift = b0_rshift < 0 ? 0 : b0_rshift;
  float scale = 1.0 * (1 << b0_rshift) * b0_scale / o_scale;
  int8_t multiplier_int8;
  quantizeToInt8(&coeff_v, &multiplier_int8, 1, scale);

  // b_const
  float b1_val = op.getConstVal().convertToDouble();
  int16_t b1_fix16b;
  auto b_scale = std::pow(2.0, float(b0_rshift)) / o_scale;
  quantizeToInt16(&b1_val, &b1_fix16b, 1, b_scale);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(b1_fix16b)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier_int8)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(b0_rshift)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::AddConstOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
