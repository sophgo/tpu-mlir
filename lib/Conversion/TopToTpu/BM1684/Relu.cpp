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

void ReluLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ReluOp op) const {
  lowering_common_f32<tpu::ReluOp>(rewriter, op);
}

void ReluLowering::LoweringINT8(PatternRewriter &rewriter, top::ReluOp op,
                                bool asymmetric) const {
  std::vector<Value> operands;
  float coeff_v = 1.0;
  int64_t re_zp, o_zp;
  double re_scale, o_scale;

  auto re = op.getOperand();
  operands.push_back(re);

  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);
  module::getScaleAndZeroPoint(re, re_scale, re_zp, asymmetric);
  auto re_rshift =
      calRightShiftNumUseCblas(coeff_v, re_scale, o_scale, BITS_INT8);
  re_rshift = re_rshift < 0 ? 0 : re_rshift;
  float scale = 1.0 * (1 << re_rshift) * re_scale / o_scale;
  int8_t multiplier_int8;
  quantizeToInt8(&coeff_v, &multiplier_int8, 1, scale);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier_int8)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(re_rshift)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::ReluOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
