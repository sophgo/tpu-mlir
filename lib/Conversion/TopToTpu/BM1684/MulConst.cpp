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

void MulConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f32<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  std::vector<Value> operands;
  int64_t b0_zp, o_zp;
  double b0_scale, o_scale;

  auto b0 = op.getOperand();
  operands.push_back(b0);

  float b1_val = op.getConstVal().convertToDouble();
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);
  module::getScaleAndZeroPoint(b0, b0_scale, b0_zp, asymmetric);
  auto b0_rshift =
      calRightShiftNumUseCblas(b1_val, b0_scale, o_scale, BITS_INT8);
  b0_rshift = b0_rshift < 0 ? 0 : b0_rshift;
  float scale = 1.0 * (1 << b0_rshift) * b0_scale / o_scale;
  int8_t multiplier_int8;
  quantizeToInt8(&b1_val, &multiplier_int8, 1, scale);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier_int8)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(b0_rshift)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::MulConstOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
