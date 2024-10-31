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

void MaxConstTryLowering::Lowering(PatternRewriter &rewriter,
                                   top::MaxConstOp op) const {
  if (!isa_shape_subnet_op(op))
    return;
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("type", rewriter.getStringAttr("Max")));
  auto constI32 = i32_array_t(new std::vector<int32_t>(1, 0));
  constI32->data()[0] =
      static_cast<int64_t>(op.getConstVal().convertToDouble());
  auto weight_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32, true));
  auto weight_op = top::WeightOp::create(op, "i64", *constI32, weight_type);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(weight_op);
  Type new_type =
      RankedTensorType::get(module::getShape(op.getOutput()),
                            IntegerType::get(op.getOutput().getContext(), 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeArithOp>(op, new_type, operands, attrs);
}

void MaxConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MaxConstOp op) const {
  lowering_common_f32<tpu::MaxConstOp>(rewriter, op);
}

void MaxConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MaxConstOp op, bool asymmetric) const {

  auto in = op.getInput();
  auto out = op.getInput();
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  module::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);

  int multiplier, rshift;
  get_scale_and_shift_positive(in_scale / out_scale, multiplier, rshift, 8);
  double const_val = op.getConstVal().convertToDouble();
  const_val = static_cast<int>(round(const_val / out_scale)) << rshift;
  op.setConstValAttr(rewriter.getF64FloatAttr(const_val));

  lowering_common_int8<tpu::MaxConstOp>(rewriter, op, asymmetric);
}

void MaxConstLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::MaxConstOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void MaxConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MaxConstOp op) const {
  lowering_common_bf16<tpu::MaxConstOp>(rewriter, op);
}

void MaxConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MaxConstOp op) const {
  lowering_common_f16<tpu::MaxConstOp>(rewriter, op);
}

void MaxConstLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::MaxConstOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MaxConstOp op) const {
  lowering_common<tpu::MaxConstOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
