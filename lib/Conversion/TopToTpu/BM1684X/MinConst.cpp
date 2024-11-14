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

void MinConstTryLowering::Lowering(PatternRewriter &rewriter,
                                   top::MinConstOp op) const {
  auto prev_op = op.getInput().getDefiningOp();
  if (!prev_op->hasTrait<trait::ShapeProducer>()) {
    return;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("type", rewriter.getStringAttr("Min")));
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

void MinConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MinConstOp op) const {
  lowering_common_f32<tpu::MinConstOp>(rewriter, op);
}

void MinConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MinConstOp op, bool asymmetric) const {
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

  lowering_common_int8<tpu::MinConstOp>(rewriter, op, asymmetric);
}

void MinConstLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::MinConstOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void MinConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MinConstOp op) const {
  lowering_common_bf16<tpu::MinConstOp>(rewriter, op);
}

void MinConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MinConstOp op) const {
  lowering_common_f16<tpu::MinConstOp>(rewriter, op);
}

void MinConstLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::MinConstOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void MinConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MinConstOp op) const {
  lowering_common<tpu::MinConstOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
