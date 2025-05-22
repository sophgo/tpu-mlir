//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-maxconst"
namespace tpu_mlir {
namespace cv18xx {
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

  auto input_shape = module::getShape(op.getInput());
  int nums = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                             std::multiplies<int>());
  auto in_type = op.getInput().getType().cast<RankedTensorType>();
  auto weight_type =
      RankedTensorType::get(in_type.getShape(), rewriter.getI8Type());
  auto weight_operand = top::WeightOp::create(
      op, "const_max",
      std::vector<int8_t>(nums, to_int8(const_val, ROUNDING_HALF_UP)),
      weight_type);
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(weight_operand);
  auto new_type = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MaxOp>(op, new_type, operands);
}

void MaxConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MaxConstOp op) const {
  double const_val = op.getConstVal().convertToDouble();
  auto weight_type = op.getInput().getType().cast<RankedTensorType>();
  auto input_shape = module::getShape(op.getInput());
  int nums = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                             std::multiplies<int>());
  auto weight_operand = top::WeightOp::create(
      op, "const_max", std::vector<float>(nums, const_val), weight_type);
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(weight_operand);
  auto new_type = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<top::MaxOp>(op, new_type, operands);
}
} // namespace cv18xx
} // namespace tpu_mlir
