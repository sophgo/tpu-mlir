//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult ConvertDivOp::matchAndRewrite(top::DivOp op,
                                            PatternRewriter &rewriter) const {
  std::vector<Value> operands;
  auto input_shape1 = module::getShape(op.getInputs()[0]);
  auto input_shape2 = module::getShape(op.getInputs()[1]);

  auto weight_op =
      dyn_cast<top::WeightOp>(op.getInputs()[1].getDefiningOp());
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.emplace_back(
      rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
  operands.emplace_back(op.getInputs()[0]);
  if (weight_op) {
    assert(weight_op);
    auto const_f32 = weight_op.read<float>();
    for (auto &const_value : *const_f32) {
      const_value = 1 / const_value;
    }
    auto weight_type = weight_op.getType().cast<RankedTensorType>();
    auto new_weight_operand =
        top::WeightOp::create(op, "weight", *const_f32, weight_type);
    operands.emplace_back(new_weight_operand);
    rewriter.replaceOpWithNewOp<top::MulConstOp>(
        op.getOperation(),
        op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  } else {
    rewriter.setInsertionPointAfterValue(op.getInputs()[1]);
    std::string name =
        module::getName(op.getInputs()[1]).str() + "_reciprocal";
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<NamedAttribute> reci_attrs;
    reci_attrs.emplace_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0)));
    auto reciprocal_type =
        RankedTensorType::get(input_shape2, rewriter.getF32Type());
    auto reciprocal_op = rewriter.create<top::ReciprocalOp>(
        loc, reciprocal_type, ValueRange{op.getInputs()[1]}, reci_attrs);

    operands.emplace_back(reciprocal_op.getOutput());
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<top::MulOp>(
        op.getOperation(),
        op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  }
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
