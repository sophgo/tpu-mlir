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

LogicalResult
ConvertAddConstOp::matchAndRewrite(top::AddConstOp op,
                                   PatternRewriter &rewriter) const {
  std::vector<Value> operands;
  std::vector<float> weight_data;
  std::string weight_name =
      module::getName(op.getOutput()).str() + "_const_val";
  weight_data.emplace_back(op.getConstVal().convertToDouble());
  auto weight_type = RankedTensorType::get({1}, rewriter.getF32Type());
  auto weight_operand =
      top::WeightOp::create(op, weight_name, weight_data, weight_type);
  operands.emplace_back(op.getInput());
  operands.emplace_back(weight_operand);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<top::AddOp>(
      op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
