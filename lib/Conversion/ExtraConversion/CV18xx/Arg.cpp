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
ConvertArgmaxOp::matchAndRewrite(top::ArgOp op,
                                   PatternRewriter &rewriter) const {
  if (op.getMode() == "ArgMin") {
    return failure();
  }
  auto axis = op.getAxis();
  auto shape = module::getShape(op.getInput());
  if (axis == shape.size() -1) {
    return failure();
  }

  std::vector<int64_t> order(shape.size());
  std::iota(order.begin(), order.end(), 0);
  order.erase(order.begin() + axis);
  order.push_back(axis);

  auto op_name = module::getName(op);

  // add transposeOp
  std::vector<int64_t> output_shape;
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  for (uint32_t i = 0; i < shape.size(); ++i) {
    output_shape.emplace_back(shape[order[i]]);
  }

  operands.emplace_back(op.getInput());
  auto loc = NameLoc::get(rewriter.getStringAttr(op_name + "_permute"));
  attrs.emplace_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  auto cali_type = op.getInput().getType().cast<RankedTensorType>().getElementType();
  auto type = RankedTensorType::get(output_shape, cali_type);
  auto permute_op =
      rewriter.create<top::PermuteOp>(loc, type, operands, attrs);

  // add argmax
  op.setOperand(permute_op.getResult());
  op.setAxis(output_shape.size() - 1);
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
