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
ConvertPixelNormOp::matchAndRewrite(top::PixelNormOp op,
                                    PatternRewriter &rewriter) const {
  auto shape = module::getShape(op.getInput());
  bool has_weight = true, has_bias = true;
  if (!op.getWeight().getType().isa<mlir::NoneType>()) {
    has_weight = false;
  }
  if (!op.getBias().getType().isa<mlir::NoneType>()) {
    has_bias = false;
  }
  std::vector<int64_t> new_shape;
  // (NCHW) -> (NHWC)
  std::vector<int64_t> _order(shape.size());
  std::vector<int64_t> order;
  std::iota(_order.begin(), _order.end(), 0);
  int32_t axis = 1;
  for (int32_t i = 0; i < _order.size(); i++) {
    if (i == axis) {
      continue;
    }
    order.emplace_back(_order[i]);
  }
  order.emplace_back(_order[axis]);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.emplace_back(op.getInput());
  auto inEltType = module::getElementType(op.getInput());
  auto outEltType = module::getElementType(op.getOutput());
  std::string op_name = module::getName(op.getOutput()).str();
  auto name0 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute0"));
  auto name1 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute1"));
  auto name2 = NameLoc::get(rewriter.getStringAttr(op_name + "_transposed"));
  attrs.emplace_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  for (uint32_t i = 0; i < order.size(); ++i) {
    new_shape.push_back(shape[order[i]]);
  }
  auto permuteType = RankedTensorType::get(new_shape, inEltType);
  auto permuteOp =
      rewriter.create<top::PermuteOp>(name0, permuteType, operands, attrs);
  operands.clear();
  attrs.clear();
  operands.emplace_back(permuteOp.getOutput());
  auto ic = new_shape[new_shape.size() - 1];
  if (has_weight) {
    std::vector<int64_t> wshape = module::getShape(op.getWeight());
    assert(ic == wshape[1]);
    op.getWeight().setType(
        RankedTensorType::get({ic}, module::getStorageType(op.getWeight())));
  } else {
    operands.emplace_back(op.getWeight());
  }
  if (has_bias) {
    std::vector<int64_t> bshape = module::getShape(op.getBias());
    assert(ic == bshape[1]);
    op.getBias().setType(
        RankedTensorType::get({ic}, module::getStorageType(op.getBias())));
  } else {
    operands.emplace_back(op.getBias());
  }
  attrs.emplace_back(rewriter.getNamedAttr("eps", op.getEpsAttr()));
  attrs.emplace_back(rewriter.getNamedAttr(
      "axis", rewriter.getSI32IntegerAttr(new_shape.size() - 1)));
  attrs.emplace_back(rewriter.getNamedAttr("normalized_shape",
                                           rewriter.getI64ArrayAttr({ic})));
  auto outTypes = TypeRange{RankedTensorType::get(new_shape, outEltType),
                            rewriter.getNoneType(), rewriter.getNoneType()};
  auto layerNormOp =
      rewriter.create<top::LayerNormOp>(name2, outTypes, operands, attrs);
  operands.clear();
  attrs.clear();
  operands.emplace_back(layerNormOp.getOutput());
  // (NHWC) -> (NCHW)
  axis = _order.back();
  _order.pop_back();
  _order.insert(_order.begin() + 1, axis);
  attrs.emplace_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(_order)));
  permuteType = RankedTensorType::get(shape, outEltType);
  permuteOp =
      rewriter.create<top::PermuteOp>(name1, permuteType, operands, attrs);
  rewriter.replaceOp(op, {permuteOp});
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
