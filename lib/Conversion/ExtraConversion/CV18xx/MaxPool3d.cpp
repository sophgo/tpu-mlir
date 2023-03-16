//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult
convertMaxPool3D::matchAndRewrite(top::MaxPoolOp op,
                                  PatternRewriter &rewriter) const {
  std::vector<Value> operands;
  std::vector<int64_t> tmp_shape0(4, 1);
  std::vector<int64_t> tmp_shape1;
  std::vector<int64_t> _kernel;
  std::vector<int64_t> _strides;
  std::vector<int64_t> _pad;
  if (op.getKernelShape().size() != 3) {
    return failure();
  }

  auto input_shape = module::getShape(op.getInput());
  auto output_shape = module::getShape(op.getOutput());
  auto kernel = module::getI64Array(op.getKernelShape());
  auto strides = module::getI64Array(op.getStrides());
  auto pads = module::getI64Array(op.getPads());
  auto op_name = module::getName(op.getOperation()).str();
  auto type =
      op.getOutput().getType().cast<RankedTensorType>().getElementType();
  // 0. reshape [n c f h w] -> [n*c h w f].
  module::getNCHW(input_shape, tmp_shape0[0], tmp_shape0[1], tmp_shape0[2],
                  tmp_shape0[3], false);
  auto newType = RankedTensorType::get(tmp_shape0, type);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape"));
  auto reshapeOp =
      rewriter.create<top::ReshapeOp>(name_loc, newType, op->getOperands());
  // 1. do pool at last 2 dim
  for (int i = 1; i < 3; i++) {
    _kernel.push_back(kernel->at(i));
    _strides.push_back(strides->at(i));
    _pad.push_back(pads->at(i));
  }
  for (int i = 4; i < 6; i++) {
    _pad.push_back(pads->at(i));
  }
  auto dims = input_shape.size();
  tmp_shape0[2] = output_shape[dims - 2];
  tmp_shape0[3] = output_shape[dims - 1];
  newType = RankedTensorType::get(tmp_shape0, type);
  name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_0"));
  auto newOp0 = rewriter.create<top::MaxPoolOp>(
      name_loc, newType, ValueRange{reshapeOp.getOutput()}, op->getAttrs());
  newOp0->setAttr("kernel_shape", rewriter.getI64ArrayAttr(_kernel));
  newOp0->setAttr("strides", rewriter.getI64ArrayAttr(_strides));
  newOp0->setAttr("pads", rewriter.getI64ArrayAttr(_pad));
  // 2. trans [n*c f h w] -> [n*c h w f]
  std::vector<int64_t> order(tmp_shape0.size());
  std::iota(order.begin(), order.end(), 0);
  order.erase(order.begin() + tmp_shape0.size() - 3);
  order.push_back(tmp_shape0.size() - 3);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  for (auto i : order) {
    tmp_shape1.push_back(tmp_shape0[i]);
  }
  newType = RankedTensorType::get(tmp_shape1, type);
  name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_trans1"));
  auto newOp1 = rewriter.create<top::PermuteOp>(
      name_loc, newType, ValueRange{newOp0.getOutput()},
      attrs);
  // 3. do pool last dim
  tmp_shape1[tmp_shape1.size() - 1] = output_shape[output_shape.size() - 3];
  newType = RankedTensorType::get(tmp_shape1, type);
  name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_1"));
  auto newOp2 = rewriter.create<top::MaxPoolOp>(
      name_loc, newType, ValueRange{newOp1.getOutput()}, op->getAttrs());
  newOp2->setAttr("kernel_shape", rewriter.getI64ArrayAttr({1, kernel->at(0)}));
  newOp2->setAttr("strides", rewriter.getI64ArrayAttr({1, strides->at(0)}));
  newOp2->setAttr("pads",
                  rewriter.getI64ArrayAttr({0, pads->at(0), 0, pads->at(3)}));
  // 4. trans back  [n c h w f] -> [n c f h w]
  name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_2"));
  newType = RankedTensorType::get(output_shape, type);
  std::iota(order.begin(), order.end(), 0);
  order.pop_back();
  order.insert(order.begin() + tmp_shape1.size() - 3, tmp_shape1.size() - 1);
  attrs.clear();
  attrs.push_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  auto newOp3 = rewriter.create<top::PermuteOp>(
      name_loc, newType, ValueRange{newOp2.getOutput()},
      attrs);
  // 5. reshape back
  newType = RankedTensorType::get(output_shape, type);
  auto reshape_backOp = rewriter.create<top::ReshapeOp>(
      op->getLoc(), newType, ValueRange{newOp3.getOutput()});

  rewriter.replaceOp(op, {reshape_backOp.getOutput()});
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
