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
ConvertMaxPoolWithMaskOp::matchAndRewrite(top::MaxPoolWithMaskOp op,
                                   PatternRewriter &rewriter) const {
  auto kernel_shape = module::getI64Array(op.getKernelShape());
  assert(kernel_shape->size() == 2 &&
         kernel_shape->at(0) == kernel_shape->at(1));
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }

  // create max_pool op
  auto max_pool_op = rewriter.create<top::MaxPoolOp>(
      op->getLoc(), op.getOutput().getType().cast<RankedTensorType>(),
      ValueRange{op.getInput()}, attrs);
  rewriter.setInsertionPointAfter(max_pool_op);

  // create pool mask op
  attrs.clear();
  attrs.emplace_back(rewriter.getNamedAttr(
      "scale", rewriter.getI64IntegerAttr(kernel_shape->at(0))));
  std::string name = module::getName(op.getMask()).str() + "_convert";
  auto loc = NameLoc::get(rewriter.getStringAttr(name));
  auto input_shape = module::getShape(op.getInput());
  std::vector<int64_t> mask_shape = input_shape.vec();
  mask_shape[2] = align_up(mask_shape[2], kernel_shape->at(0));
  mask_shape[3] = align_up(mask_shape[3], kernel_shape->at(0));

  auto pool_mask_type =
      RankedTensorType::get(mask_shape, rewriter.getF32Type());
  auto pool_mask_op = rewriter.create<top::PoolMaskOp>(
      loc, pool_mask_type, ValueRange{op.getInput()}, attrs);
  op.getMask().replaceAllUsesWith(pool_mask_op.getOutput());
  rewriter.replaceOp(op, {max_pool_op, pool_mask_op});
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
