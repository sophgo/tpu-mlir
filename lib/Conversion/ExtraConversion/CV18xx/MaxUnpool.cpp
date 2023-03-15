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
ConvertMaxUnpoolOp::matchAndRewrite(top::MaxUnpoolOp op,
                                   PatternRewriter &rewriter) const {
  auto mask_op = op.getMask().getDefiningOp();
  if (!isa<top::PoolMaskOp>(mask_op)) {
    return failure();
  }
  auto output_shape = module::getShape(op.getOutput());
  std::vector<int64_t> mask_shape;
  mask_shape = module::getShape(op.getMask());
  bool need_crop = false;
  if (mask_shape[3] != output_shape[3] || mask_shape[2] != output_shape[2]) {
    need_crop = true;
  }
  std::string max_unpool_name = module::getName(op.getOutput()).str();
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  std::string name = max_unpool_name + "_nearst";

  // create upsample op
  auto loc = NameLoc::get(rewriter.getStringAttr(name));
  operands.emplace_back(op.getInput());
  attrs.emplace_back(rewriter.getNamedAttr("scale_h", op.getScaleHAttr()));
  attrs.emplace_back(rewriter.getNamedAttr("scale_w", op.getScaleWAttr()));
  auto new_type = RankedTensorType::get(
      mask_shape, op.getOutput().getType().getElementType());
  auto upsample_op =
      rewriter.create<top::UpsampleOp>(loc, new_type, operands, attrs);

  // create mul op
  attrs.clear();
  operands.clear();
  if (need_crop) {
    name = max_unpool_name + "_multi";
  } else {
    name = max_unpool_name;
  }

  loc = NameLoc::get(rewriter.getStringAttr(name));
  operands.emplace_back(upsample_op);
  operands.emplace_back(op.getMask());
  auto mul_op = rewriter.create<top::MulOp>(loc, new_type, operands, attrs);

  if (need_crop) {
    // create crop op
    attrs.clear();
    operands.clear();
    name = max_unpool_name;
    loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<int64_t> crop_offset(4, 0);
    std::vector<int64_t> steps(4, 1);
    attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr(ArrayRef<int64_t>({crop_offset}))));
    attrs.emplace_back(rewriter.getNamedAttr(
        "steps", rewriter.getI64ArrayAttr(ArrayRef<int64_t>({steps}))));
    operands.emplace_back(mul_op);
    auto crop_op = rewriter.create<top::SliceOp>(
        loc, op.getOutput().getType().cast<RankedTensorType>(), operands,
        attrs);
    rewriter.replaceOp(op, {crop_op});
  } else {
    rewriter.replaceOp(op, {mul_op});
  }
  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
