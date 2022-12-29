//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

namespace tpu_mlir {
namespace cv18xx {

void MaxUnpoolConvert(PatternRewriter &rewriter, top::MaxUnpoolOp &op) {
  auto mask_op = op.mask().getDefiningOp();
  if (!isa<top::MaxPoolWithMaskOp>(mask_op) && !isa<top::PoolMaskOp>(mask_op) &&
      !isa<tpu::PoolMaskOp>(mask_op)) {
    mask_op->dump();
    llvm_unreachable("not supported!");
  }
  auto output_shape = Module::getShape(op.output());
  std::vector<int64_t> mask_shape;
  if (isa<top::MaxPoolWithMaskOp>(mask_op)) {
    // if MaxPoolWithMaskOp' input shape not equal to maxUnpool's output shape
    // may cause error
    mask_shape = output_shape.vec();
    mask_shape[2] = align_up(mask_shape[2], static_cast<int64_t>(op.scale_h()));
    mask_shape[3] = align_up(mask_shape[3], static_cast<int64_t>(op.scale_w()));
  } else {
    Module::getShapeVec(op.mask(), mask_shape);
  }
  bool need_crop = false;
  if (mask_shape[3] != output_shape[3] || mask_shape[2] != output_shape[2]) {
    need_crop = true;
  }
  std::string max_unpool_name = Module::getName(op.output()).str();
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  std::string name = max_unpool_name + "_nearst";

  // create upsample op
  auto loc = NameLoc::get(rewriter.getStringAttr(name));
  operands.emplace_back(op.input());
  attrs.emplace_back(rewriter.getNamedAttr("scale_h", op.scale_hAttr()));
  attrs.emplace_back(rewriter.getNamedAttr("scale_w", op.scale_wAttr()));
  auto new_type = RankedTensorType::get(mask_shape, op.output().getType().getElementType());
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
  operands.emplace_back(op.mask());
  auto mul_op = rewriter.create<top::MulOp>(
      loc, new_type, operands, attrs);

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
        loc, op.output().getType().cast<RankedTensorType>(), operands, attrs);
    rewriter.replaceOp(op, {crop_op});
  } else {
    rewriter.replaceOp(op, {mul_op});
  }
}

void MaxUnpoolLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::MaxUnpoolOp op,
                                     bool asymmetric) const {
  MaxUnpoolConvert(rewriter, op);
}

void MaxUnpoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::MaxUnpoolOp op) const {
  MaxUnpoolConvert(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
