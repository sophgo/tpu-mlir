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
ConvertAvgPoolOp::matchAndRewrite(top::AvgPoolOp op,
                                   PatternRewriter &rewriter) const {
  const size_t kernel_size = op.getKernelShape().size();
  if (kernel_size != 2) {
    return failure();
  }
  Value input_val = op.getOperand();
  Value output_val = op.getResult();
  int64_t on, oc, oh, ow;
  module::getNCHW(output_val, on, oc, oh, ow, false);

  uint64_t lmem_size = 32 * 1024;
  int64_t n, c, ih, iw;
  module::getNCHW(input_val, n, c, ih, iw, false);

  int64_t output_bytes = 2 * on * oh * ow;
  int64_t input_bytes = 2 * n * ih * iw;

  if ((uint64_t)(input_bytes + output_bytes) < lmem_size ||
                  !(oh == 1 && ow == 1)) {
    return failure();
  }
  std::string name = module::getName(output_val).str();
  auto elementType_ = output_val.getType().cast<TensorType>().getElementType();
  std::vector<int> h_slices;
  int h_slice_size = (int)((lmem_size - output_bytes) / (2 * n * iw));
  int total_h = ih;
  while (total_h > 0) {
    if (total_h > h_slice_size) {
      total_h -= h_slice_size;
      h_slices.push_back(h_slice_size);
    } else {
      h_slices.push_back(total_h);
      break;
    }
  }

  rewriter.setInsertionPointAfterValue(input_val);
  int offset = 0;
  std::vector<Value> concat_operands;
  for (auto &slice : h_slices) {
    std::vector<Value> slice_operands;
    slice_operands.emplace_back(input_val);
    std::vector<NamedAttribute> slice_attrs;
    slice_attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr({0, 0, offset, 0})));
    slice_attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1})));
    offset += slice;
    std::string slice_name = "slice_" + name + std::to_string(offset);
    auto slice_loc = NameLoc::get(rewriter.getStringAttr(slice_name));
    auto slice_type = RankedTensorType::get({n, c, slice, iw}, elementType_);
    auto slice_op = rewriter.create<top::SliceOp>(slice_loc, slice_type,
                                                  slice_operands, slice_attrs);
    auto slice_out = slice_op.getResult();

    rewriter.setInsertionPointAfterValue(slice_out);
    std::vector<Value> small_pool_operands;
    small_pool_operands.emplace_back(slice_out);
    std::vector<NamedAttribute> small_pool_attrs;
    small_pool_attrs.emplace_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr({slice, iw})));
    small_pool_attrs.emplace_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
    small_pool_attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    std::string small_pool_name = "pool_" + name + std::to_string(offset);
    auto small_pool_loc = NameLoc::get(rewriter.getStringAttr(small_pool_name));
    auto small_pool_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
    auto small_pool_op = rewriter.create<top::AvgPoolOp>(
        small_pool_loc, small_pool_type, small_pool_operands, small_pool_attrs);
    auto small_pool_out = small_pool_op.getResult();
    concat_operands.emplace_back(small_pool_out);
    rewriter.setInsertionPointAfterValue(small_pool_out);
  }

  std::vector<NamedAttribute> concat_attrs;
  concat_attrs.emplace_back(
      rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(2)));
  int h_slices_num = h_slices.size();
  std::vector<int64_t> multilpier_arr(h_slices_num, 1);
  std::vector<int64_t> rshifts_arr(h_slices_num, 0);
  std::string concat_name = "concat_" + name;
  auto concat_loc = NameLoc::get(rewriter.getStringAttr(concat_name));
  auto concat_type =
      RankedTensorType::get({n, c, h_slices_num, 1}, elementType_);
  auto concat_op = rewriter.create<top::ConcatOp>(
      concat_loc, concat_type, concat_operands, concat_attrs);
  auto concat_out = concat_op.getResult();
  rewriter.setInsertionPointAfterValue(concat_out);

  std::vector<NamedAttribute> final_attrs;
  final_attrs.emplace_back(rewriter.getNamedAttr(
      "kernel_shape", rewriter.getI64ArrayAttr({h_slices_num, 1})));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
  auto final_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
  rewriter.replaceOpWithNewOp<top::AvgPoolOp>(
      op.getOperation(), final_type, ValueRange{concat_out}, final_attrs);
  return success();
}
}
}
