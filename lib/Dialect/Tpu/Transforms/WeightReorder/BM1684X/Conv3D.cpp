//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "ConvUtils.h"

using namespace bm1684x;

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  int32_t oc = shape[0];
  int32_t ic = shape[1];
  int32_t kd = shape[2];
  int32_t kh = shape[3];
  int32_t kw = shape[4];
  auto type_bytes = sizeof(T);
  int32_t IC_PARALLEL = BM168x::ic_num(type_bytes);
  auto kernel_hw = kh * kw;
  int32_t new_ic = ceiling_func(ic * kd, IC_PARALLEL);
  int32_t new_hw = kernel_hw * IC_PARALLEL;
  auto filter_new = std::make_shared<std::vector<T>>(oc * new_ic * new_hw, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic * kd)
            break;
          int orig_offset = oc_idx * ic * kd * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter->at(orig_offset);
        }
      }
    }
  }
  filter = filter_new;
  shape = {1, oc, new_ic, kh * kw, IC_PARALLEL};
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, int8_t>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();

  auto attr = op.parseParam();
  // filter reorder
  auto filter_type = module::getStorageType(op.getFilter());
  auto data_type = BM168x::getDataType(op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  int64_t filter_shape[5];
  // (oc, ic, kd, kh, kw) -> (1, oc, kt, ic/IC_PARALLEL, kh*kw * IC_PARALLEL)
  // int8/uint8 local layer shape, only change shape info for layer group,
  // the real weight reorder will be done in GroupPostTransformPass
  filter_shape[0] = 1;
  filter_shape[1] = attr.oc;
  filter_shape[2] = attr.kd;
  filter_shape[3] = ceiling_func((attr.ic / attr.groups), IC_PARALLEL);
  filter_shape[4] = attr.kh * attr.kw * IC_PARALLEL;
  auto new_type = RankedTensorType::get(filter_shape, filter_type);
  op.getFilter().setType(new_type);
  // bias op
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type =
        RankedTensorType::get(bias_shape, module::getStorageType(op.getBias()));
    op.getBias().setType(new_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Conv3DOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  // if (attr.is_dw || attr.groups > 1) {
  //   llvm_unreachable("depthwise should support !!");
  // }
  // filter reorder
  auto filter_type = module::getStorageType(op.getFilter());
  auto data_type = BM168x::getDataType(op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  int64_t filter_shape[5];
  if (filter_type.isF16() || filter_type.isBF16()) {
    // (oc, ic, kd, kh, kw) -> (1, oc, kt, ic/IC_PARALLEL, kh*kw * IC_PARALLEL)
    // f16/bf16 local layer shape, only change shape info for layer group,
    // the real weight reorder will be done in GroupPostTransformPass
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.kd;
    filter_shape[3] = ceiling_func((attr.ic / attr.groups), IC_PARALLEL);
    filter_shape[4] = attr.kh * attr.kw * IC_PARALLEL;
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }
  // bias op
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto bias_type = module::getStorageType(op.getBias());
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float16Type>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float32Type>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  int64_t filter_shape[5];
  if (out_type.isF32()) {
    // (oc, ic, kd, kh, kw) -> (kd, oc,ic, kh, kw)
    // f32 local layer shape, only change shape info for layer group
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.kd;
    filter_shape[3] = attr.ic / attr.groups;
    filter_shape[4] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}
