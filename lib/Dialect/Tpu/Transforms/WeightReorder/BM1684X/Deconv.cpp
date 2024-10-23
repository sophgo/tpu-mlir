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
  int64_t oc, ic, kh, kw;
  module::getNCHW(shape, oc, ic, kh, kw);
  auto type_bytes = sizeof(T);
  int64_t IC_PARALLEL = BM168x::ic_num(type_bytes);
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
  auto filter_new = std::make_shared<std::vector<T>>(oc * new_ic * new_hw, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic)
            break;
          int orig_offset = oc_idx * ic * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter->at(orig_offset);
        }
      }
    }
  }
  filter = filter_new;
  shape = {1, oc, 1, new_ic * new_hw};
}

// refer to net_compiler: bool BM1684XCoeffArranger::DeconvWeightArr(GraphEdge*
// edge)
template <>
LogicalResult WeightReorder<tpu::DeconvOp, int8_t>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  // assume that ic = input_channel / groups, oc = output_channel / groups
  // for original model, deconv kernel is {groups * ic, oc, kh, kw},
  // but kernel is arranged to {groups * oc, ic, kh, kw} when adding_layer
  // here we arrange kernel to {groups * oc, ceil(ic, IC_PARALLEL), kh * kw *
  // IC_PARALLEL}
  auto attr = op.parseParam();

  // filter op
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.is_dw) {
    filter_shape = {1, attr.oc, attr.kh, attr.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
  } else {
    filter_reorder(filter_i8, filter_shape);
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp =
        top::WeightOp::create(op, "_reordered", *filter_i8, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (module::isWeight(op.getBias())) {
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::DeconvOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();

  // filter op
  if (dyn_cast<top::WeightOp>(op.getFilter().getDefiningOp())){
    auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
    auto filter_u16 = filterOp.read<uint16_t>();
    // do kernel_rotate first
    if(module::isMARS3()){
      std::vector<uint16_t> newFilter(filter_u16->size(), 0);
      for (uint32_t i = 0; i < attr.oc*(attr.ic/ attr.g); ++i) {
        int swap_count = 0;
        for (uint32_t j = 0; j < attr.kh; ++j) {
          for (uint32_t k = 0; k < attr.kw; ++k) {
            if(swap_count < (attr.kh * attr.kw / 2)){
              newFilter.at(i * attr.kh * attr.kw + j * attr.kw + k) =
                filter_u16->at(i * attr.kh * attr.kw + (attr.kh - 1 - j) * attr.kw + (attr.kw - 1 - k));
              newFilter.at(i * attr.kh * attr.kw + (attr.kh - 1 - j) * attr.kw + (attr.kw - 1 - k)) =
                filter_u16->at(i * attr.kh * attr.kw + j * attr.kw + k);
              swap_count++;
            }
          }
        }
      }
      filterOp.update(newFilter, newFilter.size());
    }
    // end kernel_rotate
    auto filter_rotate_u16 = filterOp.read<uint16_t>();
    auto filter_type = module::getStorageType(op.getFilter());
    std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                        attr.kw};
    if (attr.is_dw) {
      filter_shape = {1, attr.oc, attr.kh, attr.kw};
      auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
      op.getFilter().setType(new_filter_type);
    } else {
      filter_reorder(filter_rotate_u16, filter_shape);
      auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
      auto newFilterOp =
          top::WeightOp::create(op, "_reordered", *filter_rotate_u16, new_filter_type);
      op->setOperand(1, newFilterOp);
    }
  }
  else if (dyn_cast<top::InputOp>(op.getFilter().getDefiningOp())){
    auto filter_type = module::getStorageType(op.getFilter());
    auto filter_shape = op.getFilter().getType().cast<RankedTensorType>().getShape();
    int64_t oc, ic, kh, kw;
    module::getNCHW(filter_shape, oc, ic, kh, kw);
    auto type_bytes = sizeof(int16_t);
    int64_t IC_PARALLEL = BM168x::ic_num(type_bytes);
    auto kernel_hw = kh * kw;
    int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
    int64_t new_hw = kernel_hw * IC_PARALLEL;
    filter_shape = {1, oc, 1, new_ic * new_hw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
  }

  // bias op
  if (attr.with_bias) {
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float16Type>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float32Type>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();

  // filter op
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {1, attr.oc, attr.ic / attr.g,
                                       attr.kh * attr.kw};
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);


  if (!isa<top::WeightOp>(op.getOperand(1).getDefiningOp())) {
    // dynamic op, usually used in GAN
    rewriter.setInsertionPointAfterValue(op.getOperand(1));
    auto name = module::getName(op.getOutput());
    auto reshape_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reorder_filter"));
    auto new_type = op.getOperand(1).getType();
    auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
        reshape_loc, new_type, ValueRange{op.getOperand(1)});
    new_reshape_op.getOutput().setType(new_filter_type);

    new_reshape_op->setAttr("dynamic_weight", rewriter.getBoolAttr(true));
    op.setOperand(1, new_reshape_op);
  } else {
    op.getFilter().setType(new_filter_type);
  }

  // bias op
  if (attr.with_bias) {
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}
