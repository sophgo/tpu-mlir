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
static void filter_rotate(std::shared_ptr<std::vector<T>> &filter,
                          std::shared_ptr<std::vector<T>> &filter_rotated,
                          int64_t groups, int64_t oc, int64_t ic, int64_t kd,
                          int64_t kh, int64_t kw) {
  ic /= groups;
  std::vector<int64_t> strides = {ic * kd * kh * kw, kd * kh * kw, kh * kw, kw,
                                  1};
  for (auto n = 0; n < oc; ++n) {
    for (auto c = 0; c < ic; ++c) {
      for (auto d = 0; d < kd; ++d) {
        for (auto h = 0; h < kh; ++h) {
          for (auto w = 0; w < kw; ++w) {
            auto src_idx = n * strides[0] + c * strides[1] + d * strides[2] +
                           h * strides[3] + w;
            auto dst_idx = n * strides[0] + c * strides[1] +
                           (kd - 1 - d) * strides[2] +
                           (kh - 1 - h) * strides[3] + (kw - 1 - w);
            filter_rotated->data()[dst_idx] = filter->data()[src_idx];
          }
        }
      }
    }
  }
}

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape, int64_t kd) {
  // (oc, ic, kt, kh, kw) -> (oc, (ic*kt)/IC_PARALLEL, kh, kw,
  // IC_PARALLEL)
  int64_t OC, IC, KH, KW;
  module::getNCHW(shape, OC, IC, KH, KW);
  int64_t KT = kd;
  auto type_bytes = sizeof(T);
  int64_t IC_PARALLEL = BM168x::ic_num(type_bytes);
  std::vector<int64_t> filter_shape = {
      1, OC, ceiling_func(IC * KT, IC_PARALLEL), KH * KW, IC_PARALLEL};
  auto filter_new = std::make_shared<std::vector<T>>(
      OC * ceiling_func(IC * KT, IC_PARALLEL) * KH * KW * IC_PARALLEL, 0);
  for (int oc = 0; oc < OC; ++oc) {
    for (int ic = 0; ic < ceiling_func(IC * KT, IC_PARALLEL); ++ic) {
      for (int khw = 0; khw < KH * KW; ++khw) {
        for (int inner = 0; inner < IC_PARALLEL; ++inner) {
          if (ic * IC_PARALLEL + inner >= IC * KT)
            break;
          long long src = oc * IC * KT * KH * KW +
                          (ic * IC_PARALLEL + inner) * KH * KW + khw;
          long long dst = oc * align_up(IC * KT, IC_PARALLEL) * KH * KW +
                          ic * IC_PARALLEL * KH * KW + khw * IC_PARALLEL +
                          inner;
          filter_new->at(dst) = filter->at(src);
        }
      }
    }
  }
  filter = filter_new;
  shape = filter_shape;
}

// refer to net_compiler: bool BM1684XCoeffArranger::DeconvWeightArr(GraphEdge*
// edge)
template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, int8_t>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();

  // filter op
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<uint8_t>();
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.is_dw) {
    filter_shape = {1, attr.oc, attr.kd, attr.kh, attr.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
  } else {
    // kernel_rotate
    auto newFilter =
        std::make_shared<std::vector<uint8_t>>(filter_i8->size(), 0);
    filter_rotate(filter_i8, newFilter, attr.g, attr.ic, attr.oc, attr.kd,
                  attr.kh, attr.kw);
    filterOp.update(*newFilter, newFilter->size());

    // kernel_reorder
    auto filter_rotate_i8 = filterOp.read<uint8_t>();
    filter_reorder(filter_rotate_i8, filter_shape, attr.kd);
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp = top::WeightOp::create(
        op, "_reordered", *filter_rotate_i8, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (module::isWeight(op.getBias())) {
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[5] = {1, attr.oc, 1, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Deconv3DOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();

  // filter op
  if (dyn_cast<top::WeightOp>(op.getFilter().getDefiningOp())) {
    auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
    auto filter_u16 = filterOp.read<uint16_t>();

    // do kernel_rotate first
    auto newFilter =
        std::make_shared<std::vector<uint16_t>>(filter_u16->size(), 0);
    filter_rotate(filter_u16, newFilter, attr.g, attr.ic, attr.oc, attr.kd,
                  attr.kh, attr.kw);
    filterOp.update(*newFilter, newFilter->size());
    // end kernel_rotate

    auto filter_rotate_u16 = filterOp.read<uint16_t>();
    auto filter_type = module::getStorageType(op.getFilter());
    std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                         attr.kw};
    if (attr.is_dw) {
      // unspoort now
      llvm::errs() << "not support for 3D now, pleaser check filter_shape\n";
      filter_shape = {1, attr.oc, attr.kh, attr.kw};
      auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
      op.getFilter().setType(new_filter_type);
    } else {
      filter_reorder(filter_rotate_u16, filter_shape, attr.kd);
      auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
      auto newFilterOp = top::WeightOp::create(
          op, "_reordered", *filter_rotate_u16, new_filter_type);
      op->setOperand(1, newFilterOp);
    }

  } else if (dyn_cast<top::InputOp>(op.getFilter().getDefiningOp())) {
    auto filter_type = module::getStorageType(op.getFilter());
    auto filter_shape =
        op.getFilter().getType().cast<RankedTensorType>().getShape();
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
    int64_t bias_shape[5] = {1, attr.oc, 1, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, Float16Type>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();

  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, Float32Type>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();

  // filter op
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {1, attr.oc, attr.ic / attr.g,
                                       attr.kd * attr.kh * attr.kw};
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
  // kernel rorate
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter = filterOp.read<float>();
  auto filter_rotated = std::make_shared<std::vector<float>>(filter->size(), 0);
  filter_rotate(filter, filter_rotated, attr.g, attr.oc, attr.ic, attr.kd,
                attr.kh, attr.kw);
  auto rotatedFilterOp =
      top::WeightOp::create(op, "_rotated", *filter_rotated, new_filter_type);
  op->setOperand(1, rotatedFilterOp);

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
    int64_t bias_shape[5] = {1, attr.oc, 1, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}
