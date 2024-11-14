//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace bm1684;

void deconv_weight_transform(int ic, int oc, int kh, int kw,
                             const void *weight_orig, const void *weight_trans,
                             int type_bytes) {
  int trans_offset;
  int hw = kw * kh;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < hw; k_idx++) {
        int orig_offset = ic_idx * oc * hw + k_idx + oc_idx * hw;
        orig_offset = oc_idx * ic * hw + ic_idx * hw + k_idx;
        switch (type_bytes) {
        case 4:
          trans_offset = oc_idx * align_up(ic, 2) * hw + ic_idx / 2 * hw * 2 +
                         k_idx * 2 + ic_idx % 2;
          *((float *)weight_trans + trans_offset) =
              *((float *)weight_orig + orig_offset);
          break;
        default:
          llvm_unreachable("wrong conv weight data type");
        }
      }
    }
  }
}

// use for deconv depthwise weight tensor
template <typename T>
static void deconv_weight_transform(int oc, int ic, int h, int w, T *src,
                                    T *dst) {
  int hw = h * w;
  for (int idxc = 0; idxc < oc * ic; ++idxc) {
    for (int idxh = 0; idxh < h; ++idxh) {
      for (int idxw = 0; idxw < w; ++idxw) {
        memcpy(dst + idxh * w + idxw, src + hw - idxh * w - 1 - idxw,
               sizeof(T));
      }
    }
    src += hw;
    dst += hw;
  }
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, int8_t>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  if (!attr.is_dw) {
    int new_size =
        attr.oc * attr.g * (align_up(attr.ic, 4l)) * attr.kh * attr.kw;
    std::vector<int64_t> new_shape = {
        1, attr.oc * attr.g, attr.kh * attr.kw * align_up(attr.ic, 4l), 1};
    auto new_type =
        RankedTensorType::get(new_shape, filter_type.getElementType());
    auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
    for (int oc_idx = 0; oc_idx < attr.oc; oc_idx++) {
      for (int ic_idx = 0; ic_idx < attr.ic; ic_idx++) {
        for (int k_idx = 0; k_idx < attr.kh * attr.kw; k_idx++) {
          int orig_offset = oc_idx * attr.ic * attr.kh * attr.kw +
                            ic_idx * attr.kh * attr.kw + k_idx;
          int trans_offset =
              oc_idx * align_up(attr.ic, 4l) * attr.kw * attr.kh +
              ic_idx / 4 * attr.kh * attr.kw * 4 + k_idx * 4 + ic_idx % 4;
          filter_new->at(trans_offset) = filter_int8->at(orig_offset);
        }
      }
    }
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  } else {
    int new_size = attr.g * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
    std::vector<int64_t> new_shape = {1, attr.g, attr.kh, attr.kw};
    deconv_weight_transform(1, attr.g, attr.kh, attr.kw,
                            (int8_t *)filter_int8->data(),
                            (int8_t *)filter_new->data());
    auto new_type =
        RankedTensorType::get(new_shape, filter_type.getElementType());
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float32Type>::matchAndRewriteImpl(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();
  auto attr = op.parseParam();
  auto type_bytes = 4;
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {
        1, attr.oc * attr.g, align_up(attr.ic, 2l) / 2, attr.kh * attr.kw * 2};
    int new_count =
        align_up(attr.ic, 2l) * attr.oc * attr.g * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<float>>(new_count, 0);
    deconv_weight_transform(attr.ic, attr.oc * attr.g, attr.kh, attr.kw,
                            weight_data->data(), filter_new->data(),
                            type_bytes);
    auto new_type = RankedTensorType::get(new_shape, out_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);

  } else {
    auto ic = attr.ic / attr.g;
    std::vector<int64_t> new_filter_shape = {ic, attr.oc, attr.kh, attr.kw};
    auto filter_new = std::make_shared<std::vector<float>>(
        ic * attr.oc * attr.kh * attr.kw, 0);
    deconv_weight_transform(attr.oc, ic, attr.kh, attr.kw,
                            (float *)weight_data->data(),
                            (float *)filter_new->data());
    auto new_type = RankedTensorType::get(new_filter_shape, out_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  }
  // bias op
  if (attr.with_bias) {
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}
