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

static void deconv3d_weight_transform(const std::vector<int64_t> &weight_shape,
                                      const void *weight_orig,
                                      void *weight_trans, int groups) {
  /*
    weight orig shape with groups (g, OC(oc/g), ic/g, kh, kw)
    weight orig shape no groups (oc, ic, kh, kw)
  */
  int IC = weight_shape[1] / groups;
  int OC = weight_shape[0];
  int KT = weight_shape[2];
  int KH = weight_shape[3];
  int KW = weight_shape[4];
  for (int gp = 0; gp < groups; ++gp) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int ic = 0; ic < IC; ++ic) {
        for (int kt = 0; kt < KT; ++kt) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              uint64_t src = (gp * OC + oc) * (IC * KT * KH * KW) +
                             ic * (KT * KH * KW) + kt * (KH * KW) + kh * KW +
                             kw;
              uint64_t dst = gp * OC * KH * KW * align_up(IC * KT, 2) +
                             oc * KH * KW * align_up(IC * KT, 2) +
                             kh * KW * align_up(IC * KT, 2) +
                             kw * align_up(IC * KT, 2) + ic * KT + kt;
              *((float *)weight_trans + dst) = *((float *)weight_orig + src);
            }
          }
        }
      }
    }
  }
}

template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, Float32Type>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32()) {
    return failure();
  }
  auto attr = op.parseParam();
  auto out_type = module::getStorageType(op.getOutput());
  std::vector<int64_t> filter_shape = {attr.oc / attr.g, attr.ic, attr.kd,
                                       attr.kh, attr.kw};
  std::vector<int64_t> new_filter_shape = {
      attr.g, attr.oc / attr.g, attr.kh * attr.kw,
      align_up((attr.ic / attr.g * attr.kd), 2), 1};
  int new_filter_count = new_filter_shape[0] * new_filter_shape[1] *
                         new_filter_shape[2] * new_filter_shape[3];
  auto new_filter = std::make_shared<std::vector<float>>(new_filter_count, 0);
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  deconv3d_weight_transform(filter_shape, weight_data->data(),
                            new_filter->data(), attr.g);
  auto new_type = RankedTensorType::get(new_filter_shape, out_type);
  auto new_filterOp = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *new_filter, new_type);
  op->setOperand(1, new_filterOp);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Deconv3DOp, int8_t>::matchAndRewriteImpl(
    tpu::Deconv3DOp op, PatternRewriter &rewriter) const {
  llvm_unreachable("not support");
  return failure();
}
