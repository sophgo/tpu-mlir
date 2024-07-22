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

void conv3d_weight_transform_bm1684(int IC, int OC, int KT, int KH, int KW,
                                    const void *weight_orig,
                                    const void *weight_trans, int method,
                                    int type_bytes) {

  if (type_bytes == 4) {
    if (method == 0) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = kt * (OC * KW * align_up(IC * KH, 2)) +
                                oc * (KW * align_up(IC * KH, 2)) +
                                kw * align_up(IC * KH, 2) + ic * KH + kh;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else if (method == 1) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = kt * (OC * KH * KW * align_up(IC, 2)) +
                                oc * (KH * KW * align_up(IC, 2)) +
                                kh * (KW * align_up(IC, 2)) +
                                kw * align_up(IC, 2) + ic;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else if (method == 2) {
      for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          for (int kt = 0; kt < KT; ++kt) {
            for (int kh = 0; kh < KH; ++kh) {
              for (int kw = 0; kw < KW; ++kw) {
                long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                                kt * (KH * KW) + kh * KW + kw;
                long long dst = oc * KH * KW * align_up(IC * KT, 2) +
                                kh * KW * align_up(IC * KT, 2) +
                                kw * align_up(IC * KT, 2) + ic * KT + kt;
                *((float *)weight_trans + dst) = *((float *)weight_orig + src);
              }
            }
          }
        }
      }
    } else {
      llvm_unreachable("wrong conv weight data type");
    }
  } else if (type_bytes == 1) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int ic = 0; ic < IC; ++ic) {
        for (int kt = 0; kt < KT; ++kt) {
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              long long src = oc * (IC * KT * KH * KW) + ic * (KT * KH * KW) +
                              kt * (KH * KW) + kh * KW + kw;
              long long dst = oc * KH * KW * align_up(IC * KT, 4) +
                              kh * KW * align_up(IC * KT, 4) +
                              kw * align_up(IC * KT, 4);
              dst += (ic * KT + kt);
              if (method == 0) {
              } else if (method == 1) {
                // dst += (kt * IC + ic);
              } else {
                llvm_unreachable("wrong conv weight data type");
              }
              *((char *)weight_trans + dst) = *((char *)weight_orig + src);
            }
          }
        }
      }
    }
  } else {
    llvm_unreachable("wrong conv weight data type");
  }
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, int8_t>::matchAndRewriteImpl(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();
  const int method = attr.ic / attr.groups > 10 || attr.dh > 1;

  auto type_bytes = 1;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());
  std::vector<int64_t> new_shape = {1, attr.oc, attr.kd * attr.kh * attr.kw,
                                    align_up(attr.ic / attr.groups, 4l)};
  int new_count = align_up(attr.ic / attr.groups, 4l) * attr.oc * attr.kd *
                  attr.kh * attr.kw;
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_count, 0);
  conv3d_weight_transform_bm1684(attr.ic / attr.groups, attr.oc, attr.kd,
                                 attr.kh, attr.kw, filter_int8->data(),
                                 filter_new->data(), method, type_bytes);
  auto new_type = RankedTensorType::get(new_shape, filter_type);
  auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                          "reorderd", *filter_new, new_type);
  op->setOperand(1, new_filter);

  // bias op
  if (attr.has_bias) {
    auto bias_type = module::getElementType(op.getBias());
    int64_t bias_shape[5] = {1, attr.oc, 1, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
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
    // f32 local layer shape, only change shape info for layer group,
    // the real weight reorder will be done in GroupPostTransformPass
    filter_shape[0] = attr.kd;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.kh * attr.kw;
    filter_shape[3] = align_up(attr.ic / attr.groups, 2l);
    filter_shape[4] = 1;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }
  // bias op
  if (attr.has_bias) {
    auto bias_type = module::getElementType(op.getBias());
    int64_t bias_shape[5] = {1, attr.oc, 1, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}
