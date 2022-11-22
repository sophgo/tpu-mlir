//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;

namespace tpu_mlir {
namespace tpu {

// convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
template <typename T>
static void
reshape_coeff_for_broadcast_channel(std::shared_ptr<std::vector<T>> &coeff,
                                    std::vector<int64_t> &shape,
                                    bool align = false) {
  int64_t n, c, h, w, eu_num;
  Module::getNCHW(shape, n, c, h, w);
  if (n != 1 || h != 1 || c <= BM168x::NPU_NUM) {
    return;
  }
  eu_num = BM168x::eu_num(sizeof(T));
  auto old_w_align = align_up(w, eu_num);

  // convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
  int64_t new_c = BM168x::NPU_NUM;
  auto c2w = ceiling_func(c, new_c);
  int64_t new_w = (align ? old_w_align : w) * (c2w - 1) + w;
  auto coeff_new = std::make_shared<std::vector<T>>(new_w * new_c, 0);
  for (uint i = 0; i < c2w; i++) {
    for (uint j = 0; j < new_c; j++) {
      if (i * new_c + j >= c) {
        break;
      }
      for (uint k = 0; k < w; k++) {
        uint src_idx = (i * new_c + j) * w + k;
        uint dst_idx = j * new_w + i * (align ? old_w_align : w) + k;
        coeff_new->at(dst_idx) = coeff->at(src_idx);
      }
    }
  }
  assert(shape.size() > 2);
  shape.assign(shape.size(), 1);
  shape[1] = new_c;
  shape.back() = new_w;
  coeff = coeff_new;
}

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  int64_t oc, ic, kh, kw;
  Module::getNCHW(shape, oc, ic, kh, kw);
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
  assert(shape.size() > 2);
  shape.assign(shape.size(), 1);
  shape[1] = oc;
  shape.back() = new_ic * new_hw;
}

template <typename T>
static void reshape_coeff_for_3ic(std::shared_ptr<std::vector<T>> &weight,
                                  std::vector<int64_t> &shape,
                                  int64_t use_3ic_optimize) {
  int64_t oc, ic, kh, kw;
  Module::getNCHW(shape, oc, ic, kh, kw);
  use_3ic_optimize = use_3ic_optimize & 0x3;

  // if merge kw to ic, it need convert (oc, ic, kh, kw) to (oc, ic, kw, kh).
  if (use_3ic_optimize == 2) {
    auto khw = std::make_shared<std::vector<T>>(kh * kw, 0);
    for (uint i = 0; i < oc * ic; ++i) {
      for (uint j = 0; j < kh; ++j) {
        for (uint k = 0; k < kw; ++k) {
          khw->at(k * kh + j) = weight->at(i * kh * kw + j * kw + k);
        }
      }
      for (uint j = 0; j < kh * kw; ++j) {
        weight->at(i * kh * kw + j) = khw->at(j);
      }
    }
  }

  int64_t new_ic, new_kernel;
  switch (use_3ic_optimize) {
  case 1: // merge kh to ic
    new_ic = ic * kh;
    new_kernel = kw;
    break;
  case 2: // merge kw to ic
    new_ic = ic * kw;
    new_kernel = kh;
    break;
  case 3: // merge kh and kw to ic
    new_ic = ic * kh * kw;
    new_kernel = 1;
    break;
  default: // not merge
    new_ic = ic;
    new_kernel = kh * kw;
    break;
  }

  shape.assign(shape.size(), 1);
  shape[0] = oc;
  shape[1] = new_ic;
  shape.back() = new_kernel;
  filter_reorder(weight, shape);
}

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TPU_DIM 65535

typedef struct conv_common_spec {
  int32_t groups;
  int32_t input_c;
  int32_t output_c;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t has_bias;
  int32_t if_relu;
  float upper_limit;
  int32_t rshift;
  int32_t round_mode;
  int32_t is_asym;
  int32_t kzp_is_const;
  int32_t kzp_value;
  int32_t ipad_is_const;
  int32_t ipad_value;
  int32_t bias_sign; // For merged coeff
  int32_t use_3ic_optimize;
} conv_common_spec_t;

typedef struct conv_global_spec {
  conv_common_spec_t common;
  /**
   * merge_coeff:
   *    0: Not merge and not reshape weight and bias
   *    1. reshape and merge weight and bias as (bias, weight) align to (4, 1)
   * bytes for depthwise_fix8b or (4, 64) bytes for conv_fix8b
   *    2. reshape and merge weight, bias and requant as has bias-(requant,
   * bias, weight) align to (64, 4, 1) bytes for depthwise_fix8b or (64, 4, 64)
   * bytes for conv_fix8b or no bias-(requant, weight) align to (64, 1) bytes
   * for depthwise_fix8b or (64, 64) bytes for conv_fix8b
   */
  int32_t merge_coeff;
  int32_t weight_is_tensor;
} conv_global_spec_t;

typedef struct conv_local_spec {
  conv_common_spec_t common;
  uint32_t buffer_local_addr;
  int32_t result_add;
  int32_t unused_ht_for_input;
  int32_t unused_hb_for_input;
  int32_t unused_wl_for_input;
  int32_t unused_wr_for_input;
  int32_t group_one_conv;
  int32_t with_requant;
  int32_t merge_coeff;

  // For dynamic inference
  uint32_t concat_c;
  int32_t concat_c_idx;
  int32_t reference_id;
} conv_local_spec_t;

typedef struct conv_local_param {
  conv_local_spec_t spec;
} conv_local_param_t;

#ifdef __cplusplus
}
#endif

} // namespace tpu
} // namespace tpu_mlir
