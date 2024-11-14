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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

// convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
template <typename T>
static void reshape_coeff_for_broadcast_channel(
    std::shared_ptr<std::vector<T>> &coeff, std::vector<int64_t> &shape,
    bool align = false, bool isINT4Conv = false) {
  int64_t n, c, h, w, eu_num;
  module::getNCHW(shape, n, c, h, w);

  if (n != 1 || h != 1 || c <= BM168x::NPU_NUM) {
    return;
  }
  eu_num = BM168x::eu_num(isINT4Conv ? 0.5 : sizeof(T));
  auto old_w_align = align_up(w, eu_num);
  int64_t new_c = BM168x::NPU_NUM;
  // convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
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

  coeff = coeff_new;
  assert(shape.size() > 2);
  shape.assign(shape.size(), 1);
  shape[1] = new_c;
  shape.back() = new_w;
}

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape,
                           bool isINT4Conv = false) {
  int64_t oc, ic, kh, kw;
  module::getNCHW(shape, oc, ic, kh, kw);
  auto type_bytes = sizeof(T);
  int64_t IC_PARALLEL = BM168x::ic_num(isINT4Conv ? 0.5 : type_bytes);
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
  assert(shape.size() > 2);
  filter = filter_new;
  shape.assign(shape.size(), 1);
  shape[1] = oc;
  if ((new_ic * new_hw > 65535) && (shape.size() == 4)) {
    shape[shape.size() - 1] = new_hw;
    shape[shape.size() - 2] = new_ic;
  } else {
    shape.back() = new_ic * new_hw;
  }
}

template <typename T>
static void reshape_coeff_for_3ic(std::shared_ptr<std::vector<T>> &weight,
                                  std::vector<int64_t> &shape,
                                  int64_t use_3ic_optimize,
                                  bool isINT4Conv = false) {
  int64_t oc, ic, kh, kw;
  module::getNCHW(shape, oc, ic, kh, kw);
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
  filter_reorder(weight, shape, isINT4Conv);
}

static void
compact_coeff_for_int4(std::shared_ptr<std::vector<int8_t>> &weight_nIC,
                       std::vector<int64_t> &shape, bool isINT4Conv = true) {
  int64_t N, C, H, W, K;
  module::getNCHW(shape, N, C, H, W);
  if (isINT4Conv) {
    // conv shape : (1, oc, 1, DIV_UP(ic, nIC) * w * nIC) --> (1, oc, 1,
    // DIV_UP(oc, nIC) * w * nIC/2  )
    auto new_w = align_up(W, (int64_t)2) / 2;
    auto filter_new = std::make_shared<std::vector<int8_t>>(new_w * C, 0);
    int col_size = align_up(W, (int64_t)2);
    for (uint i = 0; i < C; i++) {
      for (uint j = 0; j < W; j++) {
        int src_index = i * W + j;
        int dst_index = (i * col_size + j) >> 1;
        if ((j & 1) == 0) {
          filter_new->at(dst_index) = (weight_nIC->at(src_index) & 0x0f);
        } else {
          filter_new->at(dst_index) &= 0x0f;
          filter_new->at(dst_index) |= (weight_nIC->at(src_index) << 4);
        }
      }
    }
    assert(shape.size() > 2);
    shape.assign(shape.size(), 1);
    shape[1] = C;
    shape.back() = new_w;
    weight_nIC = filter_new;
  } else {
    // MMII shape : ( K, N) --> (K, N/2) ,  for (M,K) *(K,N)
    assert(shape.size() == 2);
    K = shape[0];
    N = shape[1];
    auto new_n = align_up(N, (int64_t)2) / 2;
    auto filter_new = std::make_shared<std::vector<int8_t>>(K * new_n, 0);
    int col_size = align_up(N, (int64_t)2);
    for (uint i = 0; i < K; i++) {
      for (uint j = 0; j < N; j++) {
        int src_index = i * N + j;
        int dst_index = (i * col_size + j) >> 1;
        if ((j & 1) == 0) {
          filter_new->at(dst_index) = (weight_nIC->at(src_index) & 0x0f);
        } else {
          filter_new->at(dst_index) &= 0x0f;
          filter_new->at(dst_index) |= (weight_nIC->at(src_index) << 4);
        }
      }
    }
    shape.assign(shape.size(), 1);
    shape[0] = K;
    shape.back() = N;
    weight_nIC = filter_new;
  }
}

} // namespace tpu
} // namespace tpu_mlir
