//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

// `ptr_mask` is a *fixed-size* lower-triangular causal mask template of shape
// [mask_size, mask_size]: 0 on/below the diagonal and -inf above. The Q tile
// size along M is `block_m` (expected to be `mask_size` or `mask_size/2`);
// the kernel reuses this single template for every (block_m x block_k_iter)
// tile that straddles the causal diagonal by loading a shifted sub-view, so
// the global mask footprint stays at O(mask_size^2) regardless of qm / kvm.
//
// Sub-view fit constraint:
//   for a straddling block with D = kv_q_off + _m - _k, we load
//   template[shift_m : shift_m + real_m, shift_k : shift_k + real_k]
//   with (shift_m, shift_k) = (max(D,0), max(-D,0)). Both ends must lie in
//   [0, mask_size], which simplifies to
//       real_m + real_k - 1 <= mask_size .
//   Since block_m and block_k_iter are both <= mask_size (and typically
//   block_m + block_k_iter <= mask_size + 1 for block_m == mask_size/2),
//   every straddling block can fit a shifted sub-view of the template.
template <typename T>
void fattention_prefill(T *ptr_out, T *ptr_q, T *ptr_k, T *ptr_v,
                        fp32 *ptr_mask, int b, int qm, int kvm, float sqrt_d,
                        float pool_pad, const int core_num, const int q_head,
                        const int kv_head, const int d, const int block_m,
                        const int block_h, const int mask_size) {
  int head_rep = q_head / kv_head;
  int core_index = get_core_index();
  if (core_index >= core_num)
    return;
  int kv_head_per_core = div_up(kv_head, core_num);

  int kv_head_start = core_index * kv_head_per_core;
  int kv_head_end = min(kv_head_start + kv_head_per_core, kv_head);

  int block_k_iter = max(min(block_m, kvm / 2), 1);
  assert(block_h >= 1);
  assert(block_m <= mask_size);
  // `block_h` is the number of KV heads per block; the corresponding Q-head
  // block is `block_h * head_rep`.
  int block_kv_h_iter = block_h;
  // Causal-diagonal offset: for query qi in [_m, _m+real_m) attending to key
  // ki in [_k, _k+real_k), the position is unmasked iff ki <= qi + kv_q_off.
  int kv_q_off = kvm - qm;

  int block_q_h = block_h * head_rep;
  int block_k_h = block_h;
  dim4 q_shape = {block_q_h, block_m, 1, d};
  dim4 kv_shape = {block_k_h, block_m, 1, d};
  dim4 qk_shape = {block_q_h, block_m, 1, block_m};
  dim4 mask_shape = {1, block_m, 1, block_m};

  dim4 mi_shape = {block_q_h, block_m, 1, 1};
  dim4 li_shape = {block_q_h, block_m, 1, 1};
  dim4 acc_shape = {block_q_h, block_m, 1, d};

  dim4 qo_global_shape = {b, qm, q_head, d};
  auto q_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_q);
  auto out_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_out);

  dim4 kv_global_shape = {b, kvm, kv_head, d};
  auto k_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_k);
  auto v_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_v);

  // Causal mask template, reused across all (b, _m, _k) tiles.
  dim4 mask_global_shape = {1, mask_size, 1, mask_size};
  auto mask_global_tensor = gtensor<fp32>(mask_global_shape, GLOBAL, ptr_mask);

  for (int _b = 0; _b < b; _b += 1) {
    dim4 q_sub_shape = {1, qm, q_head, d};
    dim4 q_sub_reshape = {qm, q_head, 1, d};
    dim4 kv_sub_shape = {1, kvm, kv_head, d};
    dim4 kv_sub_reshape = {kvm, kv_head, 1, d};
    dim4 sub_offset = {_b, 0, 0, 0};
    auto q_sub_global =
        q_global_tensor.sub_view(q_sub_shape, sub_offset).view(q_sub_reshape);
    auto k_sub_global =
        k_global_tensor.sub_view(kv_sub_shape, sub_offset).view(kv_sub_reshape);
    auto v_sub_global =
        v_global_tensor.sub_view(kv_sub_shape, sub_offset).view(kv_sub_reshape);
    auto out_sub_global =
        out_global_tensor.sub_view(q_sub_shape, sub_offset).view(q_sub_reshape);
    for (int _kh = kv_head_start; _kh < kv_head_end; _kh += block_kv_h_iter) {
      int real_kv_h = min(block_kv_h_iter, kv_head_end - _kh);
      real_kv_h = min(real_kv_h, block_h);
      int real_q_h = real_kv_h * head_rep;
      int _h = _kh * head_rep;
      for (int _m = 0; _m < qm; _m += block_m) {
        int real_m = min(block_m, qm - _m);
        dim4 qi_real_local_shape = {real_q_h, real_m, 1, d};
        dim4 qi_real_global_shape = {real_m, real_q_h, 1, d};
        dim4 qi_offset = {_m, _h, 0, 0};
        auto qi_tensor = make_tensor<T>(q_shape, qi_real_local_shape);
        dma::load_transpose_nc(
            qi_tensor, q_sub_global.sub_view(qi_real_global_shape, qi_offset));

        dim4 mi_real_shape = {real_q_h, real_m, 1, 1};
        dim4 li_real_shape = {real_q_h, real_m, 1, 1};
        dim4 acc_real_shape = {real_q_h, real_m, 1, d};
        auto mi_sub_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
        auto li_sub_tensor = make_tensor<fp32>(li_shape, li_real_shape);
        auto acc_sub_tensor = make_tensor<fp32>(acc_shape, acc_real_shape);
        tiu::fill(mi_sub_tensor, pool_pad);
        tiu::zero(li_sub_tensor);
        tiu::zero(acc_sub_tensor);
        // Causal-mask K boundaries hoisted out of the inner loop:
        //   K in [0, K_unmasked_end)        : every (qi, ki) is unmasked
        //   K in [K_unmasked_end, K_end)    : straddling, mask template needed
        //   K in [K_end, kvm)               : every (qi, ki) is masked, skip
        // Derived from "unmasked iff ki <= qi + kv_q_off":
        //   smallest qi = _m            -> last all-unmasked ki = _m + kv_q_off
        //   largest  qi = _m+real_m-1   -> last any-unmasked  ki =
        //   _m+real_m-1+kv_q_off
        int K_unmasked_end = min(kvm, max(0, _m + kv_q_off + 1));
        int K_end = min(kvm, max(0, _m + real_m + kv_q_off));
        for (int _k = 0; _k < K_end; _k += block_k_iter) {
          ppl::enable_pipeline();
          int real_k = min(block_k_iter, K_end - _k);
          // Block needs the mask iff it crosses the causal diagonal.
          bool block_no_mask = (_k + real_k) <= K_unmasked_end;
          int D = kv_q_off + _m - _k;
          dim4 kvi_real_local_shape = {real_kv_h, real_k, 1, d};
          dim4 kvi_real_global_shape = {real_k, real_kv_h, 1, d};
          dim4 kvi_offset = {_k, _h / head_rep, 0, 0};
          dim4 qk_real_shape = {real_q_h, real_m, 1, real_k};
          dim4 mask_real_shape = {1, real_m, 1, real_k};
          // Shift into the lower-triangular template so that
          //   template[qi_local + a][ki_local + b] gives the desired mask.
          // Need (b - a) == -D. Choose non-negative a, b with the smallest
          // shifts that fit inside [mask_size, mask_size].
          int shift_m = D > 0 ? D : 0;
          int shift_k = D < 0 ? -D : 0;
          dim4 mask_offset = {0, shift_m, 0, shift_k};

          auto ki_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          auto vi_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          dma::load_transpose_nc(
              ki_tensor,
              k_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          dma::load_transpose_nc(
              vi_tensor,
              v_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          auto mask_tensor = make_tensor<fp32>(mask_shape, mask_real_shape);
          if (!block_no_mask) {
            dma::load(mask_tensor, mask_global_tensor.sub_view(mask_real_shape,
                                                               mask_offset));
          }
          dim4 qk_batch_shape = {1, real_m, 1, real_k};
          dim4 qi_batch_shape = {1, real_m, 1, d};
          dim4 ki_batch_shape = {1, real_k, 1, d};
          auto qk_sub_tensor = make_tensor<fp32>(qk_shape, qk_real_shape);
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_q_offset = {i, 0, 0, 0};
            dim4 batch_k_offset = {i / head_rep, 0, 0, 0};
            auto qk_tensor_batch =
                qk_sub_tensor.sub_view(qk_batch_shape, batch_q_offset);
            auto qi_tensor_batch =
                qi_tensor.sub_view(qi_batch_shape, batch_q_offset);
            auto ki_tensor_batch =
                ki_tensor.sub_view(ki_batch_shape, batch_k_offset);

            tiu::fmm2(qk_tensor_batch, qi_tensor_batch, ki_tensor_batch, false,
                      true, false);
          }
          tiu::fmul(qk_sub_tensor, qk_sub_tensor, sqrt_d);
          if (!block_no_mask) {
            tiu::fadd(qk_sub_tensor, qk_sub_tensor, mask_tensor);
          }

          auto max_out = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto mi_new_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
          quick_pooling(max_out, qk_sub_tensor, &qk_shape, &qk_real_shape,
                        pool_pad, 0);

          tiu::fmax(mi_new_tensor, mi_sub_tensor, max_out);

          auto alpha = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto sub_out = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto li_tmp_tensor = make_tensor<fp32>(li_shape, li_real_shape);
          tiu::fsub(sub_out, mi_sub_tensor, mi_new_tensor);
          tiu::move(mi_sub_tensor, mi_new_tensor);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real_shape);
          // broadcast mul (w)
          tiu::fmul(acc_sub_tensor, acc_sub_tensor, alpha);
          tiu::fmul(li_tmp_tensor, li_sub_tensor, alpha);
          auto sub_out1 = make_tensor<fp32>(qk_shape, qk_real_shape);
          // broadcast sub (w)
          tiu::fsub(sub_out1, qk_sub_tensor, mi_new_tensor);

          auto p_T = make_tensor<fp32>(qk_shape, qk_real_shape);
          auto sum = make_tensor<fp32>(li_shape, li_real_shape);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real_shape);

          quick_pooling(sum, p_T, &qk_shape, &qk_real_shape, 0, 1);
          tiu::fadd(li_sub_tensor, li_tmp_tensor, sum);

          auto pv_tensor = make_tensor<fp32>(acc_shape, acc_real_shape);
          auto p_T_a16 = make_tensor<T>(qk_shape, qk_real_shape);
          tiu::cast(p_T_a16, p_T);
          dim4 pv_batch_shape = {1, real_m, 1, d};
          dim4 p_batch_shape = {1, real_m, 1, real_k};
          dim4 vi_batch_shape = {1, real_k, 1, d};
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_p_offset = {i, 0, 0, 0};
            dim4 batch_v_offset = {i / head_rep, 0, 0, 0};
            auto pv_tensor_batch =
                pv_tensor.sub_view(pv_batch_shape, batch_p_offset);
            auto p_tensor_batch =
                p_T_a16.sub_view(p_batch_shape, batch_p_offset);
            auto vi_tensor_batch =
                vi_tensor.sub_view(vi_batch_shape, batch_v_offset);

            tiu::fmm2(pv_tensor_batch, p_tensor_batch, vi_tensor_batch);
          }
          tiu::fadd(acc_sub_tensor, acc_sub_tensor, pv_tensor);
        }

        auto qkvo_tensor_a16 = make_tensor<T>(acc_shape, acc_real_shape);
        tiu::fdiv(li_sub_tensor, 1.0f, li_sub_tensor, 3);
        // broadcast mul (w)
        tiu::fmul(acc_sub_tensor, acc_sub_tensor, li_sub_tensor);
        tiu::cast(qkvo_tensor_a16, acc_sub_tensor);

        dim4 qkv_offset = {_m, _h, 0, 0};
        dma::store_transpose_nc(
            out_sub_global.sub_view(qi_real_global_shape, qkv_offset),
            qkvo_tensor_a16);
      }
    }
  }
}

__KERNEL__ void
fattention_prefill_bf16(bf16 *ptr_out, bf16 *ptr_q, bf16 *ptr_k, bf16 *ptr_v,
                        fp32 *ptr_mask, int b, int qm, int kvm, float sqrt_d,
                        int keep_dim, const int g_core_num, const int q_head,
                        const int kv_head, const int d, const int block_m,
                        const int block_h, const int mask_size) {
  fattention_prefill<bf16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm,
                           sqrt_d, -1.5e10f, g_core_num, q_head, kv_head, d,
                           block_m, block_h, mask_size);
}

__KERNEL__ void fattention_prefill_f16(fp16 *ptr_out, fp16 *ptr_q, fp16 *ptr_k,
                                       fp16 *ptr_v, fp32 *ptr_mask, int b,
                                       int qm, int kvm, float sqrt_d,
                                       int keep_dim, const int g_core_num,
                                       const int q_head, const int kv_head,
                                       const int d, const int block_m,
                                       const int block_h, const int mask_size) {
  fattention_prefill<fp16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm,
                           sqrt_d, -15000.0f, g_core_num, q_head, kv_head, d,
                           block_m, block_h, mask_size);
}
