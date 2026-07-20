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

// FlexAttention: block-sparse flash attention with non-square head dim and
// optional fp32 lse. Generic op (any model with a structured mask); Falcon is
// the first user. FAttentionOp / FAttentionLseOp and their v2/_lse kernels (in
// fattention_v2.pl) are untouched.
//
// Differences from flash_attention_v2:
//  - qk_d / v_d split: q,k use qk_d for the QK matmul; v uses v_d for the
//    AV matmul and output. When qk_d == v_d this is ordinary flash attention;
//    cross-attention with v_dim != qk_dim (e.g. anyup qk=32, v=64) is supported.
//  - block_bitmap [num_q_blocks, num_kv_blocks] (fp32 0/1): before processing a
//    (_m,_k) tile the scalar core reads the tile's bitmap cell via
//    get_gmem_addr + get_value (PPL scalar-read API, p270) and `continue`s if
//    0, skipping the fully-masked block's QK/softmax/AV compute (online-
//    softmax exact: a skipped block contributes exp(-inf)=0 to li/acc and does
//    not change mi). Bitmap granularity = flex_block (default 128);
//    block_m/block_k <= flex_block (enforced by the host tiler) so each tile
//    maps to exactly one bitmap cell.
//  - has_lse: also write fp32 logsumexp [b,qm,q_head,1] (attention sink).
// has_bitmap=0 / has_lse=0 degenerates to dense flash attention with lse off.
template <typename T>
void flash_attention_flex(
    T *ptr_out, T *ptr_q, T *ptr_k, T *ptr_v, T *ptr_mask, fp32 *ptr_lse,
    int b, int qm, int kvm, int q_head, int kv_head,
    float sqrt_d, int has_mask, int has_lse, int has_bitmap, int flex_block,
    const int core_num, const int qk_d, const int v_d, const int block_m,
    const int block_k, const int block_qh, const int block_kh) {
  const bool is_bf16 = std::is_same_v<T, bf16>;
  const float neg_inf = is_bf16 ? -1.5e10f : -15000.0f;
  // num_q_blocks / num_kv_blocks were computed here for the (now-disabled)
  // block-sparse bitmap_local. Removed: the runtime-computed shape broke the
  // PPL compile-time local-addr assignment. Block-sparse deferred.
  int head_rep = q_head / kv_head;
  int core_index = get_core_index();
  if (core_index >= core_num)
    return;

  int kv_head_per_core = div_up(kv_head, core_num);
  int q_head_per_core = kv_head_per_core * head_rep;
  int q_head_start = core_index * q_head_per_core;
  int q_head_end = min(q_head_start + q_head_per_core, q_head);

  // q,k use qk_d; v and the output/accumulator use v_d.
  dim4 q_shape = {block_qh, block_m, 1, qk_d};
  dim4 k_shape = {block_kh, block_k, 1, qk_d};
  dim4 v_shape = {block_kh, block_k, 1, v_d};
  dim4 qk_shape = {block_qh, block_m, 1, block_k};
  dim4 mask_shape = {1, block_m, 1, block_k};
  dim4 mi_shape = {block_qh, block_m, 1, 1};
  dim4 li_shape = {block_qh, block_m, 1, 1};
  dim4 acc_shape = {block_qh, block_m, 1, v_d};

  dim4 q_global_shape = {b, qm, q_head, qk_d};
  dim4 out_global_shape = {b, qm, q_head, v_d};
  dim4 k_global_shape = {b, kvm, kv_head, qk_d};
  dim4 v_global_shape = {b, kvm, kv_head, v_d};
  auto q_global_tensor = gtensor<T>(q_global_shape, GLOBAL, ptr_q);
  auto out_global_tensor = gtensor<T>(out_global_shape, GLOBAL, ptr_out);
  auto k_global_tensor = gtensor<T>(k_global_shape, GLOBAL, ptr_k);
  auto v_global_tensor = gtensor<T>(v_global_shape, GLOBAL, ptr_v);
  dim4 mask_global_shape = {b, qm, 1, kvm};
  auto mask_global_tensor = gtensor<T>(mask_global_shape, GLOBAL, ptr_mask);
  dim4 lse_global_shape = {b, qm, q_head, 1};
  auto lse_global_tensor = gtensor<fp32>(lse_global_shape, GLOBAL, ptr_lse);
  // block_bitmap local/global tensors removed (block-sparse deferred; the
  // runtime-computed shape broke PPL local-addr assignment). ptr_bitmap is
  // still passed (has_bitmap attr drives the codegen path) but unused in-body.

  for (int _b = 0; _b < b; _b += 1) {
    dim4 q_sub_shape = {1, qm, q_head, qk_d};
    dim4 q_sub_reshape = {qm, q_head, 1, qk_d};
    dim4 kv_qk_sub_shape = {1, kvm, kv_head, qk_d};
    dim4 kv_qk_sub_reshape = {kvm, kv_head, 1, qk_d};
    dim4 kv_v_sub_shape = {1, kvm, kv_head, v_d};
    dim4 kv_v_sub_reshape = {kvm, kv_head, 1, v_d};
    dim4 out_sub_shape = {1, qm, q_head, v_d};
    dim4 out_sub_reshape = {qm, q_head, 1, v_d};
    dim4 sub_offset = {_b, 0, 0, 0};
    auto q_sub_global =
        q_global_tensor.sub_view(q_sub_shape, sub_offset).view(q_sub_reshape);
    auto k_sub_global =
        k_global_tensor.sub_view(kv_qk_sub_shape, sub_offset)
            .view(kv_qk_sub_reshape);
    auto v_sub_global =
        v_global_tensor.sub_view(kv_v_sub_shape, sub_offset)
            .view(kv_v_sub_reshape);
    auto out_sub_global =
        out_global_tensor.sub_view(out_sub_shape, sub_offset)
            .view(out_sub_reshape);
    dim4 lse_sub_shape = {1, qm, q_head, 1};
    dim4 lse_sub_reshape = {qm, q_head, 1, 1};
    auto lse_sub_global =
        lse_global_tensor.sub_view(lse_sub_shape, sub_offset)
            .view(lse_sub_reshape);
    for (int _h = q_head_start; _h < q_head_end; _h += block_qh) {
      int real_q_h = min(block_qh, q_head_end - _h);
      int real_kv_h = real_q_h / head_rep;
      for (int _m = 0; _m < qm; _m += block_m) {
        int real_m = min(block_m, qm - _m);
        dim4 qi_real_local_shape = {real_q_h, real_m, 1, qk_d};
        dim4 qi_real_global_shape = {real_m, real_q_h, 1, qk_d};
        dim4 qi_offset = {_m, _h, 0, 0};
        auto qi_tensor = make_tensor<T>(q_shape, qi_real_local_shape);
        dma::load_transpose_nc(
            qi_tensor, q_sub_global.sub_view(qi_real_global_shape, qi_offset));

        dim4 mi_real_shape = {real_q_h, real_m, 1, 1};
        dim4 li_real_shape = {real_q_h, real_m, 1, 1};
        dim4 acc_real_shape = {real_q_h, real_m, 1, v_d};
        auto mi_sub_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
        auto li_sub_tensor = make_tensor<fp32>(li_shape, li_real_shape);
        auto acc_sub_tensor = make_tensor<fp32>(acc_shape, acc_real_shape);
        tiu::fill(mi_sub_tensor, neg_inf);
        tiu::zero(li_sub_tensor);
        tiu::zero(acc_sub_tensor);
        for (int _k = 0; _k < kvm; _k += block_k) {
          // Block-sparse skip is DISABLED (see /tmp/fattention_flex_blocksparse.bak.pl):
          // the bitmap_local make_tensor used a runtime-computed shape
          // (num_q_blocks), which makes the PPL compile-time local-addr
          // assignment pass skip the whole kernel -> all local addrs stay -1
          // -> codegen LUT broadcast hard-exits. Block-sparse also conflicts
          // with layer-group profiling (B1). Deferred; FlexAttention ships dense.
          ppl::enable_pipeline();
          int real_k = min(block_k, kvm - _k);
          dim4 kvi_real_local_shape = {real_kv_h, real_k, 1, qk_d};
          dim4 kvi_real_global_shape = {real_k, real_kv_h, 1, qk_d};
          dim4 vi_real_local_shape = {real_kv_h, real_k, 1, v_d};
          dim4 vi_real_global_shape = {real_k, real_kv_h, 1, v_d};
          dim4 kvi_offset = {_k, _h / head_rep, 0, 0};
          dim4 qk_real_shape = {real_q_h, real_m, 1, real_k};
          dim4 mask_real_shape = {1, real_m, 1, real_k};
          dim4 mask_offset = {_b, _m, 0, _k};

          auto ki_tensor = make_tensor<T>(k_shape, kvi_real_local_shape);
          auto vi_tensor = make_tensor<T>(v_shape, vi_real_local_shape);
          dma::load_transpose_nc(
              ki_tensor, k_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          dma::load_transpose_nc(
              vi_tensor, v_sub_global.sub_view(vi_real_global_shape, kvi_offset));
          auto mask_tensor = make_tensor<T>(mask_shape, mask_real_shape);
          if (has_mask) {
            dma::load(mask_tensor, mask_global_tensor.sub_view(mask_real_shape,
                                                               mask_offset));
          }
          dim4 qk_batch_shape = {1, real_m, 1, real_k};
          dim4 qi_batch_shape = {1, real_m, 1, qk_d};
          dim4 ki_batch_shape = {1, real_k, 1, qk_d};
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
          if (has_mask) {
            auto mask_tensor_fp32 =
                make_tensor<fp32>(mask_shape, mask_real_shape);
            tiu::cast(mask_tensor_fp32, mask_tensor);
            tiu::fadd(qk_sub_tensor, qk_sub_tensor, mask_tensor_fp32);
          }

          auto max_out = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto mi_new_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
          quick_pooling(max_out, qk_sub_tensor, &qk_shape, &qk_real_shape,
                        neg_inf, 0);
          tiu::fmax(mi_new_tensor, mi_sub_tensor, max_out);

          auto alpha = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto sub_out = make_tensor<fp32>(mi_shape, mi_real_shape);
          auto li_tmp_tensor = make_tensor<fp32>(li_shape, li_real_shape);
          tiu::fsub(sub_out, mi_sub_tensor, mi_new_tensor);
          tiu::move(mi_sub_tensor, mi_new_tensor);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real_shape);
          tiu::fmul(acc_sub_tensor, acc_sub_tensor, alpha);
          tiu::fmul(li_tmp_tensor, li_sub_tensor, alpha);
          auto sub_out1 = make_tensor<fp32>(qk_shape, qk_real_shape);
          tiu::fsub(sub_out1, qk_sub_tensor, mi_new_tensor);

          auto p_T = make_tensor<fp32>(qk_shape, qk_real_shape);
          auto sum = make_tensor<fp32>(li_shape, li_real_shape);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real_shape);

          quick_pooling(sum, p_T, &qk_shape, &qk_real_shape, 0, 1);
          tiu::fadd(li_sub_tensor, li_tmp_tensor, sum);

          // PV: p [real_m, real_k] @ v [real_k, v_d] -> pv [real_m, v_d]
          auto pv_tensor = make_tensor<fp32>(acc_shape, acc_real_shape);
          auto p_T_a16 = make_tensor<T>(qk_shape, qk_real_shape);
          tiu::cast(p_T_a16, p_T);
          dim4 pv_batch_shape = {1, real_m, 1, v_d};
          dim4 p_batch_shape = {1, real_m, 1, real_k};
          dim4 vi_batch_shape = {1, real_k, 1, v_d};
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

        // lse = mi + log(li) (computed before li is inverted to 1/li below).
        if (has_lse) {
          auto log_li = make_tensor<fp32>(li_shape, li_real_shape);
          flog(log_li, li_sub_tensor, &li_shape, &li_real_shape);
          auto lse_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
          tiu::fadd(lse_tensor, mi_sub_tensor, log_li);
          dim4 lse_real_global_shape = {real_m, real_q_h, 1, 1};
          dim4 lse_offset = {_m, _h, 0, 0};
          dma::store_transpose_nc(
              lse_sub_global.sub_view(lse_real_global_shape, lse_offset),
              lse_tensor);
        }

        auto qkvo_tensor_a16 = make_tensor<T>(acc_shape, acc_real_shape);
        tiu::fdiv(li_sub_tensor, 1.0f, li_sub_tensor, 3);
        tiu::fmul(acc_sub_tensor, acc_sub_tensor, li_sub_tensor);
        tiu::cast(qkvo_tensor_a16, acc_sub_tensor);

        dim4 qkv_offset = {_m, _h, 0, 0};
        // output is v_d wide (not qk_d); use a v_d store shape (the Q load
        // shape qi_real_global_shape is qk_d wide and must NOT be reused here).
        dim4 out_real_global_shape = {real_m, real_q_h, 1, v_d};
        dma::store_transpose_nc(
            out_sub_global.sub_view(out_real_global_shape, qkv_offset),
            qkvo_tensor_a16);
      }
    }
  }
}

__KERNEL__ void fattention_v2_bf16_flex(
    bf16 *ptr_out, bf16 *ptr_q, bf16 *ptr_k, bf16 *ptr_v, bf16 *ptr_mask,
    fp32 *ptr_lse, int b, int qm, int kvm, int q_head,
    int kv_head, float sqrt_d, int has_mask, int has_lse, int has_bitmap,
    int flex_block, const int g_core_num, const int qk_d, const int v_d,
    const int block_m, const int block_k, const int block_qh,
    const int block_kh) {
  flash_attention_flex<bf16>(
      ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, ptr_lse, b, qm, kvm,
      q_head, kv_head, sqrt_d, has_mask, has_lse, has_bitmap, flex_block,
      g_core_num, qk_d, v_d, block_m, block_k, block_qh, block_kh);
}

__KERNEL__ void fattention_v2_f16_flex(
    fp16 *ptr_out, fp16 *ptr_q, fp16 *ptr_k, fp16 *ptr_v, fp16 *ptr_mask,
    fp32 *ptr_lse, int b, int qm, int kvm, int q_head,
    int kv_head, float sqrt_d, int has_mask, int has_lse, int has_bitmap,
    int flex_block, const int g_core_num, const int qk_d, const int v_d,
    const int block_m, const int block_k, const int block_qh,
    const int block_kh) {
  flash_attention_flex<fp16>(
      ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, ptr_lse, b, qm, kvm,
      q_head, kv_head, sqrt_d, has_mask, has_lse, has_bitmap, flex_block,
      g_core_num, qk_d, v_d, block_m, block_k, block_qh, block_kh);
}
