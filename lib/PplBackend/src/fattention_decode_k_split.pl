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

template <typename T>
void flash_attention_decode_k_split(T *ptr_out, T *ptr_q, T *ptr_k, T *ptr_v,
                                    T *ptr_mask, int b, int qm, int kvm,
                                    int q_head, int kv_head, float sqrt_d,
                                    int has_mask, const int core_num,
                                    const int d, const int keep_dim,
                                    const int block_m, const int block_k,
                                    const int block_qh, const int block_kh,
                                    fp32 *buffer_addr, int k_split) {
  // Decode-only kernel:
  // assert(qm == 1 && block_kh == 1);
  const bool is_bf16 = std::is_same_v<T, bf16>;
  const float neg_inf = is_bf16 ? -1.5e10f : -15000.0f;
  int head_rep = q_head / kv_head;
  int core_index = get_core_index();

  // Split-K (flash-decoding): when kv_head alone cannot fill the cores
  // (e.g. qwen3.5-moe kv_head=2 on 4 cores), the host splits the kvm axis
  // into k_split chunks so idle cores pick up extra chunks. Each core runs
  // online softmax over its own chunk into private acc/mi/li, then a
  // barrier + merge produces the final result. k_split==1 keeps the legacy
  // path. Host guarantees: block_kh==1, b*kv_head*k_split <= core_num,
  // qm <= block_m, q_head <= block_qh when k_split > 1.
  // Select-based core mapping (no if/else phi): host check_mem drops loops
  // whose bound contains the core intrinsic, but emits scf.if phis into host
  // C++. Keeping core_index inside plain arithmetic trees keeps the
  // core-dependent loops device-only (same as the original kernel).
  int kv_head_per_core = div_up(kv_head, core_num);
  int my_chunk = k_split > 1 ? core_index % k_split : 0;
  int my_head = k_split > 1 ? (core_index / k_split) % kv_head
                            : core_index * kv_head_per_core;
  int my_b_start = k_split > 1 ? core_index / (kv_head * k_split) : 0;
  int my_b_end = k_split > 1 ? min(my_b_start + 1, b) : b;
  int q_head_per_core = k_split > 1 ? head_rep : kv_head_per_core * head_rep;
  if (core_index >= core_num)
    return;
  // chunk_len is block_k-aligned (rounded DOWN) so the per-core k-loop tiling
  // is unchanged; the ragged tail lands in the last chunk as one narrow tile.
  // Host guarantees tiles >= 2 * k_split, so the floor never yields an empty
  // chunk.
  int chunk_len = k_split > 1 ? div_up(kvm, k_split) / block_k * block_k : kvm;
  // Empty chunk (k_split overshoot): idle this core. When k_split == 1,
  // my_chunk is 0 so the condition is always false and my_b_end keeps b.
  my_b_end = my_chunk * chunk_len >= kvm ? my_b_start : my_b_end;
  // Core-dependent head offset; device-only, so it must NEVER feed a tensor
  // dimension — host check_mem would otherwise emit tpu_core_index() into
  // host C++. Absolute offsets go through pointer arithmetic below.
  int my_head_off = my_head * head_rep * d;

  // q/qk/mi/li/acc keep the m dim on dim0 and the head dim on dim1, so that in
  // the decode case (real_m == 1) the per-head slice leaves m on the N axis of
  // fmm2 and the global layout [m, qh, 1, d] matches the local layout — no
  // NC transpose needed on q load / out & lse store.
  dim4 q_shape = {block_m, block_qh, 1, d};
  dim4 kv_shape = {block_kh, block_k, 1, d};
  dim4 qk_shape = {block_m, block_qh, 1, block_k};
  // mask keeps the m dim on dim0 (matching q/qk/li/acc)
  dim4 mask_shape = {block_m, 1, 1, block_k};

  dim4 mi_shape = {block_m, block_qh, 1, 1};
  dim4 li_shape = {block_m, block_qh, 1, 1};
  dim4 acc_shape = {block_m, block_qh, 1, d};

  // Static split-K workspaces. All shapes are block-dim-derived (host
  // check_mem must evaluate every dimension; core_index is device-only).
  // Unused when k_split == 1.
  //  - local {k_split, block_qh, ...}: merged result, one buffer recycled
  //    for both merge stages.
  //  - global {core_num, block_m, block_qh, ...} fp32: partial acc/mi/li
  //    spilled to gmem. mi must be spilled too — it cannot be recovered
  //    from acc (acc can underflow to zero when exp(qk - mi) underflows).
  dim4 l_acc_blk = {k_split, block_qh, 1, d};
  dim4 l_ml_blk = {k_split, block_qh, 1, 1};
  auto l_acc = make_tensor<fp32>(l_acc_blk, l_acc_blk);
  auto l_mi = make_tensor<fp32>(l_ml_blk, l_ml_blk);
  auto l_li = make_tensor<fp32>(l_ml_blk, l_ml_blk);
  dim4 g_acc_blk = {core_num, block_m, block_qh, d};
  auto g_acc = gtensor<fp32>(g_acc_blk, GLOBAL, buffer_addr);
  dim4 g_ml_blk = {core_num, block_m, block_qh, 1};
  auto g_mi = gtensor<fp32>(g_ml_blk, GLOBAL, buffer_addr + g_acc_blk.n *
                                                            g_acc_blk.c *
                                                            g_acc_blk.h *
                                                            g_acc_blk.w);
  auto g_li = gtensor<fp32>(g_ml_blk, GLOBAL,
                            buffer_addr + g_acc_blk.n * g_acc_blk.c *
                                                g_acc_blk.h * g_acc_blk.w +
                                            g_ml_blk.n * g_ml_blk.c *
                                                g_ml_blk.h * g_ml_blk.w);

  dim4 qo_global_shape = {b, qm, q_head, d};
  auto q_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_q);
  auto out_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_out);

  dim4 kv_global_shape = {b, kvm, kv_head, d};
  auto k_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_k);
  auto v_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_v);

  dim4 mask_global_shape = {b, qm, 1, kvm};
  auto mask_global_tensor = gtensor<T>(mask_global_shape, GLOBAL, ptr_mask);

  for (int _b = my_b_start; _b < my_b_end; _b += 1) {
    // k/v slices are pre-offset to this core's k chunk. The kv_head axis is
    // kept in the gtensor shape ({kvm, kv_head, 1, d}) so the gmem n-stride
    // stays kv_head * d — a {kvm, 1, 1, d} view would compute a contiguous
    // stride of d and read head-interleaved rows as consecutive k positions.
    // The core-dependent kv-head offset is a sub_view offset ({_k, my_head,
    // 0, 0}), which only appears inside the host-elided _k loop.
    int my_k_start = my_chunk * chunk_len;
    int my_k_len = k_split > 1 ? min(chunk_len, kvm - my_k_start) : kvm;
    if (my_k_len <= 0)
      my_k_len = 0;
    // The core-dependent q-head offset is applied through pointer arithmetic
    // on the global base pointers (host check_mem must never see core_index
    // inside a tensor dimension).
    T *q_b = ptr_q + (long long)_b * qm * q_head * d + my_head_off;
    T *out_b = ptr_out + (long long)_b * qm * q_head * d + my_head_off;
    T *k_b = ptr_k + ((long long)_b * kvm + my_k_start) * kv_head * d;
    T *v_b = ptr_v + ((long long)_b * kvm + my_k_start) * kv_head * d;
    // 1-batch reshapes with the head axis shifted to the block dimension.
    // Shapes stay param-derived (q_head_per_core / my_k_len are core-dependent
    // and must never reach a tensor dimension host-side). Full q_head count
    // is kept on the c axis so the gmem c-stride stays d.
    dim4 qo_blk_shape = {qm, q_head, 1, d};
    dim4 kv_blk_shape = {kvm, kv_head, 1, d};
    auto q_sub_global = gtensor<T>(qo_blk_shape, GLOBAL, q_b);
    auto out_sub_global = gtensor<T>(qo_blk_shape, GLOBAL, out_b);
    auto k_sub_global = gtensor<T>(kv_blk_shape, GLOBAL, k_b);
    auto v_sub_global = gtensor<T>(kv_blk_shape, GLOBAL, v_b);
    // Head tiles: the loop bound is core-dependent, which makes the host
    // check_mem codegen drop the whole loop (same as the original kernel).
    // The core-dependent head offset is already folded into the base pointers
    // above.
    int my_h_end = min(q_head_per_core, q_head - my_head * head_rep);
    for (int _h = 0; _h < my_h_end; _h += block_qh) {
      int real_q_h = min(block_qh, my_h_end - _h);
      int real_kv_h = real_q_h / head_rep;
      for (int _m = 0; _m < qm; _m += block_m) {
        int real_m = min(block_m, qm - _m);
        dim4 qi_real_local_shape = {real_m, real_q_h, 1, d};
        dim4 qi_real_global_shape = {real_m, real_q_h, 1, d};
        dim4 qi_offset = {_m, _h, 0, 0};
        auto qi_tensor = make_tensor<T>(q_shape, qi_real_local_shape);
        dma::load(qi_tensor,
                  q_sub_global.sub_view(qi_real_global_shape, qi_offset));

        dim4 mi_real_shape = {real_m, real_q_h, 1, 1};
        dim4 li_real_shape = {real_m, real_q_h, 1, 1};
        dim4 acc_real_shape = {real_m, real_q_h, 1, d};
        auto mi_sub_tensor = make_tensor<fp32>(mi_shape, mi_real_shape);
        auto li_sub_tensor = make_tensor<fp32>(li_shape, li_real_shape);
        auto acc_sub_tensor = make_tensor<fp32>(acc_shape, acc_real_shape);
        tiu::fill(mi_sub_tensor, neg_inf);
        tiu::zero(li_sub_tensor);
        tiu::zero(acc_sub_tensor);
        for (int _k = 0; _k < my_k_len; _k += block_k) {
          ppl::enable_pipeline();
          int real_k = min(block_k, my_k_len - _k);
          dim4 kvi_real_local_shape = {real_kv_h, real_k, 1, d};
          dim4 kvi_real_global_shape = {real_k, real_kv_h, 1, d};
          // my_head is this core's first kv head; _h / head_rep advances
          // within the core's kv-head group (0 when each core holds one kv
          // head, e.g. split-K or kv_head <= core_num).
          dim4 kvi_offset = {_k, my_head + _h / head_rep, 0, 0};
          dim4 qk_real_shape = {real_m, real_q_h, 1, real_k};
          dim4 mask_real_shape = {real_m, 1, 1, real_k};
          dim4 mask_offset = {_b, _m, 0, my_k_start + _k};

          auto ki_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          auto vi_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          dma::load_transpose_nc(
              ki_tensor,
              k_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          dma::load_transpose_nc(
              vi_tensor,
              v_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          auto mask_tensor = make_tensor<T>(mask_shape, mask_real_shape);
          if (has_mask) {
            dma::load(mask_tensor, mask_global_tensor.sub_view(mask_real_shape,
                                                               mask_offset));
          }
          // Per-head slice: m stays on dim0 (fmm2 N axis), head is sliced on
          // dim1. k/v keep head on dim0 (unchanged).
          // GQA: group head_rep q-heads sharing one kv-head so each fmm2 covers
          // head_rep heads at once (loop count real_q_h -> real_kv_h). cur_rep
          // handles the tail when real_q_h is not a multiple of head_rep.
          auto qk_sub_tensor = make_tensor<fp32>(qk_shape, qk_real_shape);
          for (int i = 0; i < real_kv_h; i++) {
            int cur_rep = min(head_rep, real_q_h - i * head_rep);
            dim4 qk_batch_shape = {real_m, cur_rep, 1, real_k};
            dim4 qi_batch_shape = {real_m, cur_rep, 1, d};
            dim4 ki_batch_shape = {1, real_k, 1, d};
            dim4 batch_q_offset = {0, i * head_rep, 0, 0};
            dim4 batch_k_offset = {i, 0, 0, 0};
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
            // broadcast add (bc dim c)
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
          // GQA: group head_rep q-heads sharing one kv-head, mirroring the QK
          // loop. cur_rep handles the tail when real_q_h is not a multiple of
          // head_rep.
          for (int i = 0; i < real_kv_h; i++) {
            int cur_rep = min(head_rep, real_q_h - i * head_rep);
            dim4 pv_batch_shape = {real_m, cur_rep, 1, d};
            dim4 p_batch_shape = {real_m, cur_rep, 1, real_k};
            dim4 vi_batch_shape = {1, real_k, 1, d};
            dim4 batch_p_offset = {0, i * head_rep, 0, 0};
            dim4 batch_v_offset = {i, 0, 0, 0};
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

        // No lse support here: the host forces k_split==1 whenever the lse
        // output is requested, so ptr_lse is always nullptr and the legacy
        // kernel owns that path (see the guard in fattention_tiling).

        if (k_split == 1) {
          auto qkvo_tensor_a16 = make_tensor<T>(acc_shape, acc_real_shape);
          tiu::fdiv(li_sub_tensor, 1.0f, li_sub_tensor, 3);
          // broadcast mul (w)
          tiu::fmul(acc_sub_tensor, acc_sub_tensor, li_sub_tensor);
          tiu::cast(qkvo_tensor_a16, acc_sub_tensor);

          dim4 qkv_offset = {_m, _h, 0, 0};
          dma::store(out_sub_global.sub_view(qi_real_global_shape, qkv_offset),
                     qkvo_tensor_a16);
        } else {
          // Split-K: keep acc/mi/li UNNORMALIZED (no 1/li, no cast). The
          // group spills partials to the gmem scratch; after one sync_all
          // barrier every core merges its (b, head) from all chunks:
          //   out = sum_s(acc_s * exp(mi_s - M)) / sum_s(li_s * exp(mi_s - M))
          // and stores its own slot straight to the output tensor.
          int my_out_core = _b * kv_head + my_head;
          dim4 zero_slot_ = {0, 0, 0, 0};
          // Group N-dim is shape-derived (sized by core_num), so the
          // device-only core_index must NOT appear in it. Per-core slot s is
          // reached through pointer arithmetic instead. Scratch slot tensors
          // keep constant block shapes (block_m x block_qh): real_*_shape is
          // inside the elided head loop and must not be referenced here.
          dim4 l_slot_acc = {block_m, block_qh, 1, d};
          dim4 l_slot_ml = {block_m, block_qh, 1, 1};
          dim4 slot_g_acc_shape = {block_m, block_qh, 1, d};
          dim4 slot_g_ml_shape = {block_m, block_qh, 1, 1};
          fp32 *my_g_acc = buffer_addr +
                           (long long)core_index * block_m * block_qh * d;
          fp32 *my_g_mi =
              buffer_addr +
              (long long)core_num * block_m * block_qh * d +
              (long long)core_index * block_m * block_qh * 1;
          fp32 *my_g_li =
              buffer_addr +
              (long long)core_num * block_m * block_qh * d +
              (long long)core_num * block_m * block_qh * 1 +
              (long long)core_index * block_m * block_qh * 1;
          gtensor<fp32> g_acc_mine(slot_g_acc_shape, GLOBAL, my_g_acc);
          gtensor<fp32> g_mi_mine(slot_g_ml_shape, GLOBAL, my_g_mi);
          gtensor<fp32> g_li_mine(slot_g_ml_shape, GLOBAL, my_g_li);
          // Spill straight from the loop-result tensors. l_acc/l_mi/l_li are
          // merge staging buffers; their slot 0 was never written by the
          // main loop, so storing l_*_mine would spill uninitialized memory.
          dma::store(g_acc_mine, acc_sub_tensor);
          dma::store(g_mi_mine, mi_sub_tensor);
          dma::store(g_li_mine, li_sub_tensor);
          // Only cores of this task group touch their own slots, but the
          // barrier is still required: TPU engines are asynchronous, so the
          // loads below must wait for the other cores' stores to complete.
          sync_all();
          // ---- merge pass 1: M = max_s(mi_s), L = sum_s(li_s * exp(mi_s - M)).
          // ---- mi must come from the spill — it cannot be recomputed from
          // ---- acc (acc can underflow to zero while mi stays finite).
          // ---- Merge scalars are single-slot tensors: mixing slot-shaped
          // ---- views with the {block_m, block_qh, 1, d} acc view gives
          // ---- c-stride 1 broadcast views; a k_split-sized group shape would
          // ---- emit c-stride k_split views that read past the buffer.
          auto l_M = make_tensor<fp32>(l_slot_ml, l_slot_ml);
          auto l_L = make_tensor<fp32>(l_slot_ml, l_slot_ml);
          {
            tiu::fill(l_M, neg_inf);
            tiu::zero(l_L);
            for (int s = 0; s < k_split; s++) {
              int p_core = my_out_core * k_split + s;
              dim4 l_slot = {s, 0, 0, 0};
              auto l_acc_s = l_acc.sub_view(l_slot_acc, l_slot);
              auto l_li_s = l_li.sub_view(l_slot_ml, l_slot);
              auto l_mi_s = l_mi.sub_view(l_slot_ml, l_slot);
              fp32 *p_acc = buffer_addr +
                            (long long)p_core * block_m * block_qh * d;
              fp32 *p_mi = buffer_addr +
                           (long long)core_num * block_m * block_qh * d +
                           (long long)p_core * block_m * block_qh * 1;
              fp32 *p_li = buffer_addr +
                           (long long)core_num * block_m * block_qh * d +
                           (long long)core_num * block_m * block_qh * 1 +
                           (long long)p_core * block_m * block_qh * 1;
              gtensor<fp32> g_acc_s(slot_g_acc_shape, GLOBAL, p_acc);
              gtensor<fp32> g_mi_s(slot_g_ml_shape, GLOBAL, p_mi);
              gtensor<fp32> g_li_s(slot_g_ml_shape, GLOBAL, p_li);
              dma::load(l_acc_s, g_acc_s);
              dma::load(l_mi_s, g_mi_s);
              dma::load(l_li_s, g_li_s);
              tiu::fmax(l_M, l_M, l_mi_s);
            }
            for (int s = 0; s < k_split; s++) {
              dim4 l_slot = {s, 0, 0, 0};
              auto l_mi_s = l_mi.sub_view(l_slot_ml, l_slot);
              auto l_li_s = l_li.sub_view(l_slot_ml, l_slot);
              // factor_s = exp(mi_s - M); li_s *= factor (kept for pass 2)
              auto l_tmp = make_tensor<fp32>(l_slot_ml, l_slot_ml);
              tiu::fsub(l_tmp, l_mi_s, l_M);
              exp_no_overflow(l_tmp, l_tmp, &l_slot_ml, &l_slot_ml);
              tiu::fmul(l_li_s, l_li_s, l_tmp);
              tiu::fadd(l_L, l_L, l_li_s);
            }
            // ---- merge pass 2: scale each chunk's acc by its factor ----
            for (int s = 0; s < k_split; s++) {
              dim4 l_slot = {s, 0, 0, 0};
              auto l_mi_s = l_mi.sub_view(l_slot_ml, l_slot);
              auto l_acc_s = l_acc.sub_view(l_slot_acc, l_slot);
              auto l_tmp = make_tensor<fp32>(l_slot_ml, l_slot_ml);
              tiu::fsub(l_tmp, l_mi_s, l_M);
              exp_no_overflow(l_tmp, l_tmp, &l_slot_ml, &l_slot_ml);
              // broadcast mul (w): factor per head applied across d
              tiu::fmul(l_acc_s, l_acc_s, l_tmp);
            }
          }
          // ---- fold all chunks into slot 0, normalize by 1/L, cast to T and
          // ---- store straight to the output tensor. Every core of the group
          // ---- computes the identical merged result, so the redundant store
          // ---- below writes the same values (safe).
          auto l_acc_my = l_acc.sub_view(l_slot_acc, zero_slot_);
          for (int s = 1; s < k_split; s++) {
            dim4 s_slot = {s, 0, 0, 0};
            tiu::fadd(l_acc_my, l_acc_my, l_acc.sub_view(l_slot_acc, s_slot));
          }
          // broadcast div (w): l_acc_my /= L per head
          tiu::fdiv(l_acc_my, l_acc_my, l_L, 2);
          // The cast destination must NOT alias the fp32 source: viewing l_acc
          // as T would overlap the source slot and corrupt the cast. Use a
          // dedicated bf16/f16 buffer instead.
          auto l_out_T = make_tensor<T>(l_slot_acc, l_slot_acc);
          tiu::cast(l_out_T, l_acc_my);
          dim4 qkv_offset = {_m, _h, 0, 0};
          dma::store(out_sub_global.sub_view(qi_real_global_shape, qkv_offset),
                     l_out_T);
        }
      }
    }
  }
}

// Flash-decoding split-K variants of fattention_decode (kvm-axis split to
// fill idle cores when num_core > kv_head). Only reached when the host
// decides k_split > 1; the legacy kernel handles every k_split == 1 case.
// bf16/f16 only: has_lse is always 0 here (the lse path forces k_split==1),
// ptr_lse is nullptr.
__KERNEL__ void fattention_decode_k_split_bf16(
    bf16 *ptr_out, bf16 *ptr_q, bf16 *ptr_k, bf16 *ptr_v, bf16 *ptr_mask,
    int b, int qm, int kvm, int q_head, int kv_head, float sqrt_d, int has_mask,
    fp32 *buffer_addr, const int g_core_num, const int d, const int keep_dim,
    const int block_m, const int block_k, const int block_qh,
    const int block_kh, const int k_split) {
  // buffer_addr must come before the const args: the JIT entry strips const
  // arguments from the runtime api struct, so any non-const arg after them
  // lands at a different offset on host (full header) and device (stripped).
  // k_split stays const (it selects the JIT specialization, like block_*).
  flash_attention_decode_k_split<bf16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, b,
                                       qm, kvm, q_head, kv_head, sqrt_d,
                                       has_mask, g_core_num, d, keep_dim,
                                       block_m, block_k, block_qh, block_kh,
                                       buffer_addr, k_split);
}

__KERNEL__ void fattention_decode_k_split_f16(
    fp16 *ptr_out, fp16 *ptr_q, fp16 *ptr_k, fp16 *ptr_v, fp16 *ptr_mask,
    int b, int qm, int kvm, int q_head, int kv_head, float sqrt_d, int has_mask,
    fp32 *buffer_addr, const int g_core_num, const int d, const int keep_dim,
    const int block_m, const int block_k, const int block_qh,
    const int block_kh, const int k_split) {
  // See the bf16 wrapper above for why buffer_addr precedes the const args.
  flash_attention_decode_k_split<fp16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, b,
                                       qm, kvm, q_head, kv_head, sqrt_d,
                                       has_mask, g_core_num, d, keep_dim,
                                       block_m, block_k, block_qh, block_kh,
                                       buffer_addr, k_split);
}
