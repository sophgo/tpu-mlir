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

#if defined(__bm1684x2__)

// w8a8_block_matmul, BM1684X2; router: M==1 -> decode_*, M>1 -> prefill_*.
// Split strategy: split_strategy.md. K-split merge must stay in L2 and needs
// the ppl-compile "non-LMEM src -> S2S" patch (build prerequisite).

// Balanced shard split; K cuts whole quant blocks. Shapes are core-index-free.
int bal_first(int idx, int blocks, int parts) {
  const int lo = blocks / parts;
  return idx * lo + min(idx, blocks - lo * parts);
}

int bal_size(int idx, int blocks, int parts) {
  const int lo = blocks / parts;
  return lo + (idx < blocks - lo * parts ? 1 : 0);
}

// Per-k-block amax requant to fp8e4m3, reference semantics (ACT_TYPE math).
template <typename ACT_TYPE>
void act_requant(tensor<fp8e4m3> &dst, tensor<fp32> &scale,
                 tensor<ACT_TYPE> &src, const int block_c, const int block_w,
                 const int real_c, const int real_w, const int group) {
  dim4 q_block_shape = {1, block_c, block_w / group, group};
  dim4 q_real_shape = {1, real_c, real_w / group, group};
  dim4 s_block_shape = {1, block_c, block_w / group, 1};
  dim4 s_real_shape = {1, real_c, real_w / group, 1};

  auto amax = make_tensor<ACT_TYPE>(s_block_shape, s_real_shape);
  auto scale_act = make_tensor<ACT_TYPE>(s_block_shape, s_real_shape);
  auto buf = make_tensor<ACT_TYPE>(q_block_shape, q_real_shape, TPU_ROW_ALIGN);
  auto recip = make_tensor<ACT_TYPE>(s_block_shape, s_real_shape);

  tiu::move(buf, src.view(q_real_shape));
  tiu::reduce(amax, buf, ALL_REDUCE_AMAX);
  tiu::fmax(amax, amax, 1e-4f);
  tiu::fmul(scale_act, amax, 1.0f / 448.0f);
  tiu::fdiv(recip, 1.0f, scale_act, 4);
  tiu::fmul(buf, src.view(q_real_shape), recip);
  tiu::cast(dst, buf);
  tiu::cast(scale, scale_act);
}

// Fold slab1.. into slab0 per row band (L2->L2 reduce), then move out.
template <typename ACT_TYPE>
void drain_bands(gtensor<ACT_TYPE> &dst_g, gtensor<ACT_TYPE> &src_l2,
                 const int num_rows, const int N, const int P_K,
                 const int core_idx, const int core_num) {
  const int band = div_up(num_rows, core_num);
  const int m_tile_lo = min(core_idx * band, num_rows);
  const int r1 = min(m_tile_lo + band, num_rows);
  if (m_tile_lo < r1) {
    dim4 band_shape = {1, r1 - m_tile_lo, 1, N};
    dim4 band_out_off = {0, m_tile_lo, 0, 0};
    auto my_slab = src_l2.sub_view(band_shape, band_out_off);
    for (int k = 1; k < P_K; k += 1) {
      dim4 slab_offset = {0, k * num_rows + m_tile_lo, 0, 0};
      dma::reduce(my_slab, src_l2.sub_view(band_shape, slab_offset),
                  ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    }
    dma::move(dst_g.sub_view(band_shape, band_out_off), my_slab);
  }
}

template <typename ACT_TYPE>
void w8a8_block_matmul_alpha_kernel(
    ACT_TYPE *ptr_output, ACT_TYPE *ptr_input, fp8e4m3 *ptr_weight,
    ACT_TYPE *ptr_weight_scale, const int G, const int M, const int K,
    const int N, const int core_num, const int P_M, const int P_N,
    const int P_K, const int TILE_M, const int TILE_N, const int weight_k_span,
    const int act_k_span, const int block_size_k, const int block_size_n) {
  // Without this only core 0 runs.
  set_block_num_max();
  const int core_idx = get_core_index();
  if (core_idx >= core_num) {
    return;
  }

  const bool need_reduce = (P_K > 1);

  // core_idx = (idx_m * P_N + idx_n) * P_K + idx_k
  const int idx_k = core_idx % P_K;
  const int rest = core_idx / P_K;
  const int idx_n = rest % P_N;
  const int idx_m = rest / P_N;

  const int num_rows = G * M;
  const int rows_lo = bal_first(idx_m, num_rows, P_M);
  const int rows_hi = rows_lo + bal_size(idx_m, num_rows, P_M);

  const int n_lo = bal_first(idx_n, N, P_N);
  const int n_hi = n_lo + bal_size(idx_n, N, P_N);
  const int k_lo = bal_first(idx_k, K / block_size_k, P_K) * block_size_k;
  const int k_hi = k_lo + bal_size(idx_k, K / block_size_k, P_K) * block_size_k;
  const int k_shard_len = k_hi - k_lo;

  dim4 input_global_shape = {1, num_rows, 1, K};
  dim4 output_global_shape = {1, num_rows, 1, N};
  dim4 w_global_shape = {1, N, 1, K};
  dim4 w_scale_global_shape = {1, div_up(N, block_size_n), 1,
                               div_up(K, block_size_k)};

  auto input_gtensor =
      gtensor<ACT_TYPE>(input_global_shape, GLOBAL, ptr_input);
  auto output_gtensor =
      gtensor<ACT_TYPE>(output_global_shape, GLOBAL, ptr_output);
  auto w_gtensor = gtensor<fp8e4m3>(w_global_shape, GLOBAL, ptr_weight);
  auto w_scale_gtensor =
      gtensor<ACT_TYPE>(w_scale_global_shape, GLOBAL, ptr_weight_scale);

// Slab capacity P_K*num_rows*N*2B <= 1 MiB is host-enforced.
  const int l2_stage_rows = 16;
  const int l2_stage_n = 8192;
  dim4 l2_block_shape = {1, need_reduce ? P_K * l2_stage_rows : l2_stage_rows,
                         1, l2_stage_n};
  dim4 l2_real_shape = {1, need_reduce ? P_K * num_rows : 1, 1,
                        need_reduce ? N : 1};
  auto slab_l2 = make_l2tensor<ACT_TYPE>(l2_block_shape, L2, l2_real_shape);

  // No early return: the drain sync is an all-core rendezvous.

  const int q_blk_k = block_size_k;
  const int q_blk_n = block_size_n;
  const int num_k_blocks = div_up(k_shard_len, q_blk_k);
  const int k_shard_len_max = div_up(K / q_blk_k, P_K) * q_blk_k;
  const int num_k_blocks_max = div_up(k_shard_len_max, q_blk_k);

  dim4 act_in_block_shape = {1, TILE_M, 1, act_k_span};
  dim4 act_cache_block_shape = {1, TILE_M, num_k_blocks_max, q_blk_k};
  dim4 act_scale_block_shape = {1, TILE_M, num_k_blocks_max, 1};
  dim4 w_scale_panel_block_shape = {1, 1, 1, weight_k_span / q_blk_k};
  dim4 w_chunk_block_shape = {1, TILE_N, 1, weight_k_span};
  // fmm2_nt wants TPU_ALIGN operands; sub_view gives FREE, so carve blocks.
  dim4 act_blk_block_shape = {1, TILE_M, 1, q_blk_k};
  dim4 w_blk_block_shape = {1, TILE_N, 1, q_blk_k};
  dim4 out_block_shape = {1, TILE_M, 1, TILE_N};
  dim4 w_scale_bc_block_shape = {1, LANE_NUM, 1, 1};
  dim4 scalar_shape = {1, 1, 1, 1};

  for (int m_tile_lo = rows_lo; m_tile_lo < rows_hi; m_tile_lo += TILE_M) {
    const int m_tile_rows = min(TILE_M, rows_hi - m_tile_lo);

    dim4 act_cache_real_shape = {1, m_tile_rows, num_k_blocks, q_blk_k};
    dim4 act_scale_real_shape = {1, m_tile_rows, num_k_blocks, 1};
    auto act_cache =
        make_tensor<fp8e4m3>(act_cache_block_shape, act_cache_real_shape);
    auto act_scale =
        make_tensor<fp32>(act_scale_block_shape, act_scale_real_shape);

    for (int kc = 0; kc < k_shard_len; kc += act_k_span) {
      enable_pipeline();
      const int act_chunk_len = min(act_k_span, k_shard_len - kc);
      dim4 act_in_real_shape = {1, m_tile_rows, 1, act_chunk_len};
      dim4 cache_shape = {1, m_tile_rows, act_chunk_len / q_blk_k, q_blk_k};
      dim4 cache_offset = {0, 0, kc / q_blk_k, 0};
      dim4 sc_shape = {1, m_tile_rows, act_chunk_len / q_blk_k, 1};

      auto act_in_bf16 = make_tensor<ACT_TYPE>(act_in_block_shape, act_in_real_shape);
      dim4 act_in_offset = {0, m_tile_lo, 0, k_lo + kc};
      dma::load(act_in_bf16, input_gtensor.sub_view(act_in_real_shape, act_in_offset));
      act_requant<ACT_TYPE>(
          act_cache.sub_view(cache_shape, cache_offset),
          act_scale.sub_view(sc_shape, cache_offset), act_in_bf16, TILE_M, act_k_span,
          m_tile_rows, act_chunk_len, q_blk_k);
    }

    const int n_window_aligned_lo = (n_lo / q_blk_n) * q_blk_n;
    const int num_n_tiles = div_up(n_hi - n_window_aligned_lo, TILE_N);
    const int num_k_chunks = div_up(k_shard_len, weight_k_span);

    dim4 acc_all_block_shape = {1, TILE_M, 1, div_up(N, P_N) + q_blk_n};
    dim4 acc_all_real_shape = {1, m_tile_rows, 1, n_hi - n_window_aligned_lo};
    auto acc_all = make_tensor<fp32>(acc_all_block_shape, acc_all_real_shape);
    tiu::fill(acc_all, 0.0f);

// Pipeliner wants exactly one DMA + one TIU stage at the top level.
    for (int tile_id = 0; tile_id < num_n_tiles * num_k_chunks; tile_id += 1) {
      enable_pipeline();
      const int n_tile_id = tile_id / num_k_chunks;
      const int k_chunk_id = tile_id - n_tile_id * num_k_chunks;
      const int n_tile_lo = n_window_aligned_lo + n_tile_id * TILE_N;
      const int n_tile_lo_clamped = max(n_tile_lo, n_lo);
      const int n_tile_cols = min(n_tile_lo + TILE_N, n_hi) - n_tile_lo_clamped;
      const int acc_col_off = n_tile_lo_clamped - n_window_aligned_lo;
      const int k_chunk_lo = k_lo + k_chunk_id * weight_k_span;
      const int k_chunk_len = min(weight_k_span, k_hi - k_chunk_lo);
      const int cur_kb = div_up(k_chunk_len, q_blk_k);

      dim4 w_chunk_real_shape = {1, n_tile_cols, 1, k_chunk_len};
      dim4 w_chunk_offset = {0, n_tile_lo_clamped, 0, k_chunk_lo};
      auto w_chunk =
          make_tensor<fp8e4m3>(w_chunk_block_shape, w_chunk_real_shape);
      dma::load(w_chunk,
                w_gtensor.sub_view(w_chunk_real_shape, w_chunk_offset));

      dim4 w_scale_load_shape = {1, 1, 1, cur_kb};
      dim4 w_scale_load_offset = {0, n_tile_lo / q_blk_n, 0, k_chunk_lo / q_blk_k};
      dim4 w_scale_panel_real_shape = {1, 1, 1, cur_kb};
      auto w_scale_panel_bf16 = make_tensor<ACT_TYPE>(
          w_scale_panel_block_shape, w_scale_panel_real_shape);
      dma::load(
          w_scale_panel_bf16,
          w_scale_gtensor.sub_view(w_scale_load_shape, w_scale_load_offset));

      auto w_scale_panel_fp32 = make_tensor<fp32>(w_scale_panel_block_shape,
                                             w_scale_panel_real_shape);
      tiu::cast(w_scale_panel_fp32, w_scale_panel_bf16);

      for (int k_block_lo = 0; k_block_lo < k_chunk_len; k_block_lo += q_blk_k) {
        const int k_block_len = min(q_blk_k, k_chunk_len - k_block_lo);
        const int k_block_id = (k_chunk_lo + k_block_lo - k_lo) / q_blk_k;

        dim4 w_blk_real_shape = {1, n_tile_cols, 1, k_block_len};
        dim4 w_sub_shape = {1, n_tile_cols, 1, k_block_len};
        dim4 w_sub_offset = {0, 0, 0, k_block_lo};
        auto w_blk_fp8 =
            make_tensor<fp8e4m3>(w_blk_block_shape, w_blk_real_shape);
        tiu::move(w_blk_fp8, w_chunk.sub_view(w_sub_shape, w_sub_offset));

        dim4 act_blk_real_shape = {1, m_tile_rows, 1, k_block_len};
        dim4 act_blk_shape = {1, m_tile_rows, 1, q_blk_k};
        dim4 act_blk_offset = {0, 0, k_block_id, 0};
        auto act_blk_fp8 =
            make_tensor<fp8e4m3>(act_blk_block_shape, act_blk_real_shape);
        tiu::move(act_blk_fp8, act_cache.sub_view(act_blk_shape, act_blk_offset));

        dim4 sub_out_real_shape = {1, m_tile_rows, 1, n_tile_cols};
        auto mm_out_fp32 =
            make_tensor<fp32>(out_block_shape, sub_out_real_shape);
        tiu::fmm2_nt(mm_out_fp32, act_blk_fp8, w_blk_fp8);

        dim4 w_scale_scalar_offset = {0, 0, 0, k_block_lo / q_blk_k};
        auto w_scale_scalar = make_tensor<fp32>(scalar_shape, scalar_shape);
        tiu::move(
            w_scale_scalar,
            w_scale_panel_fp32.sub_view(scalar_shape, w_scale_scalar_offset));

        auto w_scale_lane_bc_raw = make_tensor<fp32>(w_scale_bc_block_shape,
                                                w_scale_bc_block_shape);
        tiu::broadcast(w_scale_lane_bc_raw, w_scale_scalar);

        dim4 bc_stride = get_stride<fp32>(w_scale_bc_block_shape, TPU_ALIGN);
        dim4 w_scale_bc_stride = {bc_stride.n, 0, bc_stride.c, bc_stride.w};
        dim4 bc_shape = {1, m_tile_rows, 1, 1};
        auto w_scale_lane_bc = w_scale_lane_bc_raw.view(bc_shape, w_scale_bc_stride);

        dim4 scale_res_block_shape = {1, TILE_M, 1, 1};
        dim4 scale_res_real_shape = {1, m_tile_rows, 1, 1};
        dim4 act_scale_sub_shape = {1, m_tile_rows, 1, 1};
        dim4 act_scale_sub_offset = {0, 0, k_block_id, 0};
        auto block_scale_col = make_tensor<fp32>(scale_res_block_shape,
                                           scale_res_real_shape);
        tiu::fmul(block_scale_col, w_scale_lane_bc,
                  act_scale.sub_view(act_scale_sub_shape,
                                     act_scale_sub_offset));

        dim4 acc_sub_shape = {1, m_tile_rows, 1, n_tile_cols};
        dim4 acc_sub_offset = {0, 0, 0, acc_col_off};
        dim4 sub_out_offset = {0, 0, 0, 0};
        tiu::fmac(acc_all.sub_view(acc_sub_shape, acc_sub_offset), block_scale_col,
                  mm_out_fp32.sub_view(acc_sub_shape, sub_out_offset));
      }
    }

    for (int n_tile_id = 0; n_tile_id < num_n_tiles; n_tile_id += 1) {
      const int n_tile_lo = n_window_aligned_lo + n_tile_id * TILE_N;
      const int n_tile_lo_clamped = max(n_tile_lo, n_lo);
      const int n_tile_cols = min(n_tile_lo + TILE_N, n_hi) - n_tile_lo_clamped;

      dim4 acc_sub_shape = {1, m_tile_rows, 1, n_tile_cols};
      dim4 acc_sub_offset = {0, 0, 0, n_tile_lo_clamped - n_window_aligned_lo};
      dim4 out_tile_real_shape = {1, m_tile_rows, 1, n_tile_cols};
      dim4 out_tile_shape = {1, m_tile_rows, 1, n_tile_cols};
      dim4 out_tile_off = {0, m_tile_lo, 0, n_tile_lo_clamped};
      auto out_tile_bf16 = make_tensor<ACT_TYPE>(out_block_shape, out_tile_real_shape);
      tiu::cast(out_tile_bf16, acc_all.sub_view(acc_sub_shape, acc_sub_offset));
      if (need_reduce) {
        dim4 slab_offset = {0, idx_k * num_rows + m_tile_lo, 0, n_tile_lo_clamped};
        dma::store(slab_l2.sub_view(out_tile_shape, slab_offset), out_tile_bf16);
      } else {
        dma::store(output_gtensor.sub_view(out_tile_shape, out_tile_off), out_tile_bf16);
      }
    }
  }

  if (need_reduce) {
    sync();
    drain_bands<ACT_TYPE>(output_gtensor, slab_l2, num_rows, N, P_K, core_idx,
                          core_num);
  }
}

//===----------------------------------------------------------------------===//
// prefill family: per-tile parity double slots + rotating collector.
//===----------------------------------------------------------------------===//

// Reduce only into the collector's own L2 slot (tile_id%P_K rotates); never
// into GLOBAL. Capacity host-enforced: 2*core_num*TILE_M*TILE_N*2B <= 1 MiB.
template <typename ACT_TYPE>
void w8a8_block_matmul_prefill_alpha_kernel(
    ACT_TYPE *ptr_output, ACT_TYPE *ptr_input, fp8e4m3 *ptr_weight,
    ACT_TYPE *ptr_weight_scale, const int G, const int M, const int K,
    const int N, const int core_num, const int P_M, const int P_N,
    const int P_K, const int TILE_M, const int TILE_N, const int weight_k_span,
    const int act_k_span, const int block_size_k, const int block_size_n) {
  set_block_num_max();
  const int core_idx = get_core_index();
  if (core_idx >= core_num) {
    return;
  }

  const bool need_reduce = (P_K > 1);

  // core_idx = (idx_m * P_N + idx_n) * P_K + idx_k
  const int idx_k = core_idx % P_K;
  const int rest = core_idx / P_K;
  const int idx_n = rest % P_N;
  const int idx_m = rest / P_N;

  const int num_rows = G * M;
  const int rows_lo = bal_first(idx_m, num_rows, P_M);
  const int rows_hi = rows_lo + bal_size(idx_m, num_rows, P_M);

  const int n_lo = bal_first(idx_n, N, P_N);
  const int n_hi = n_lo + bal_size(idx_n, N, P_N);
  const int k_lo = bal_first(idx_k, K / block_size_k, P_K) * block_size_k;
  const int k_hi = k_lo + bal_size(idx_k, K / block_size_k, P_K) * block_size_k;
  const int k_shard_len = k_hi - k_lo;

  dim4 input_global_shape = {1, num_rows, 1, K};
  dim4 output_global_shape = {1, num_rows, 1, N};
  dim4 w_global_shape = {1, N, 1, K};
  dim4 w_scale_global_shape = {1, div_up(N, block_size_n), 1,
                               div_up(K, block_size_k)};

  auto input_gtensor =
      gtensor<ACT_TYPE>(input_global_shape, GLOBAL, ptr_input);
  auto output_gtensor =
      gtensor<ACT_TYPE>(output_global_shape, GLOBAL, ptr_output);
  auto w_gtensor = gtensor<fp8e4m3>(w_global_shape, GLOBAL, ptr_weight);
  auto w_scale_gtensor =
      gtensor<ACT_TYPE>(w_scale_global_shape, GLOBAL, ptr_weight_scale);

  const int slots_rows_blk = need_reduce ? 2 * core_num * TILE_M : 16;
  dim4 slots_block_shape = {1, slots_rows_blk, 1, TILE_N};
  dim4 slots_real_shape = {1, need_reduce ? 2 * core_num * TILE_M : 1, 1,
                           need_reduce ? TILE_N : 1};
  auto slot_l2 = make_l2tensor<ACT_TYPE>(slots_block_shape, L2,
                                           slots_real_shape);

  // No early return: sync is an all-core rendezvous.

  const int q_blk_k = block_size_k;
  const int q_blk_n = block_size_n;
  const int num_k_blocks = div_up(k_shard_len, q_blk_k);
  const int k_shard_len_max = div_up(K / q_blk_k, P_K) * q_blk_k;
  const int num_k_blocks_max = div_up(k_shard_len_max, q_blk_k);

  dim4 act_in_block_shape = {1, TILE_M, 1, act_k_span};
  dim4 act_cache_block_shape = {1, TILE_M, num_k_blocks_max, q_blk_k};
  dim4 act_scale_block_shape = {1, TILE_M, num_k_blocks_max, 1};
  dim4 w_scale_panel_block_shape = {1, 1, 1, weight_k_span / q_blk_k};
  dim4 w_chunk_block_shape = {1, TILE_N, 1, weight_k_span};
  dim4 act_blk_block_shape = {1, TILE_M, 1, q_blk_k};
  dim4 w_blk_block_shape = {1, TILE_N, 1, q_blk_k};
  dim4 out_block_shape = {1, TILE_M, 1, TILE_N};
  dim4 w_scale_bc_block_shape = {1, LANE_NUM, 1, 1};
  dim4 scalar_shape = {1, 1, 1, 1};

  const int n_window_aligned_lo = (n_lo / q_blk_n) * q_blk_n;
  const int num_my_m_tiles = div_up(rows_hi - rows_lo, TILE_M);
  const int num_my_n_tiles = div_up(n_hi - n_window_aligned_lo, TILE_N);
  const int num_my_tiles = num_my_m_tiles * num_my_n_tiles;
  // All cores must sync the same number of times: parameter-only bound.
  const int num_tile_slots = div_up(div_up(num_rows, P_M), TILE_M) *
                     div_up(div_up(N, P_N) + q_blk_n, TILE_N);
  dim4 act_cache_real_shape = {1, TILE_M, num_k_blocks_max, q_blk_k};
  dim4 act_scale_real_shape = {1, TILE_M, num_k_blocks_max, 1};
  auto act_cache =
      make_tensor<fp8e4m3>(act_cache_block_shape, act_cache_real_shape);
  auto act_scale =
      make_tensor<fp32>(act_scale_block_shape, act_scale_real_shape);

  for (int tile_id = 0; tile_id < num_tile_slots; tile_id += 1) {
    const int mine_this_round = tile_id < num_my_tiles;
    if (mine_this_round) {
    const int m_tile_id = tile_id / num_my_n_tiles;
    const int n_tile_id = tile_id - m_tile_id * num_my_n_tiles;
    const int m_tile_lo = rows_lo + m_tile_id * TILE_M;
    const int m_tile_rows = min(TILE_M, rows_hi - m_tile_lo);

    if (n_tile_id == 0) {
    for (int kc = 0; kc < k_shard_len; kc += act_k_span) {
      enable_pipeline();
      const int act_chunk_len = min(act_k_span, k_shard_len - kc);
      dim4 act_in_real_shape = {1, m_tile_rows, 1, act_chunk_len};
      dim4 cache_shape = {1, m_tile_rows, act_chunk_len / q_blk_k, q_blk_k};
      dim4 cache_offset = {0, 0, kc / q_blk_k, 0};
      dim4 sc_shape = {1, m_tile_rows, act_chunk_len / q_blk_k, 1};

      auto act_in_bf16 = make_tensor<ACT_TYPE>(act_in_block_shape, act_in_real_shape);
      dim4 act_in_offset = {0, m_tile_lo, 0, k_lo + kc};
      dma::load(act_in_bf16, input_gtensor.sub_view(act_in_real_shape, act_in_offset));
      act_requant<ACT_TYPE>(
          act_cache.sub_view(cache_shape, cache_offset),
          act_scale.sub_view(sc_shape, cache_offset), act_in_bf16, TILE_M, act_k_span,
          m_tile_rows, act_chunk_len, q_blk_k);
    }
    }

    const int n_tile_lo = n_window_aligned_lo + n_tile_id * TILE_N;
    const int n_tile_lo_clamped = max(n_tile_lo, n_lo);
    const int n_tile_cols = min(n_tile_lo + TILE_N, n_hi) - n_tile_lo_clamped;

      dim4 tile_acc_real_shape = {1, m_tile_rows, 1, n_tile_cols};
      auto tile_acc_fp32 = make_tensor<fp32>(out_block_shape, tile_acc_real_shape);
      tiu::fill(tile_acc_fp32, 0.0f);

      for (int k_chunk_lo = k_lo; k_chunk_lo < k_hi; k_chunk_lo += weight_k_span) {
        enable_pipeline();
        const int k_chunk_len = min(weight_k_span, k_hi - k_chunk_lo);
        const int cur_kb = div_up(k_chunk_len, q_blk_k);

        dim4 w_chunk_real_shape = {1, n_tile_cols, 1, k_chunk_len};
        dim4 w_chunk_offset = {0, n_tile_lo_clamped, 0, k_chunk_lo};
        auto w_chunk =
            make_tensor<fp8e4m3>(w_chunk_block_shape, w_chunk_real_shape);
        dma::load(w_chunk,
                  w_gtensor.sub_view(w_chunk_real_shape, w_chunk_offset));

        dim4 w_scale_load_shape = {1, 1, 1, cur_kb};
        dim4 w_scale_load_offset = {0, n_tile_lo / q_blk_n, 0, k_chunk_lo / q_blk_k};
        dim4 w_scale_panel_real_shape = {1, 1, 1, cur_kb};
        auto w_scale_panel_bf16 = make_tensor<ACT_TYPE>(
            w_scale_panel_block_shape, w_scale_panel_real_shape);
        dma::load(
            w_scale_panel_bf16,
            w_scale_gtensor.sub_view(w_scale_load_shape, w_scale_load_offset));

        auto w_scale_panel_fp32 = make_tensor<fp32>(w_scale_panel_block_shape,
                                               w_scale_panel_real_shape);
        tiu::cast(w_scale_panel_fp32, w_scale_panel_bf16);

        for (int k_block_lo = 0; k_block_lo < k_chunk_len; k_block_lo += q_blk_k) {
          const int k_block_len = min(q_blk_k, k_chunk_len - k_block_lo);
          const int k_block_id = (k_chunk_lo + k_block_lo - k_lo) / q_blk_k;

          dim4 w_blk_real_shape = {1, n_tile_cols, 1, k_block_len};
          dim4 w_sub_shape = {1, n_tile_cols, 1, k_block_len};
          dim4 w_sub_offset = {0, 0, 0, k_block_lo};
          auto w_blk_fp8 =
              make_tensor<fp8e4m3>(w_blk_block_shape, w_blk_real_shape);
          tiu::move(w_blk_fp8, w_chunk.sub_view(w_sub_shape, w_sub_offset));

          dim4 act_blk_real_shape = {1, m_tile_rows, 1, k_block_len};
          dim4 act_blk_shape = {1, m_tile_rows, 1, q_blk_k};
          dim4 act_blk_offset = {0, 0, k_block_id, 0};
          auto act_blk_fp8 =
              make_tensor<fp8e4m3>(act_blk_block_shape, act_blk_real_shape);
          tiu::move(act_blk_fp8, act_cache.sub_view(act_blk_shape, act_blk_offset));

          dim4 sub_out_real_shape = {1, m_tile_rows, 1, n_tile_cols};
          auto mm_out_fp32 =
              make_tensor<fp32>(out_block_shape, sub_out_real_shape);
          tiu::fmm2_nt(mm_out_fp32, act_blk_fp8, w_blk_fp8);

          dim4 w_scale_scalar_offset = {0, 0, 0, k_block_lo / q_blk_k};
          auto w_scale_scalar = make_tensor<fp32>(scalar_shape, scalar_shape);
          tiu::move(
              w_scale_scalar,
              w_scale_panel_fp32.sub_view(scalar_shape, w_scale_scalar_offset));

          auto w_scale_lane_bc_raw = make_tensor<fp32>(w_scale_bc_block_shape,
                                                  w_scale_bc_block_shape);
          tiu::broadcast(w_scale_lane_bc_raw, w_scale_scalar);

          dim4 bc_stride = get_stride<fp32>(w_scale_bc_block_shape, TPU_ALIGN);
          dim4 w_scale_bc_stride = {bc_stride.n, 0, bc_stride.c, bc_stride.w};
          dim4 bc_shape = {1, m_tile_rows, 1, 1};
          auto w_scale_lane_bc = w_scale_lane_bc_raw.view(bc_shape, w_scale_bc_stride);

          dim4 scale_res_block_shape = {1, TILE_M, 1, 1};
          dim4 scale_res_real_shape = {1, m_tile_rows, 1, 1};
          dim4 act_scale_sub_shape = {1, m_tile_rows, 1, 1};
          dim4 act_scale_sub_offset = {0, 0, k_block_id, 0};
          auto block_scale_col = make_tensor<fp32>(scale_res_block_shape,
                                             scale_res_real_shape);
          tiu::fmul(block_scale_col, w_scale_lane_bc,
                    act_scale.sub_view(act_scale_sub_shape,
                                       act_scale_sub_offset));

          dim4 acc_sub_shape = {1, m_tile_rows, 1, n_tile_cols};
          dim4 acc_sub_offset = {0, 0, 0, 0};
          tiu::fmac(tile_acc_fp32.sub_view(acc_sub_shape, acc_sub_offset), block_scale_col,
                    mm_out_fp32.sub_view(acc_sub_shape, acc_sub_offset));
        }
      }

      dim4 out_tile_real_shape = {1, m_tile_rows, 1, n_tile_cols};
      dim4 out_tile_shape = {1, m_tile_rows, 1, n_tile_cols};
      dim4 out_tile_off = {0, m_tile_lo, 0, n_tile_lo_clamped};
      auto out_tile_bf16 = make_tensor<ACT_TYPE>(out_block_shape, out_tile_real_shape);
      tiu::cast(out_tile_bf16, tile_acc_fp32);
      if (need_reduce) {
        const int slot_parity = tile_id - (tile_id / 2) * 2;
        dim4 slot_off = {0, (core_idx * 2 + slot_parity) * TILE_M, 0, 0};
        dma::store(slot_l2.sub_view(out_tile_shape, slot_off), out_tile_bf16);
        sync();
        if (idx_k == tile_id - (tile_id / P_K) * P_K) {
          dim4 my_slot_off = {0, (core_idx * 2 + slot_parity) * TILE_M, 0, 0};
          auto my_slot = slot_l2.sub_view(out_tile_shape, my_slot_off);
          const int group_base_core = core_idx - idx_k;
          for (int kk = 0; kk < P_K; kk += 1) {
            if (kk != idx_k) {
              const int slot_owner_core = group_base_core + kk;
              dim4 other_slot_off = {0, (slot_owner_core * 2 + slot_parity) * TILE_M, 0, 0};
              dma::reduce(my_slot, slot_l2.sub_view(out_tile_shape, other_slot_off),
                          ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
            }
          }
          dma::move(output_gtensor.sub_view(out_tile_shape, out_tile_off),
                    my_slot);
        }
      } else {
        dma::store(output_gtensor.sub_view(out_tile_shape, out_tile_off), out_tile_bf16);
      }
    } else if (need_reduce) {
      sync();
    }
  }
}

// One entry per dtype keeps the generated fill/entry/ctrl structs unique.
__KERNEL__ void w8a8_block_matmul_decode_bf16(
    bf16 *ptr_output, bf16 *ptr_input, fp8e4m3 *ptr_weight,
    bf16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_alpha_kernel<bf16>(
      ptr_output, ptr_input, ptr_weight, ptr_weight_scale, G, M, K, N,
      core_num, P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span, act_k_span,
      block_size_k, block_size_n);
}

__KERNEL__ void w8a8_block_matmul_decode_f16(
    fp16 *ptr_output, fp16 *ptr_input, fp8e4m3 *ptr_weight,
    fp16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_alpha_kernel<fp16>(
      ptr_output, ptr_input, ptr_weight, ptr_weight_scale, G, M, K, N,
      core_num, P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span, act_k_span,
      block_size_k, block_size_n);
}

__TEST__ void w8a8_block_matmul_decode_test() {
  const int G = 1;
  const int M = 16;
  const int K = 1024;
  const int N = 1024;
  const int block_size_k = 128;
  const int block_size_n = 128;

  dim4 input_shape = {G, M, 1, K};
  dim4 output_shape = {G, M, 1, N};
  dim4 w_shape = {1, N, 1, K};
  dim4 w_scale_shape = {1, div_up(N, block_size_n), 1, div_up(K, block_size_k)};

  auto output = ppl::malloc<bf16>(&output_shape);
  auto input = ppl::rand<bf16>(&input_shape, -448, 448);
  auto w = ppl::rand<fp8e4m3>(&w_shape, -448, 448);
  auto w_scale = ppl::rand<bf16>(&w_scale_shape, 0.0022, 0.0046);

  const int core_num = 4;
  const int P_M = 1, P_N = 4, P_K = 1;
  const int TILE_M = 16, TILE_N = 128;
  const int weight_k_span = 512, act_k_span = 512;
  w8a8_block_matmul_decode_bf16(output, input, w, w_scale, G, M, K, N, core_num,
                         P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span,
                         act_k_span, block_size_k, block_size_n);
}

__KERNEL__ void w8a8_block_matmul_prefill_bf16(
    bf16 *ptr_output, bf16 *ptr_input, fp8e4m3 *ptr_weight,
    bf16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_prefill_alpha_kernel<bf16>(
      ptr_output, ptr_input, ptr_weight, ptr_weight_scale, G, M, K, N,
      core_num, P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span, act_k_span,
      block_size_k, block_size_n);
}

__KERNEL__ void w8a8_block_matmul_prefill_f16(
    fp16 *ptr_output, fp16 *ptr_input, fp8e4m3 *ptr_weight,
    fp16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_prefill_alpha_kernel<fp16>(
      ptr_output, ptr_input, ptr_weight, ptr_weight_scale, G, M, K, N,
      core_num, P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span, act_k_span,
      block_size_k, block_size_n);
}

__TEST__ void w8a8_block_matmul_prefill_test() {
  const int G = 1;
  const int M = 16;
  const int K = 1024;
  const int N = 1024;
  const int block_size_k = 128;
  const int block_size_n = 128;

  dim4 input_shape = {G, M, 1, K};
  dim4 output_shape = {G, M, 1, N};
  dim4 w_shape = {1, N, 1, K};
  dim4 w_scale_shape = {1, div_up(N, block_size_n), 1, div_up(K, block_size_k)};

  auto output = ppl::malloc<bf16>(&output_shape);
  auto input = ppl::rand<bf16>(&input_shape, -448, 448);
  auto w = ppl::rand<fp8e4m3>(&w_shape, -448, 448);
  auto w_scale = ppl::rand<bf16>(&w_scale_shape, 0.0022, 0.0046);

  const int core_num = 4;
  const int P_M = 1, P_N = 4, P_K = 1;
  const int TILE_M = 16, TILE_N = 128;
  const int weight_k_span = 512, act_k_span = 512;
  w8a8_block_matmul_prefill_bf16(output, input, w, w_scale, G, M, K, N, core_num,
                         P_M, P_N, P_K, TILE_M, TILE_N, weight_k_span,
                         act_k_span, block_size_k, block_size_n);
}

#else
__KERNEL__ void w8a8_block_matmul_decode_bf16(
    bf16 *ptr_output, bf16 *ptr_input, fp8e4m3 *ptr_weight,
    bf16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  return;
}

__KERNEL__ void w8a8_block_matmul_decode_f16(
    fp16 *ptr_output, fp16 *ptr_input, fp8e4m3 *ptr_weight,
    fp16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  return;
}

__KERNEL__ void w8a8_block_matmul_prefill_bf16(
    bf16 *ptr_output, bf16 *ptr_input, fp8e4m3 *ptr_weight,
    bf16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  return;
}

__KERNEL__ void w8a8_block_matmul_prefill_f16(
    fp16 *ptr_output, fp16 *ptr_input, fp8e4m3 *ptr_weight,
    fp16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_M, const int P_N, const int P_K, const int TILE_M,
    const int TILE_N, const int weight_k_span, const int act_k_span,
    const int block_size_k, const int block_size_n) {
  return;
}
#endif
