
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

template <typename DataType>
void l2norm(tensor<DataType> &out, tensor<DataType> &in, const int Tile_C,
            const int Tile_W, int C, int W) {
  dim4 block_shape = {1, Tile_C, 1, Tile_W};
  dim4 shape = {1, C, 1, W};
  dim4 reduce_block_shape = {1, Tile_C, 1, 1};
  dim4 reduce_shape = {1, C, 1, 1};
  auto rsqrt_buffer_fp32 = make_tensor<fp32>(reduce_block_shape, reduce_shape);
  auto reduce_power_lt =
      make_tensor<DataType>(reduce_block_shape, reduce_shape);
  auto reduce_power_add_lt =
      make_tensor<DataType>(reduce_block_shape, reduce_shape);
  auto frsqrt_lt = make_tensor<DataType>(reduce_block_shape, reduce_shape);
  auto buffer0 = make_tensor<DataType>(block_shape, shape);
  tiu::fmul(buffer0, in, in);
  quick_pooling(reduce_power_lt, buffer0, &block_shape, &shape, 0, 1);
  tiu::fadd(reduce_power_add_lt, reduce_power_lt, 1e-6);
  tiu::cast(rsqrt_buffer_fp32, reduce_power_add_lt);
  tiu::frsqrt(rsqrt_buffer_fp32, rsqrt_buffer_fp32, 3);
  tiu::cast(reduce_power_lt, rsqrt_buffer_fp32);
  tiu::fmul(out, in, reduce_power_lt);
}

/*
mask @ in = out
[1,0,0]   [x1]   [x1]
[1,1,0] @ [x2] = [x1+x2]
[1,1,1]   [x3]   [x1+x2+x3]
*/
template <typename DataType>
void cumsum(tensor<DataType> &out, tensor<DataType> &in,
            tensor<DataType> &mask) {
  tiu::fmm2(out, mask, in);
}

/*
in - {1,chunk_size,vhead_per_khead,1}
mask_tril - {1,chunk_size,1,chunk_size}
out - {1,chunk_size,vhead_per_khead,chunk_size}
[Note]
Currently vhead_per_khead=1 in practice, so in and out shapes are both
{1,chunk_size,1,chunk_size}
*/
template <typename DataType>
void decay(tensor<DataType> &out, tensor<DataType> &in,
           tensor<DataType> &mask_tril, const int chunk_size,
           const int vhead_per_khead) {
  dim4 trans_in_shape = {1, 1, vhead_per_khead, chunk_size};
  auto in_align = in.view(TPU_ALIGN);
  auto trans_in =
      make_tensor<DataType>(trans_in_shape, trans_in_shape, TPU_ROW_ALIGN);
  tiu::transpose_cw(trans_in, in_align);

  float zero = 0.f;
  dim4 out_shape = {1, chunk_size, vhead_per_khead, chunk_size};
  auto decay_mask = make_tensor<DataType>(out_shape, out_shape);
  // tiu::fsub(decay_mask, in, trans_in_broadcast);
  tiu::fsub(decay_mask, in, trans_in.view(TPU_ALIGN));
  tiu::gt_select(out, mask_tril, zero, decay_mask, zero);
  exp_no_overflow(decay_mask, out, &out_shape, &out_shape);
  tiu::gt_select(out, mask_tril, zero, decay_mask, zero);
}

/*
out = -((k_beta @ k) * decay_mask).masked_fill(chunk_attn_mask, 0)
k_beta - {1, chunk_size, 1, d}
k - {1, chunk_size, 1, d}
decay_mask - {1, chunk_size, 1, chunk_size}
chunk_attn_mask - {1, chunk_size, 1, chunk_size} upper triangular matrix with 1s
*/
template <typename DataType>
void attn_kbeta_k(tensor<DataType> &out, tensor<DataType> &k_beta,
                  tensor<DataType> &k, tensor<DataType> &decay_mask,
                  tensor<DataType> &chunk_attn_mask, const int chunk_size) {
  dim4 out_shape = {1, chunk_size, 1, chunk_size};
  auto buffer = make_tensor<DataType>(out_shape, out_shape);
  float zero = 0.f;
  tiu::fmm2_nt(buffer, k_beta, k);
  tiu::fmul(out, buffer, decay_mask);
  tiu::gt_select(buffer, chunk_attn_mask, zero, zero, out);
  tiu::fmul(out, buffer, -1);
}

template <typename DataType>
void attn_q_k(tensor<DataType> &out, tensor<DataType> &q, tensor<DataType> &k,
              tensor<DataType> &decay_mask, tensor<DataType> &chunk_attn_mask,
              const int chunk_size, const int d) {
  dim4 out_shape = {1, chunk_size, 1, chunk_size};
  auto buffer = make_tensor<DataType>(out_shape, out_shape);
  float zero = 0.f;
  tiu::fmm2_nt(out, q, k);
  tiu::fmul(buffer, out, decay_mask);
  tiu::gt_select(out, chunk_attn_mask, zero, zero, buffer);
}

template <typename T>
void chunk_gated_delta_rule_kernel(
    T *ptr_core_attn_out, T *ptr_last_recurrent_state, T *ptr_Q, T *ptr_K,
    T *ptr_V, T *ptr_g, T *ptr_beta,
    T *ptr_triu_mask,        // upper triangular matrix with 1s
    T *ptr_strict_triu_mask, // strict upper triangular matrix with 1s
    T *ptr_tril_mask,        // lower triangular matrix with 1s
    T *ptr_eye,              // diagonal matrix with 1s
    int B, int S, const int num_k_heads, const int num_v_heads, const int d,
    float scale, const int chunk_size, const int use_qk_l2norm,
    const int core_num,
    const int TileS // set to chunk_size
) {
  int core_index = get_core_index();
  if (B <= 0 || S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || d <= 0 ||
      core_index >= core_num) {
    return;
  }
  const int P_H = core_num;
  int K_heads_per_core = num_k_heads / P_H;
  int Hstart = core_index * K_heads_per_core;
  int Hend = min(Hstart + K_heads_per_core, num_k_heads);
  int Hslice = Hend - Hstart;

  const int vhead_per_khead = num_v_heads / num_k_heads;
  const int num_chunks = (S + chunk_size - 1) / chunk_size;

  // Global tensor shapes
  dim4 shape_qk_g = {B, S, num_k_heads, d};
  dim4 shape_v_g = {B, S, num_v_heads, d};
  dim4 shape_gb_g = {B, S, 1, num_v_heads};
  dim4 shape_mask_g = {1, chunk_size, 1, chunk_size};
  dim4 shape_state_g = {B, num_v_heads, d, d};
  dim4 shape_out_g = {B, S, num_v_heads, d};

  auto g_Q = gtensor<T>(shape_qk_g, GLOBAL, ptr_Q);
  auto g_K = gtensor<T>(shape_qk_g, GLOBAL, ptr_K);
  auto g_V = gtensor<T>(shape_v_g, GLOBAL, ptr_V);
  auto g_g = gtensor<T>(shape_gb_g, GLOBAL, ptr_g);
  auto g_beta = gtensor<T>(shape_gb_g, GLOBAL, ptr_beta);
  auto g_triu_mask = gtensor<T>(shape_mask_g, GLOBAL, ptr_triu_mask);
  auto g_strict_triu_mask =
      gtensor<T>(shape_mask_g, GLOBAL, ptr_strict_triu_mask);
  auto g_eye = gtensor<T>(shape_mask_g, GLOBAL, ptr_eye);
  auto g_tril_mask = gtensor<T>(shape_mask_g, GLOBAL, ptr_tril_mask);
  auto g_out = gtensor<T>(shape_out_g, GLOBAL, ptr_core_attn_out);
  auto g_state = gtensor<T>(shape_state_g, GLOBAL, ptr_last_recurrent_state);

  // Load masks
  auto triu_mask = tensor<T>(shape_mask_g);
  auto strict_triu_mask = tensor<T>(shape_mask_g);
  auto eye = tensor<T>(shape_mask_g);
  auto tril_mask = tensor<T>(shape_mask_g);
  dma::load(triu_mask, g_triu_mask);
  dma::load(strict_triu_mask, g_strict_triu_mask);
  dma::load(eye, g_eye);
  dma::load(tril_mask, g_tril_mask);

  // Declare block shapes
  dim4 q_block_shape = {1, TileS, 1, d};
  dim4 k_block_shape = {1, TileS, 1, d};
  dim4 k_trans_block_shape = {1, d, 1, TileS};
  dim4 v_block_shape = {1, TileS, vhead_per_khead, d};
  dim4 v_sub_block_shape = {1, TileS, 1, d};
  dim4 g_block_shape = {1, TileS, 1, vhead_per_khead};
  dim4 beta_block_shape = {1, TileS, 1, vhead_per_khead};
  dim4 decay_mask_shape = {1, TileS, 1, TileS};
  dim4 attn_kk_block_shape = {1, TileS, 1, TileS};
  dim4 gcum_exp_block_shape = {1, TileS, 1, 1};
  dim4 state_block_shape = {vhead_per_khead, d, 1, d};

  for (int b_idx = 0; b_idx < B; b_idx++) {
    for (int h_idx = Hstart; h_idx < Hend; h_idx++) {
      int h_v_idx = h_idx * vhead_per_khead;
      dim4 cur_state_offset = {b_idx, h_v_idx, 0, 0};
      dim4 cur_state_shape = {1, vhead_per_khead, d, d};
      auto state_lt = make_tensor<T>(state_block_shape, state_block_shape);
      dma::load(state_lt, g_state.sub_view(cur_state_shape, cur_state_offset)
                              .view(state_block_shape));
      for (int seq_idx = 0; seq_idx < S; seq_idx += TileS) {
        enable_pipeline();
        int cur_len = min(chunk_size, S - seq_idx);
        dim4 cur_Q_shape = {1, cur_len, 1, d};
        dim4 cur_K_shape = {1, cur_len, 1, d};
        dim4 cur_V_shape = {1, cur_len, vhead_per_khead, d};
        dim4 cur_g_shape = {1, cur_len, 1, vhead_per_khead};
        dim4 cur_beta_shape = {1, cur_len, 1, vhead_per_khead};
        dim4 cur_Out_shape = {1, cur_len, vhead_per_khead, d};
        dim4 cur_attn_kk_shape = {1, TileS, 1, TileS};
        auto Q_lt = make_tensor<T>(q_block_shape, q_block_shape);
        auto K_lt = make_tensor<T>(k_block_shape, k_block_shape);
        auto V_lt = make_tensor<T>(v_block_shape, v_block_shape);
        auto g_lt = make_tensor<T>(g_block_shape, g_block_shape);
        auto beta_lt = make_tensor<T>(beta_block_shape, beta_block_shape);
        auto Out_lt = make_tensor<T>(v_block_shape, v_block_shape);
        dim4 cur_Q_offset = {b_idx, seq_idx, h_idx, 0};
        dim4 cur_K_offset = {b_idx, seq_idx, h_idx, 0};
        dim4 cur_V_offset = {b_idx, seq_idx, h_v_idx, 0};
        dim4 cur_g_offset = {b_idx, seq_idx, 0, h_v_idx};
        dim4 cur_beta_offset = {b_idx, seq_idx, 0, h_v_idx};
        dim4 cur_Out_offset = {b_idx, seq_idx, h_v_idx, 0};
        dim4 cur_Q_local_offset = {0, 0, 0, 0};
        dim4 cur_K_local_offset = {0, 0, 0, 0};
        dim4 cur_V_local_offset = {0, 0, 0, 0};
        dim4 cur_Out_local_offset = {0, 0, 0, 0};
        dim4 cur_g_local_offset = {0, 0, 0, 0};
        dim4 cur_beta_local_offset = {0, 0, 0, 0};
        dma::load(Q_lt.sub_view(cur_Q_shape, cur_Q_local_offset),
                  g_Q.sub_view(cur_Q_shape, cur_Q_offset));
        dma::load(K_lt.sub_view(cur_K_shape, cur_K_local_offset),
                  g_K.sub_view(cur_K_shape, cur_K_offset));
        dma::load(V_lt.sub_view(cur_V_shape, cur_V_local_offset),
                  g_V.sub_view(cur_V_shape, cur_V_offset));
        dma::load(g_lt.sub_view(cur_g_shape, cur_g_local_offset),
                  g_g.sub_view(cur_g_shape,
                               cur_g_offset)); //{1, TileS, 1, vhead_per_khead}
        dma::load(
            beta_lt.sub_view(cur_beta_shape, cur_beta_local_offset),
            g_beta.sub_view(cur_beta_shape,
                            cur_beta_offset)); //{1, TileS, 1, vhead_per_khead}

        dim4 pad_offset = {0, cur_len, 0, 0};
        int pad_size = TileS - cur_len;
        dim4 pad_Q_shape = {1, pad_size, 1, d};
        dim4 pad_K_shape = {1, pad_size, 1, d};
        dim4 pad_V_shape = {1, pad_size, vhead_per_khead, d};
        dim4 pad_g_shape = {1, pad_size, 1, vhead_per_khead};
        dim4 pad_beta_shape = {1, pad_size, 1, vhead_per_khead};
        if (cur_len < TileS) {
          tiu::fill(Q_lt.sub_view(pad_Q_shape, pad_offset), 0);
          tiu::fill(K_lt.sub_view(pad_K_shape, pad_offset), 0);
          tiu::fill(V_lt.sub_view(pad_V_shape, pad_offset), 0);
          tiu::fill(g_lt.sub_view(pad_g_shape, pad_offset), 0);
          tiu::fill(beta_lt.sub_view(pad_beta_shape, pad_offset), 0);
        }

        auto Q_norm_lt = make_tensor<T>(q_block_shape, q_block_shape);
        auto K_norm_lt = make_tensor<T>(k_block_shape, k_block_shape);
        if (use_qk_l2norm) {
          l2norm<T>(Q_norm_lt, Q_lt, TileS, d, TileS, d);
          l2norm<T>(K_norm_lt, K_lt, TileS, d, TileS, d);
        }

        tiu::fmul(Q_lt, Q_norm_lt, scale);

        // chunk decay
        auto g_cum = make_tensor<T>(
            g_block_shape, g_block_shape); //{1, TileS, 1, vhead_per_khead}
        cumsum<T>(g_cum, g_lt, tril_mask); //{1, TileS, 1, vhead_per_khead}

        for (int v_idx = 0; v_idx < vhead_per_khead; v_idx++) {

          dim4 cur_v_sub_shape = {1, TileS, 1, d};
          dim4 cur_v_sub_offset = {0, 0, v_idx, 0};
          dim4 cur_beta_sub_shape = {1, TileS, 1, 1};
          dim4 cur_beta_sub_offset = {0, 0, 0, v_idx};
          auto cur_beta_sub =
              beta_lt.sub_view(cur_beta_sub_shape, cur_beta_sub_offset);
          auto cur_V_sub = V_lt.sub_view(cur_v_sub_shape, cur_v_sub_offset);

          auto k_beta = make_tensor<T>(k_block_shape, k_block_shape);
          auto v_beta = make_tensor<T>(v_sub_block_shape, v_sub_block_shape);
          tiu::fmul(k_beta, K_norm_lt, cur_beta_sub);
          tiu::fmul(v_beta, cur_V_sub, cur_beta_sub);

          dim4 cur_g_cum_sub_shape = {1, TileS, 1, 1};
          dim4 cur_g_cum_sub_offset = {0, 0, 0, v_idx};
          dim4 cur_decay_mask_shape = {1, TileS, 1, TileS};
          auto cur_g_cum_sub =
              make_tensor<T>(cur_g_cum_sub_shape, cur_g_cum_sub_shape);
          tiu::move(cur_g_cum_sub,
                    g_cum.sub_view(cur_g_cum_sub_shape, cur_g_cum_sub_offset));
          auto decay_mask = make_tensor<T>(decay_mask_shape, decay_mask_shape);

          decay<T>(decay_mask, cur_g_cum_sub, tril_mask, TileS, 1);

          auto attn_kk =
              make_tensor<T>(attn_kk_block_shape, attn_kk_block_shape);
          auto L0buffer =
              make_tensor<T>(attn_kk_block_shape, attn_kk_block_shape);
          auto L1buffer =
              make_tensor<T>(attn_kk_block_shape, attn_kk_block_shape);
          attn_kbeta_k<T>(attn_kk, k_beta, K_norm_lt, decay_mask, triu_mask,
                          chunk_size);

          tiu::move(L0buffer, attn_kk);
          for (int i = 0; i < 63; i++) {
            tiu::fmm2(L1buffer, L0buffer, attn_kk);
            tiu::fadd(L0buffer, L1buffer, attn_kk);
          }
          tiu::fadd(attn_kk, L0buffer, eye);

          auto value = make_tensor<T>(v_sub_block_shape, v_sub_block_shape);
          tiu::fmm2(value, attn_kk, v_beta);

          auto k_cumdecay =
              make_tensor<T>(v_sub_block_shape, v_sub_block_shape);
          auto gcum_exp =
              make_tensor<T>(gcum_exp_block_shape, gcum_exp_block_shape);
          auto k_beta_mul_gcum_exp =
              make_tensor<T>(k_block_shape, k_block_shape);
          exp_no_overflow(gcum_exp, cur_g_cum_sub, &gcum_exp_block_shape,
                          &gcum_exp_block_shape);
          tiu::fmul(k_beta_mul_gcum_exp, k_beta, gcum_exp);
          tiu::fmm2(k_cumdecay, attn_kk, k_beta_mul_gcum_exp);

          auto attn_qk =
              make_tensor<T>(attn_kk_block_shape, attn_kk_block_shape);
          attn_q_k(attn_qk, Q_lt, K_norm_lt, decay_mask, strict_triu_mask,
                   chunk_size, d);

          dim4 cur_state_sub_shape = {1, d, 1, d};
          dim4 cur_state_sub_offset = {v_idx, 0, 0, 0};
          auto cur_state_sub =
              state_lt.sub_view(cur_state_sub_shape, cur_state_sub_offset);

          auto v_prime = make_tensor<T>(v_sub_block_shape, v_sub_block_shape);
          auto v_new = make_tensor<T>(v_sub_block_shape, v_sub_block_shape);
          tiu::fmm2(v_prime, k_cumdecay, cur_state_sub);
          tiu::fsub(v_new, value, v_prime);

          auto q_mul_gcum_exp = make_tensor<T>(q_block_shape, q_block_shape);
          tiu::fmul(q_mul_gcum_exp, Q_lt, gcum_exp);
          auto attn_inter = make_tensor<T>(q_block_shape, q_block_shape);
          tiu::fmm2(attn_inter, q_mul_gcum_exp, cur_state_sub);

          auto qkv = make_tensor<T>(v_sub_block_shape, cur_v_sub_shape);
          tiu::fmm2(qkv, attn_qk, v_new);
          tiu::fadd(Out_lt.sub_view(cur_v_sub_shape, cur_v_sub_offset),
                    attn_inter, qkv);
          // update state
          dim4 g_cum_last_exp_shape = {1, 1, 1, 1};
          dim4 g_cum_last_exp_offset = {0, 63, 0, 0};
          auto g_cum_exp_last =
              make_tensor<T>(g_cum_last_exp_shape, g_cum_last_exp_shape);
          auto g_cum_last =
              make_tensor<T>(g_cum_last_exp_shape, g_cum_last_exp_shape);
          tiu::move_cross_lane(
              g_cum_exp_last,
              gcum_exp.sub_view(g_cum_last_exp_shape, g_cum_last_exp_offset));
          tiu::move_cross_lane(g_cum_last,
                               cur_g_cum_sub.sub_view(g_cum_last_exp_shape,
                                                      g_cum_last_exp_offset));

          auto decayed_previous_state =
              make_tensor<T>(cur_state_sub_shape, cur_state_sub_shape);
          tiu::fmul(decayed_previous_state, cur_state_sub, g_cum_exp_last);

          auto relative_g =
              make_tensor<T>(cur_g_cum_sub_shape, cur_g_cum_sub_shape);
          tiu::fsub(relative_g, g_cum_last, cur_g_cum_sub);
          auto gating_weights =
              make_tensor<T>(cur_g_cum_sub_shape, cur_g_cum_sub_shape);
          exp_no_overflow(gating_weights, relative_g, &cur_g_cum_sub_shape,
                          &cur_g_cum_sub_shape);

          auto weighted_keys = make_tensor<T>(k_block_shape, k_block_shape);
          tiu::fmul(weighted_keys, K_norm_lt, gating_weights);

          auto weighted_keys_trans = make_tensor<T>(
              k_trans_block_shape, k_trans_block_shape, TPU_ROW_ALIGN);
          tiu::transpose_cw(weighted_keys_trans, weighted_keys);

          auto current_input_contribution =
              make_tensor<T>(cur_state_sub_shape, cur_state_sub_shape);
          tiu::fmm2(current_input_contribution,
                    weighted_keys_trans.view(TPU_ALIGN), v_new);

          tiu::fadd(cur_state_sub, decayed_previous_state,
                    current_input_contribution);
        }
        dma::store(g_out.sub_view(cur_Out_shape, cur_Out_offset),
                   Out_lt.sub_view(cur_Out_shape, cur_Out_local_offset));
      }
      dma::store(g_state.sub_view(cur_state_shape, cur_state_offset)
                     .view(state_block_shape),
                 state_lt);
    }
  }
}

__KERNEL__ void chunk_gated_delta_rule_bf16(
    bf16 *ptr_core_attn_out, bf16 *ptr_last_recurrent_state, bf16 *ptr_Q,
    bf16 *ptr_K, bf16 *ptr_V, bf16 *ptr_g, bf16 *ptr_beta,
    bf16 *ptr_triu_mask,        // upper triangular matrix with 1s
    bf16 *ptr_strict_triu_mask, // strict upper triangular matrix with 1s
    bf16 *ptr_tril_mask,        // lower triangular matrix with 1s
    bf16 *ptr_eye,              // diagonal matrix with 1s
    int B, int S, const int num_k_heads, const int num_v_heads, const int d,
    float scale, const int chunk_size, const int use_qk_l2norm,
    const int core_num,
    const int TileS // set to chunk_size
) {
  chunk_gated_delta_rule_kernel<bf16>(
      ptr_core_attn_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
      ptr_beta, ptr_triu_mask, ptr_strict_triu_mask, ptr_tril_mask, ptr_eye, B,
      S, num_k_heads, num_v_heads, d, scale, chunk_size, use_qk_l2norm,
      core_num, TileS);
}

__KERNEL__ void chunk_gated_delta_rule_f16(
    fp16 *ptr_core_attn_out, fp16 *ptr_last_recurrent_state, fp16 *ptr_Q,
    fp16 *ptr_K, fp16 *ptr_V, fp16 *ptr_g, fp16 *ptr_beta,
    fp16 *ptr_triu_mask,        // upper triangular matrix with 1s
    fp16 *ptr_strict_triu_mask, // strict upper triangular matrix with 1s
    fp16 *ptr_tril_mask,        // lower triangular matrix with 1s
    fp16 *ptr_eye,              // diagonal matrix with 1s
    int B, int S, const int num_k_heads, const int num_v_heads, const int d,
    float scale, const int chunk_size, const int use_qk_l2norm,
    const int core_num,
    const int TileS // set to chunk_size
) {
  chunk_gated_delta_rule_kernel<fp16>(
      ptr_core_attn_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
      ptr_beta, ptr_triu_mask, ptr_strict_triu_mask, ptr_tril_mask, ptr_eye, B,
      S, num_k_heads, num_v_heads, d, scale, chunk_size, use_qk_l2norm,
      core_num, TileS);
}

__TEST__ void chunk_gated_delta_rule_test() {
  const int B = 1;   // batch
  const int S = 128; // sequence_length
  const int num_k_heads = 16;
  const int num_v_heads = 64;
  const int d = 128;
  const int chunk_size = 64;
  const int use_qk_l2norm = 1;
  const int TileS = 64;
  const int core_num = 1;
  const float scale = 0.08838834764831844;
  dim4 Q_shape = {B, S, num_k_heads, d};
  dim4 K_shape = {B, S, num_k_heads, d};
  dim4 V_shape = {B, S, num_v_heads, d};
  dim4 g_shape = {B, S, num_v_heads, 1};
  dim4 beta_shape = {B, S, num_v_heads, 1};
  dim4 triu_mask_shape = {1, chunk_size, 1, chunk_size};
  dim4 core_attn_out_shape = {B, S, num_v_heads, d};
  dim4 last_recurrent_state_shape = {B, num_v_heads, d, d};
  auto ptr_core_attn_out = malloc<bf16>(&core_attn_out_shape);
  auto ptr_last_recurrent_state = malloc<bf16>(&last_recurrent_state_shape);
  auto ptr_Q = malloc<bf16>(&Q_shape);
  auto ptr_K = malloc<bf16>(&K_shape);
  auto ptr_V = malloc<bf16>(&V_shape);
  auto ptr_g = malloc<bf16>(&g_shape);
  auto ptr_beta = malloc<bf16>(&beta_shape);
  auto ptr_triu_mask = malloc<bf16>(&triu_mask_shape);
  auto ptr_strict_triu_mask = malloc<bf16>(&triu_mask_shape);
  auto ptr_eye_mask = malloc<bf16>(&triu_mask_shape);
  auto ptr_tril_mask = malloc<bf16>(&triu_mask_shape);

  ppl::read_npz(ptr_Q, "chunk_gated_delta_rule_input.npz", "in0");
  ppl::read_npz(ptr_K, "chunk_gated_delta_rule_input.npz", "in1");
  ppl::read_npz(ptr_V, "chunk_gated_delta_rule_input.npz", "in2");
  ppl::read_npz(ptr_g, "chunk_gated_delta_rule_input.npz", "in3");
  ppl::read_npz(ptr_beta, "chunk_gated_delta_rule_input.npz", "in4");
  ppl::read_npz(ptr_last_recurrent_state, "chunk_gated_delta_rule_input.npz",
                "in5");

  ppl::read_npz(ptr_triu_mask,
                "chunk_gated_delta_rule_top_f32_all_origin_weight.npz",
                "weight0");
  ppl::read_npz(ptr_strict_triu_mask,
                "chunk_gated_delta_rule_top_f32_all_origin_weight.npz",
                "weight1");
  ppl::read_npz(ptr_tril_mask,
                "chunk_gated_delta_rule_top_f32_all_origin_weight.npz",
                "weight2");
  ppl::read_npz(ptr_eye_mask,
                "chunk_gated_delta_rule_top_f32_all_origin_weight.npz",
                "weight3");

  chunk_gated_delta_rule_bf16(
      ptr_core_attn_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
      ptr_beta, ptr_triu_mask, ptr_strict_triu_mask, ptr_tril_mask,
      ptr_eye_mask, B, S, num_k_heads, num_v_heads, d, scale, chunk_size,
      use_qk_l2norm, core_num, TileS);
}
