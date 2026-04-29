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

// Compute L2 normalization along the W dimension.
// in/out: real shape [1, C, 1, W], block shape [1, Tile_C, 1, Tile_W]
// Formula: out = in / sqrt(sum(in^2) + 1e-6)
template <typename DataType>
void l2norm(tensor<DataType> &out, tensor<DataType> &in, const int Tile_C,
            const int Tile_W, int C, int W) {
  dim4 block_shape = {Tile_C, 1, 1, Tile_W};   // local block allocation shape
  dim4 shape = {C, 1, 1, W};                   // real data shape
  dim4 reduce_block_shape = {Tile_C, 1, 1, 1}; // per-row reduce block shape
  dim4 reduce_shape = {C, 1, 1, 1};            // per-row reduce real shape
  // rsqrt_buffer_fp32: [C, 1, 1, 1] (fp32) — accumulates rsqrt result
  auto rsqrt_buffer_fp32 = make_tensor<fp32>(reduce_block_shape, reduce_shape);
  // reduce_power_lt: [C, 1, 1, 1] — sum of squares per row
  auto reduce_power_lt =
      make_tensor<DataType>(reduce_block_shape, reduce_shape);
  // reduce_power_add_lt: [C, 1, 1, 1] — sum of squares + epsilon
  auto reduce_power_add_lt =
      make_tensor<DataType>(reduce_block_shape, reduce_shape);
  // buffer0: [C, 1, 1, W] — element-wise in^2
  auto buffer0 = make_tensor<DataType>(block_shape, shape);
  tiu::fmul(buffer0, in, in); // buffer0 = in * in, shape: [C, 1, 1, W]
  // reduce_power_lt = sum(buffer0, axis=W), shape: [C, 1, 1, 1]
  quick_pooling(reduce_power_lt, buffer0, &block_shape, &shape, 0, 1);
  // reduce_power_add_lt = sum_sq + 1e-6, shape: [C, 1, 1, 1]
  tiu::fadd(reduce_power_add_lt, reduce_power_lt, 1e-6);
  // cast to fp32 for rsqrt precision, shape: [C, 1, 1, 1]
  tiu::cast(rsqrt_buffer_fp32, reduce_power_add_lt);
  // rsqrt_buffer_fp32 = rsqrt(sum_sq + 1e-6), shape: [C, 1, 1, 1]
  tiu::frsqrt(rsqrt_buffer_fp32, rsqrt_buffer_fp32, 3);
  // cast back to DataType, shape: [C, 1, 1, 1]
  tiu::cast(reduce_power_lt, rsqrt_buffer_fp32);
  // out = in * rsqrt(sum_sq + 1e-6), shape: [C, 1, 1, W]
  tiu::fmul(out, in, reduce_power_lt);
}

/*
Recurrent Gated Delta Rule kernel.

This implements the single-step recurrent version of the gated delta rule:
  1. Optional L2 norm on Q and K
  2. query = query * scale
  3. g_exp = exp(g)
  4. state = state * g_exp               (decay)
  5. kv_mem = state^T @ key              (retrieve from memory)
  6. delta = (value - kv_mem) * beta      (compute delta)
  7. state = state + key (outer) delta    (update memory)
  8. output = state^T @ query             (read from memory)

Inputs:
  Q:     [B, 1, num_k_heads, d]
  K:     [B, 1, num_k_heads, d]
  V:     [B, 1, num_v_heads, d]
  g:     [B, 1, 1, num_v_heads]
  beta:  [B, 1, 1, num_v_heads]
  state: [B, num_v_heads, d, d]

Outputs:
  core_attn_out:        [B, 1, num_v_heads, d]
  last_recurrent_state: [B, num_v_heads, d, d]  (updated in-place)
*/
template <typename T>
void recurrent_gated_delta_rule_kernel(
    T *ptr_core_attn_out, T *ptr_last_recurrent_state, T *ptr_Q, T *ptr_K,
    T *ptr_V, T *ptr_g, T *ptr_beta, int B, float scale, int core_num,
    const int num_k_heads, const int num_v_heads, const int d,
    const int use_qk_l2norm, const int block_h) {
  int core_index = get_core_index();
  if (core_index >= core_num) {
    return;
  }

  // Divide k-heads evenly across cores; each core owns [Hstart, Hend) k-heads
  int K_heads_per_core = num_k_heads / core_num;
  int Hstart = core_index * K_heads_per_core;
  int Hend = min(Hstart + K_heads_per_core, num_k_heads);

  // Number of v-heads that share one k-head (GQA grouping ratio)
  const int vhead_per_khead = num_v_heads / num_k_heads;

  // Global (DDR) tensor shapes — S=1 for the single-step recurrent mode
  dim4 shape_qk_g = {B, 1, num_k_heads, d};
  dim4 shape_v_g = {B, 1, num_v_heads, d};
  dim4 shape_gb_g = {B, 1, 1, num_v_heads};
  dim4 shape_state_g = {B, num_v_heads, d, d};
  dim4 shape_out_g = {B, 1, num_v_heads, d};

  auto g_Q = gtensor<T>(shape_qk_g, GLOBAL, ptr_Q);
  auto g_K = gtensor<T>(shape_qk_g, GLOBAL, ptr_K);
  auto g_V = gtensor<T>(shape_v_g, GLOBAL, ptr_V);
  auto g_g = gtensor<T>(shape_gb_g, GLOBAL, ptr_g);
  auto g_beta = gtensor<T>(shape_gb_g, GLOBAL, ptr_beta);
  auto g_out = gtensor<T>(shape_out_g, GLOBAL, ptr_core_attn_out);
  auto g_state = gtensor<T>(shape_state_g, GLOBAL, ptr_last_recurrent_state);

  // Local (SRAM) block allocation shapes — upper bounds for make_tensor
  // real shapes are narrowed inside the loop when the tail block is smaller
  dim4 qk_block_shape = {block_h, 1, 1, d}; // Q/K block: [block_h, 1, 1, d]
  dim4 v_block_shape = {block_h * vhead_per_khead, 1, 1,
                        d}; // V block:   [block_h*vhpk, 1, 1, d]
  dim4 state_block_shape = {block_h * vhead_per_khead, d, 1,
                            d};          // state blk: [block_h*vhpk, d, 1, d]
  dim4 k_col_shape = {block_h, d, 1, 1}; // K^T block: [block_h, d, 1, 1]
  dim4 scalar_shape = {block_h * vhead_per_khead, 1, 1,
                       1}; // g/beta blk:[block_h*vhpk, 1, 1, 1]

  for (int b_idx = 0; b_idx < B; b_idx++) {
    for (int h_idx = Hstart; h_idx < Hend; h_idx += block_h) {
      ppl::enable_pipeline();
      // Starting v-head index for this k-head group
      int h_v_idx = h_idx * vhead_per_khead;
      // Actual number of k-heads in this tile (may be < block_h at the tail)
      int real_block_h = min(block_h, Hend - h_idx);
      // Corresponding number of v-heads in this tile
      int real_v_block_h = real_block_h * vhead_per_khead;

      // Real (actual data) shapes for this tile iteration
      dim4 qk_real_shape = {real_block_h, 1, 1, d};
      dim4 v_real_shape = {real_v_block_h, 1, 1, d};
      // Global sub-view shapes used for DDR addressing (N/H/W layout)
      dim4 qk_global_shape = {1, 1, real_block_h, d};
      dim4 v_global_shape = {1, 1, real_v_block_h, d};
      dim4 gb_global_shape = {1, 1, 1, real_v_block_h};

      // --- Step 1: Load Q and K for the current k-head tile ---
      // Q_lt, K_lt: local real shape [real_block_h, 1, 1, d]
      auto Q_lt = make_tensor<T>(qk_block_shape, qk_real_shape);
      auto K_lt = make_tensor<T>(qk_block_shape, qk_real_shape);

      dim4 qk_offset = {b_idx, 0, h_idx, 0};
      dma::load(Q_lt,
                g_Q.sub_view(qk_global_shape, qk_offset).view(qk_real_shape));
      dma::load(K_lt,
                g_K.sub_view(qk_global_shape, qk_offset).view(qk_real_shape));

      // --- Step 2: Load V for the current v-head tile ---
      // V_lt: local real shape [real_v_block_h, 1, 1, d]
      auto V_lt = make_tensor<T>(v_block_shape, v_real_shape);
      dim4 v_offset = {b_idx, 0, h_v_idx, 0};
      dma::load(V_lt,
                g_V.sub_view(v_global_shape, v_offset).view(v_real_shape));

      // --- Step 3: Load g and compute g_exp = exp(g) ---
      // g_scalar / g_exp / beta_scalar: real shape [real_v_block_h, 1, 1, 1]
      dim4 scalar_real_shape = {real_v_block_h, 1, 1, 1};
      dim4 scalar_offset = {b_idx, 0, 0, h_v_idx};
      auto g_scalar = make_tensor<T>(scalar_shape, scalar_real_shape);
      dma::load(
          g_scalar,
          g_g.sub_view(gb_global_shape, scalar_offset).view(scalar_real_shape));

      auto g_exp = make_tensor<T>(scalar_shape, scalar_real_shape);
      exp_no_overflow(g_exp, g_scalar, &scalar_shape, &scalar_real_shape);

      // --- Step 4: Load beta ---
      // beta_scalar: real shape [real_v_block_h, 1, 1, 1]
      auto beta_scalar = make_tensor<T>(scalar_shape, scalar_real_shape);
      dma::load(beta_scalar, g_beta.sub_view(gb_global_shape, scalar_offset)
                                 .view(scalar_real_shape));

      // --- Step 5: Optional L2 normalization on Q and K ---
      // Q_norm_lt, K_norm_lt: real shape [real_block_h, 1, 1, d]
      auto Q_norm_lt = make_tensor<T>(qk_block_shape, qk_real_shape);
      auto K_norm_lt = make_tensor<T>(qk_block_shape, qk_real_shape);
      if (use_qk_l2norm) {
        l2norm<T>(Q_norm_lt, Q_lt, block_h, d, real_block_h, d);
        l2norm<T>(K_norm_lt, K_lt, block_h, d, real_block_h, d);
      } else {
        tiu::move(Q_norm_lt, Q_lt);
        tiu::move(K_norm_lt, K_lt);
      }

      // --- Step 6: Scale Q ---
      // Q_lt = Q_norm * scale, real shape [real_block_h, 1, 1, d]
      tiu::fmul(Q_lt, Q_norm_lt, scale);

      // --- Step 7: Transpose K for outer product ---
      // K_norm_lt: [real_block_h, 1, 1, d] -> K_col: [real_block_h, d, 1, 1]
      // K_col is laid out row-aligned so each row (d elements) is one key
      // vector
      dim4 k_col_real_shape = {real_block_h, d, 1, 1};
      auto K_col = make_tensor<T>(k_col_shape, k_col_real_shape, TPU_ROW_ALIGN);
      tiu::transpose_cw(K_col, K_norm_lt);

      // --- Step 8: Load state and apply decay ---
      // state_lt / state_result: real shape [real_v_block_h, d, 1, d]
      dim4 state_offset = {b_idx, h_v_idx, 0, 0};
      dim4 cur_state_global_shape = {1, real_v_block_h, d, d};
      dim4 state_real_shape = {real_v_block_h, d, 1, d};
      auto state_lt = make_tensor<T>(
          state_block_shape, state_real_shape); // [real_v_block_h, d, 1, d]
      auto state_result = make_tensor<T>(
          state_block_shape, state_real_shape); // [real_v_block_h, d, 1, d]
      dma::load(state_lt, g_state.sub_view(cur_state_global_shape, state_offset)
                              .view(state_real_shape));
      // state_result = state * g_exp  (per-v-head decay), shape:
      // [real_v_block_h, d, 1, d]
      tiu::fmul(state_result, state_lt, g_exp);

      // Fixed sub_view shapes used inside the per-head loops (declared once)
      dim4 sub_vec_shape = {1, 1, 1, d};
      dim4 sub_state_shape = {1, d, 1, d};
      dim4 sub_kcol_shape = {1, d, 1, 1};

      // --- Step 9: Retrieve from memory: kv_mem = key @ state_result ---
      // For each k-head i and its associated v-head j:
      //   key_sub  : [1, 1, 1, d]  (i-th row of K_norm_lt)
      //   state_sub: [1, d, 1, d]  (j-th state matrix in state_result)
      //   kv_sub   : [1, 1, 1, d]  (j-th row of kv_mem)
      //   kv_sub = key_sub @ state_sub  -> shape [1, 1, 1, d]
      // kv_mem overall real shape: [real_v_block_h, 1, 1, d]
      auto kv_mem = make_tensor<T>(v_block_shape, v_real_shape);
      for (int i = 0; i < real_block_h; i++) {
        for (int j = 0; j < vhead_per_khead; j++) {
          dim4 i_offset = {i, 0, 0, 0};
          auto key_sub =
              K_norm_lt.sub_view(sub_vec_shape, i_offset); // [1, 1, 1, d]
          dim4 ij_offset = {i * vhead_per_khead + j, 0, 0, 0};
          auto state_sub =
              state_result.sub_view(sub_state_shape, ij_offset); // [1, d, 1, d]
          auto kv_sub =
              kv_mem.sub_view(sub_vec_shape, ij_offset); // [1, 1, 1, d]
          tiu::fmm2(kv_sub, key_sub, state_sub); // kv_sub = key_sub @ state_sub
        }
      }

      // --- Step 10: Compute delta = (V - kv_mem) * beta ---
      // delta: real shape [real_v_block_h, 1, 1, d]
      // Result is stored in kv_mem to save memory (delta is no longer needed
      // after)
      auto delta = make_tensor<T>(v_block_shape, v_real_shape);
      tiu::fsub(delta, V_lt,
                kv_mem); // delta = V - kv_mem, shape: [real_v_block_h, 1, 1, d]
      tiu::fmul(kv_mem, delta, beta_scalar); // kv_mem (reused) = delta * beta,
                                             // shape: [real_v_block_h, 1, 1, d]

      // --- Step 11: Update state: state_result += K^T (outer) delta ---
      // For each k-head i and its associated v-head j:
      //   k_sub    : [1, d, 1, 1]  (i-th column vector from K_col)
      //   delta_sub: [1, 1, 1, d]  (j-th scaled-delta row, stored in kv_mem)
      //   outer_sub: [1, d, 1, d]  outer product k_sub @ delta_sub
      // outer_prod overall real shape: [real_v_block_h, d, 1, d]
      auto outer_prod = make_tensor<T>(
          state_block_shape, state_real_shape); // [real_v_block_h, d, 1, d]
      for (int i = 0; i < real_block_h; i++) {
        for (int j = 0; j < vhead_per_khead; j++) {
          dim4 i_offset = {i, 0, 0, 0};
          auto k_sub = K_col.sub_view(sub_kcol_shape, i_offset); // [1, d, 1, 1]
          dim4 ij_offset = {i * vhead_per_khead + j, 0, 0, 0};
          auto delta_sub =
              kv_mem.sub_view(sub_vec_shape, ij_offset); // [1, 1, 1, d]
          auto outer_sub =
              outer_prod.sub_view(sub_state_shape, ij_offset); // [1, d, 1, d]
          tiu::fmm2(outer_sub, k_sub,
                    delta_sub); // outer_sub = k_sub (outer) delta_sub
        }
      }

      // state_result += outer_prod, shape: [real_v_block_h, d, 1, d]
      tiu::fadd(state_result, state_result, outer_prod);

      // --- Step 12: Read output: kv_mem = state_result^T @ query ---
      // For each k-head i and its associated v-head j:
      //   query_sub       : [1, 1, 1, d]  (i-th scaled query row)
      //   state_sub       : [1, d, 1, d]  (j-th updated state matrix)
      //   result_state_sub: [1, 1, 1, d]  output = query_sub @ state_sub
      // Output reuses kv_mem buffer, overall real shape: [real_v_block_h, 1, 1,
      // d]
      for (int i = 0; i < real_block_h; i++) {
        dim4 i_offset = {i, 0, 0, 0};
        auto query_sub = Q_lt.sub_view(sub_vec_shape, i_offset); // [1, 1, 1, d]
        for (int j = 0; j < vhead_per_khead; j++) {
          dim4 ij_offset = {i * vhead_per_khead + j, 0, 0, 0};
          auto state_sub =
              state_result.sub_view(sub_state_shape, ij_offset); // [1, d, 1, d]
          auto result_state_sub =
              kv_mem.sub_view(sub_vec_shape, ij_offset); // [1, 1, 1, d]
          tiu::fmm2(result_state_sub, query_sub,
                    state_sub); // out = query @ state
        }
      }

      // Store attention output: [real_v_block_h, 1, 1, d] -> g_out[b_idx, 0,
      // h_v_idx, 0]
      dim4 out_offset = {b_idx, 0, h_v_idx, 0};
      dim4 out_shape = {1, 1, real_v_block_h, d};
      dma::store(g_out.sub_view(out_shape, out_offset).view(v_real_shape),
                 kv_mem);

      // Store updated state: [real_v_block_h, d, 1, d] -> g_state[b_idx,
      // h_v_idx, 0, 0]
      dma::store(g_state.sub_view(cur_state_global_shape, state_offset)
                     .view(state_real_shape),
                 state_result);
    }
  }
}

__KERNEL__ void recurrent_gated_delta_rule_bf16(
    bf16 *ptr_core_attn_out, bf16 *ptr_last_recurrent_state, bf16 *ptr_Q,
    bf16 *ptr_K, bf16 *ptr_V, bf16 *ptr_g, bf16 *ptr_beta, int B, float scale,
    int core_num, const int num_k_heads, const int num_v_heads, const int d,
    const int use_qk_l2norm, const int block_h) {
  recurrent_gated_delta_rule_kernel<bf16>(
      ptr_core_attn_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
      ptr_beta, B, scale, core_num, num_k_heads, num_v_heads, d, use_qk_l2norm,
      block_h);
}

__KERNEL__ void recurrent_gated_delta_rule_f16(
    fp16 *ptr_core_attn_out, fp16 *ptr_last_recurrent_state, fp16 *ptr_Q,
    fp16 *ptr_K, fp16 *ptr_V, fp16 *ptr_g, fp16 *ptr_beta, int B, float scale,
    int core_num, const int num_k_heads, const int num_v_heads, const int d,
    const int use_qk_l2norm, const int block_h) {
  recurrent_gated_delta_rule_kernel<fp16>(
      ptr_core_attn_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
      ptr_beta, B, scale, core_num, num_k_heads, num_v_heads, d, use_qk_l2norm,
      block_h);
}

__TEST__ void recurrent_gated_delta_rule_test() {
  const int B = 1;
  const int num_k_heads = 16;
  const int num_v_heads = 64;
  const int d = 128;
  const int use_qk_l2norm = 1;
  const int core_num = 1;
  const int block_h = 16;
  const float scale = 0.08838834764831844;

  dim4 Q_shape = {B, 1, num_k_heads, d};
  dim4 K_shape = {B, 1, num_k_heads, d};
  dim4 V_shape = {B, 1, num_v_heads, d};
  dim4 g_shape = {B, 1, 1, num_v_heads};
  dim4 beta_shape = {B, 1, 1, num_v_heads};
  dim4 core_attn_out_shape = {B, 1, num_v_heads, d};
  dim4 last_recurrent_state_shape = {B, num_v_heads, d, d};

  auto ptr_core_attn_out = malloc<bf16>(&core_attn_out_shape);
  auto ptr_last_recurrent_state = malloc<bf16>(&last_recurrent_state_shape);
  auto ptr_Q = malloc<bf16>(&Q_shape);
  auto ptr_K = malloc<bf16>(&K_shape);
  auto ptr_V = malloc<bf16>(&V_shape);
  auto ptr_g = malloc<bf16>(&g_shape);
  auto ptr_beta = malloc<bf16>(&beta_shape);

  ppl::read_npz(ptr_Q, "recurrent_gated_delta_rule_input.npz", "in0");
  ppl::read_npz(ptr_K, "recurrent_gated_delta_rule_input.npz", "in1");
  ppl::read_npz(ptr_V, "recurrent_gated_delta_rule_input.npz", "in2");
  ppl::read_npz(ptr_g, "recurrent_gated_delta_rule_input.npz", "in3");
  ppl::read_npz(ptr_beta, "recurrent_gated_delta_rule_input.npz", "in4");
  ppl::read_npz(ptr_last_recurrent_state,
                "recurrent_gated_delta_rule_input.npz", "in5");

  recurrent_gated_delta_rule_bf16(ptr_core_attn_out, ptr_last_recurrent_state,
                                  ptr_Q, ptr_K, ptr_V, ptr_g, ptr_beta, B,
                                  scale, core_num, num_k_heads, num_v_heads, d,
                                  use_qk_l2norm, block_h);
}
