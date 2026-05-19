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

// Per-tile dynamic activation requantization: bf16/fp16 -> fp8e4m3 with a
// per-(BLOCK_C, group) scale tensor.
//   in/out: real shape [1, real_c, 1, real_w], block shape [1, BLOCK_C, 1,
//   BLOCK_W] scale_local: [1, real_c, 1, real_w / group_size]
//   group_size: per-group quantization granularity along W.
template <typename TYPE>
void act_requant(tensor<fp8e4m3> &output_local, tensor<TYPE> &scale_local,
                 tensor<TYPE> &input_local, const int BLOCK_C,
                 const int BLOCK_W, int real_c, int real_w,
                 const int group_size) {
  dim4 quant_block_shape = {1, BLOCK_C, BLOCK_W / group_size, group_size};
  dim4 quant_real_shape = {1, real_c, real_w / group_size, group_size};
  dim4 max_block_shape = {1, BLOCK_C, BLOCK_W / group_size, 1};
  dim4 max_real_shape = {1, real_c, real_w / group_size, 1};

  auto cur_in_amax = make_tensor<TYPE>(max_block_shape, max_real_shape);
  auto cur_out_buffer = make_tensor<TYPE>(quant_block_shape, quant_real_shape);
  auto input_buffer =
      make_tensor<TYPE>(quant_block_shape, quant_real_shape, TPU_ROW_ALIGN);
  tiu::move(input_buffer, input_local);
  tiu::reduce(cur_in_amax, input_buffer, ALL_REDUCE_AMAX);
  fp32 eps = 1e-4f;
  tiu::fmax(cur_in_amax, cur_in_amax, eps);
  tiu::fmul(scale_local.view(max_real_shape), cur_in_amax, 1.0f / 448.0f);
  tiu::fdiv(cur_out_buffer, input_local.view(quant_real_shape),
            scale_local.view(max_real_shape));
  tiu::cast(output_local.view(quant_real_shape), cur_out_buffer);
}

/*
W8A8 Block Matmul kernel.

Computes a NT-layout block-quantized matmul with:
  - Activation in bf16/fp16, dynamically requantized per-tile to fp8e4m3
  - Weight in fp8e4m3 with a per-(block_size_n, block_size_k) bf16 scale

Inputs:
  in:           [G, M, 1, K]                                       (ACT_TYPE)
  weight:       [G, N, 1, K]                                       (fp8e4m3)
  weight_scale: [G, ceil(N/block_size_n), 1, ceil(K/block_size_k)] (ACT_TYPE)

Output:
  out:          [G, M, 1, N]                                       (ACT_TYPE)

Cores split the (G, M, N, K) iteration space along four logical axes
P_G * P_M * P_N * P_K = core_num. When P_K > 1 partial sums are combined via
all_reduce. The fp8 matmul is decomposed by inner_k = block_size_k chunks so
that each chunk gets its own (per-row, per-N-group) scale.
*/
template <typename ACT_TYPE>
void w8a8_block_matmul_kernel(ACT_TYPE *ptr_out, ACT_TYPE *ptr_in,
                              fp8e4m3 *ptr_weight, ACT_TYPE *ptr_weight_scale,
                              const int G, const int M, const int K,
                              const int N, const int core_num, const int P_G,
                              const int P_M, const int P_N, const int P_K,
                              const int TILE_K, const int TILE_N,
                              const int TILE_M, const int block_size_k,
                              const int block_size_n) {
  int core_idx = get_core_index();
  if (core_idx >= core_num) {
    return;
  }

  assert(P_G > 0 && P_M > 0 && P_N > 0 && P_K > 0);
  assert(TILE_K % block_size_k == 0);
  assert(TILE_N % block_size_n == 0);

  // Decompose linear core_idx into 4D logical (g, m, n, k) core coordinates.
  int idx_k = core_idx % P_K;
  int temp = core_idx / P_K;
  int idx_n = temp % P_N;
  temp = temp / P_N;
  int idx_m = temp % P_M;
  temp = temp / P_M;
  int idx_g = temp;

  int Gs, Ge, G_slice;
  int Ms, Me, M_slice;
  int Ns, Ne;
  int Ks, Ke;

  // G/M: simple equal split with div_up.
  G_slice = div_up(G, P_G);
  Gs = idx_g * G_slice;
  Ge = Gs + G_slice;
  if (Ge > G) Ge = G;
  if (Gs > G) Gs = G;

  M_slice = div_up(M, P_M);
  Ms = idx_m * M_slice;
  Me = Ms + M_slice;
  if (Me > M) Me = M;
  if (Ms > M) Ms = M;

  // N: align to 128 to keep eu-friendly tiles, last shard absorbs the tail.
  int n_base = max((N / P_N) / 128 * 128, 128);
  Ns = idx_n * n_base;
  if (idx_n == P_N - 1) {
    Ne = N;
  } else {
    Ne = min((idx_n + 1) * n_base, N);
  }

  // K: same 128-alignment + tail-absorption rule as N.
  int k_base = max((K / P_K) / 128 * 128, 128);
  Ks = idx_k * k_base;
  if (idx_k == P_K - 1) {
    Ke = K;
  } else {
    Ke = min((idx_k + 1) * k_base, K);
  }

  dim4 in_global_shape = {G, M, 1, K};
  dim4 out_global_shape = {G, M, 1, N};
  dim4 weight_global_shape = {G, N, 1, K};
  dim4 weight_scale_global_shape = {G, div_up(N, block_size_n), 1,
                                    div_up(K, block_size_k)};

  auto in_gtensor = gtensor<ACT_TYPE>(in_global_shape, GLOBAL, ptr_in);
  auto out_gtensor = gtensor<ACT_TYPE>(out_global_shape, GLOBAL, ptr_out);
  auto weight_gtensor =
      gtensor<fp8e4m3>(weight_global_shape, GLOBAL, ptr_weight);
  auto weight_scale_gtensor =
      gtensor<ACT_TYPE>(weight_scale_global_shape, GLOBAL, ptr_weight_scale);

  const int BLOCK_M = TILE_M;
  const int BLOCK_K = TILE_K;
  const int BLOCK_N = TILE_N;

  dim4 in_block_shape = {1, BLOCK_M, 1, BLOCK_K};
  dim4 in_scale_block_shape = {1, BLOCK_M, 1, div_up(BLOCK_K, block_size_k)};
  dim4 out_block_shape = {1, BLOCK_M, 1, BLOCK_N};
  dim4 w_sub_block_shape = {1, BLOCK_N, 1, block_size_k};
  dim4 ws_sub_block_shape = {1, div_up(BLOCK_N, block_size_n), 1, 1};
  dim4 s_block_shape = {1, BLOCK_M, 1, 1};
  // When K is sharded across cores each core only computes a partial sum and
  // we have to all-reduce on store.
  bool need_all_reduce = ((Ke - Ks) < K || TILE_K < K) ? true : false;

  for (int g_idx = Gs; g_idx < Ge; g_idx += 1) {
    for (int m_idx = Ms; m_idx < Me; m_idx += TILE_M) {
      int cur_M = min(BLOCK_M, Me - m_idx);
      for (int k_idx = Ks; k_idx < Ke; k_idx += TILE_K) {
        ppl::enable_pipeline();  // TODO
        int cur_K = min(BLOCK_K, Ke - k_idx);

        // Load activation tile and dynamically requant to fp8 (per-row, per
        // block_size_k group).
        dim4 in_shape = {1, cur_M, 1, cur_K};
        dim4 in_scale_shape = {1, cur_M, 1, div_up(cur_K, block_size_k)};
        dim4 in_offset = {g_idx, m_idx, 0, k_idx};
        auto in = make_tensor<ACT_TYPE>(in_block_shape, in_shape);
        auto quant_in = make_tensor<fp8e4m3>(in_block_shape, in_shape);
        auto in_scale =
            make_tensor<ACT_TYPE>(in_scale_block_shape, in_scale_shape);
        dma::load(in, in_gtensor.sub_view(in_shape, in_offset));
        act_requant<ACT_TYPE>(quant_in, in_scale, in, BLOCK_M, BLOCK_K, cur_M,
          cur_K, block_size_k);
        // Activation scale is consumed in fp32 to keep the manual scaling
        // numerically tight.
        auto in_scale_fp32 =
            make_tensor<fp32>(in_scale_block_shape, in_scale_shape);
        tiu::cast(in_scale_fp32, in_scale);

        for (int N_idx = Ns; N_idx < Ne; N_idx += TILE_N) {
          int cur_N = min(TILE_N, Ne - N_idx);
          dim4 out_shape = {1, cur_M, 1, cur_N};
          dim4 out_offset = {g_idx, m_idx, 0, N_idx};

          // fp32 accumulator across all inner_k blocks of this (M,N) tile.
          auto acc_fp32 = make_tensor<fp32>(out_block_shape, out_shape);
          tiu::fill(acc_fp32, 0.0f);

          for (int inner_k = 0; inner_k < cur_K; inner_k += block_size_k) {
            int bk = min(block_size_k, cur_K - inner_k);

            // Move out the inner_k slice of the requantized activation into a
            // standalone aligned tensor so fmm2_nt sees a clean stride layout.
            dim4 sub_in_shape = {1, cur_M, 1, bk};
            dim4 sub_in_offset = {0, 0, 0, inner_k};
            dim4 sub_in_block_shape = {1, BLOCK_M, 1, block_size_k};
            auto quant_in_aligned =
                make_tensor<fp8e4m3>(sub_in_block_shape, sub_in_shape);
            tiu::move(quant_in_aligned,
                      quant_in.sub_view(sub_in_shape, sub_in_offset));

            // Load weight slice for this inner_k chunk.
            dim4 weight_shape = {1, cur_N, 1, bk};
            dim4 weight_offset = {g_idx, N_idx, 0, k_idx + inner_k};
            auto cur_weight =
                make_tensor<fp8e4m3>(w_sub_block_shape, weight_shape);
            dma::load(cur_weight,
                      weight_gtensor.sub_view(weight_shape, weight_offset));

            // Load weight_scale for the (N-group, inner_k-group) cell and
            // promote to fp32.
            dim4 ws_shape = {1, cur_N / block_size_n, 1, 1};
            dim4 ws_offset = {g_idx, N_idx / block_size_n, 0,
                              (k_idx + inner_k) / block_size_k};
            auto ws_block_act =
                make_tensor<ACT_TYPE>(ws_sub_block_shape, ws_shape);
            dma::load(ws_block_act,
                      weight_scale_gtensor.sub_view(ws_shape, ws_offset));
            auto ws_block_fp32 =
                make_tensor<fp32>(ws_sub_block_shape, ws_shape);
            tiu::cast(ws_block_fp32, ws_block_act);

            // Step 1: raw fp8 NT matmul -> fp32 sub_out [cur_M, cur_N].
            dim4 sub_out_shape = {1, cur_M, 1, cur_N};
            auto sub_out_fp32 =
                make_tensor<fp32>(out_block_shape, sub_out_shape);
            tiu::fmm2_nt(sub_out_fp32, quant_in_aligned, cur_weight);

            // Step 2/3: per N-group, broadcast w_scale to [cur_M, 1], multiply
            // by per-row act scale, and fmac into the fp32 accumulator.
            int num_ng = div_up(cur_N, block_size_n);
            for (int ng = 0; ng < num_ng; ++ng) {
              int n_start = ng * block_size_n;
              int actual_n = min(block_size_n, cur_N - n_start);

              // act-scale row vector for this inner_k-group: [cur_M, 1]
              dim4 is_shape = {1, cur_M, 1, 1};
              dim4 is_offset = {0, 0, 0, inner_k / block_size_k};

              // Pick the scalar w_scale for this N-group and broadcast it
              // across LANE_NUM lanes so it can be used with stride-0 along C.
              dim4 ws_scalar_shape = {1, 1, 1, 1};
              dim4 ws_scalar_offset = {0, ng, 0, 0};
              auto ws_scalar =
                  ws_block_fp32.sub_view(ws_scalar_shape, ws_scalar_offset);

              dim4 ws_bc_shape = {1, LANE_NUM, 1, 1};
              auto tmp_ws_bc =
                  make_tensor<fp32>(ws_bc_shape, ws_bc_shape, TPU_COMPACT);
              dma::broadcast(tmp_ws_bc, ws_scalar);

              dim4 bc_shape = {1, cur_M, 1, 1};
              dim4 _stride = get_stride<fp32>(ws_bc_shape, TPU_ALIGN);
              dim4 bc_stride = {_stride.n, 0, _stride.c, _stride.w};
              auto ws_bc = tmp_ws_bc.view(bc_shape, bc_stride);

              // scale_res = w_scale_bc * in_scale -> [cur_M, 1]
              auto scale_res = make_tensor<fp32>(s_block_shape, is_shape);
              tiu::fmul(scale_res, ws_bc,
                        in_scale_fp32.sub_view(is_shape, is_offset));

              // acc[:, n_start:n_start+actual_n] +=
              //     scale_res * sub_out[:, n_start:n_start+actual_n]
              dim4 col_shape = {1, cur_M, 1, actual_n};
              dim4 col_offset = {0, 0, 0, n_start};
              tiu::fmac(acc_fp32.sub_view(col_shape, col_offset), scale_res,
                        sub_out_fp32.sub_view(col_shape, col_offset));
            }
          }

          // Cast fp32 accumulator to ACT_TYPE and either reduce-store (when
          // partial sums need cross-core combine) or plain-store.
          auto cur_out = make_tensor<ACT_TYPE>(out_block_shape, out_shape);
          tiu::cast(cur_out, acc_fp32);
          if (need_all_reduce) {   // TODO: FIXsME!
            dma::reduce(out_gtensor.sub_view(out_shape, out_offset), cur_out,
                        all_reduce_psum_t::ALL_REDUCE_PSUM_WR,
                        (all_reduce_opcode_t)ALL_REDUCE_ADD);
            sync();
          } else {
            dma::store(out_gtensor.sub_view(out_shape, out_offset), cur_out);
          }
        }
      }
    }
  }
}

__KERNEL__ void w8a8_block_matmul_bf16(
    bf16 *ptr_out, bf16 *ptr_in, fp8e4m3 *ptr_weight, bf16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_G, const int P_M, const int P_N, const int P_K,
    const int TILE_K, const int TILE_N, const int TILE_M,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_kernel<bf16>(
      ptr_out, ptr_in, ptr_weight, ptr_weight_scale, G, M, K, N, core_num, P_G,
      P_M, P_N, P_K, TILE_K, TILE_N, TILE_M, block_size_k, block_size_n);
}

__KERNEL__ void w8a8_block_matmul_f16(
    fp16 *ptr_out, fp16 *ptr_in, fp8e4m3 *ptr_weight, fp16 *ptr_weight_scale,
    const int G, const int M, const int K, const int N, const int core_num,
    const int P_G, const int P_M, const int P_N, const int P_K,
    const int TILE_K, const int TILE_N, const int TILE_M,
    const int block_size_k, const int block_size_n) {
  w8a8_block_matmul_kernel<fp16>(
      ptr_out, ptr_in, ptr_weight, ptr_weight_scale, G, M, K, N, core_num, P_G,
      P_M, P_N, P_K, TILE_K, TILE_N, TILE_M, block_size_k, block_size_n);
}

__TEST__ void w8a8_block_matmul_test() {
  const int G = 1;
  const int M = 48;
  const int K = 7168;
  const int N = 2048;
  const int block_size_k = 128;
  const int block_size_n = 128;

  dim4 input_shape = {G, M, 1, K};
  dim4 output_shape = {G, M, 1, N};
  dim4 w_shape = {G, N, 1, K};
  dim4 w_scale_shape = {G, div_up(N, block_size_n), 1, div_up(K, block_size_k)};

  auto output = ppl::malloc<bf16>(&output_shape);
  auto input = ppl::rand<bf16>(&input_shape, -448, 448);
  auto w = ppl::rand<fp8e4m3>(&w_shape, -448, 448);
  auto w_scale = ppl::rand<bf16>(&w_scale_shape, 0.0022, 0.0046);

#if defined(__sg2262__)
  const int core_num = 8;
  const int P_G = 1;
  const int P_M = 1;
  const int P_N = 2;
  const int P_K = 4;
  const int TILE_M = 16;
  const int TILE_K = 256;
  const int TILE_N = 256;
#elif defined(__bm1684x2__)
  const int core_num = 4;
  const int P_G = 1;
  const int P_M = 1;
  const int P_N = 2;
  const int P_K = 2;
  const int TILE_M = 16;
  const int TILE_K = 256;
  const int TILE_N = 256;
#else
  const int core_num = 1;
  const int P_G = 1;
  const int P_M = 1;
  const int P_N = 1;
  const int P_K = 1;
  const int TILE_M = 16;
  const int TILE_K = 256;
  const int TILE_N = 256;
#endif

  w8a8_block_matmul_bf16(output, input, w, w_scale, G, M, K, N, core_num, P_G,
                         P_M, P_N, P_K, TILE_K, TILE_N, TILE_M, block_size_k,
                         block_size_n);
}
