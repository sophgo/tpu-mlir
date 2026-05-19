//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "w8a8_block_matmul.h"
#include "ppl_static_host.h"
#include <assert.h>
#include <cstdio>
#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

static int align_2n(int x, int limit = 256) {
  int p = 1;
  if (x >= limit) {
    return limit;
  }
  while (p * 2 <= x) {
    p *= 2;
  }
  return p;
}

// ======================================
// Global GenInterface
// ======================================
void w8a8_block_matmul_tiling(gaddr_t ptr_out, gaddr_t ptr_in,
                              gaddr_t ptr_weight, gaddr_t ptr_weight_scale,
                              int G, int M, int K, int N, int core_num,
                              int block_size_k, int block_size_n, int32_t dtype,
                              int &P_G, int &P_M, int &P_N, int &P_K,
                              int &TILE_M, int &TILE_K, int &TILE_N) {
  auto func =
      dtype == DTYPE_BFP16 ? w8a8_block_matmul_bf16 : w8a8_block_matmul_f16;
  int ret = 0;

  // Initial parallelism: split N across cores; G/M/K kept on a single core.
  P_G = 1;
  P_M = 1;
  P_N = core_num;
  P_K = 1;
  TILE_M = M / P_M;
  TILE_N = N / P_N;
  TILE_K = K / P_K;

  while (TILE_K >= block_size_k && TILE_N >= block_size_n) {
    printf("W8A8BlockMatmul try P_G:%d P_M:%d P_N:%d P_K:%d "
           "TILE_M:%d TILE_K:%d TILE_N:%d\n",
           P_G, P_M, P_N, P_K, TILE_M, TILE_K, TILE_N);
    ret = func(ptr_out, ptr_in, ptr_weight, ptr_weight_scale, G, M, K, N,
               core_num, P_G, P_M, P_N, P_K, TILE_K, TILE_N, TILE_M,
               block_size_k, block_size_n);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      // On address-assign pressure, shrink tiles in order: N -> M -> K.
      if (TILE_N > block_size_n) {
        int next_tile_n = (TILE_N / 2 / block_size_n) * block_size_n;
        if (next_tile_n < block_size_n) {
          next_tile_n = block_size_n;
        }
        TILE_N = next_tile_n;
      } else if (TILE_M > 1) {
        TILE_M /= 2;
      } else if (TILE_K > block_size_k) {
        int next_tile_k = (TILE_K / 2 / block_size_k) * block_size_k;
        if (next_tile_k < block_size_k) {
          next_tile_k = block_size_k;
        }
        TILE_K = next_tile_k;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  if (ret != 0) {
    printf("Error: w8a8_block_matmul kernel failed due to address "
           "assignment failure\n");
    exit(-1);
  }
}

// static interface
void api_w8a8_block_matmul_global(void *param, size_t param_size, void *input,
                                  void *output) {
  auto *_param = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const int core_num = get_core_num();
  // Input spec layout: in_spec[0]=activation, in_spec[1]=weight,
  //                    in_spec[2]=weight_scale.
  // Activation/output use [G, M, 1, K] / [G, M, 1, N] layout, weight is
  // [G, N, 1, K].
  int G = in_spec[0].shape[0];
  int M = in_spec[0].shape[1];
  int K = in_spec[0].shape[2];
  int N = in_spec[1].shape[0];
  int P_G, P_M, P_N, P_K, TILE_M, TILE_K, TILE_N;
  w8a8_block_matmul_tiling(
      out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr, G, M,
      K, N, core_num, _param->block_size_k, _param->block_size_n,
      in_spec[0].dtype, P_G, P_M, P_N, P_K, TILE_M, TILE_K, TILE_N);
}

// dynamic interface
int api_dyn_w8a8_block_matmul_global(void *param, void *input, void *output,
                                     void *buffer) {
  auto *_param = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const int core_num = get_core_num();
  auto func = in_spec[0].dtype == DTYPE_BFP16
                  ? fill_w8a8_block_matmul_bf16_struct
                  : fill_w8a8_block_matmul_f16_struct;
  int G = in_spec[0].shape[0];
  int M = in_spec[0].shape[1];
  int K = in_spec[0].shape[2];
  int N = in_spec[1].shape[0];
  int P_G = 1, P_M = 1, P_N = core_num, P_K = 1;
  int TILE_M = M, TILE_K = K, TILE_N = N;
  if (buffer != nullptr) {
    w8a8_block_matmul_tiling(
        out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr, G,
        M, K, N, core_num, _param->block_size_k, _param->block_size_n,
        in_spec[0].dtype, P_G, P_M, P_N, P_K, TILE_M, TILE_K, TILE_N);
  }
  return func(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr,
              in_spec[2].addr, G, M, K, N, core_num, P_G, P_M, P_N, P_K, TILE_K,
              TILE_N, TILE_M, _param->block_size_k, _param->block_size_n,
              buffer);
}

#ifdef __cplusplus
}
#endif
