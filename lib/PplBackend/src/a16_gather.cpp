//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "a16_gather.h"
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
// ======================================
// Global GenInterface
// ======================================

// static interface
void api_a16_gather_global(void *param, size_t param_size, void *input,
                           void *output) {
  auto *_param = (a16_gather_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;

  // in_spec[0]: weight (uint8, [vocab_size, dim_packed])
  // in_spec[1]: indices (int32, arbitrary shape)
  // in_spec[2]: scale (bf16, [vocab_size, n_groups])
  // in_spec[3]: zp (uint8, [vocab_size, n_groups])
  // out_spec[0]: output (bf16, [...indices_shape, dim])

  int axis = _param->axis;
  int weight_bits = _param->weight_bits;
  int q_group_size = _param->q_group_size;

  assert(axis == 0 && "A16Gather only supports axis=0");

  // Calculate actual dim from packed dim
  int vocab_size = in_spec[0].shape[0];
  int dim_packed = in_spec[0].shape[1];
  int dim = (weight_bits == 4) ? (dim_packed * 2) : dim_packed;

  // Calculate total number of indices (supports arbitrary shape)
  int N = 1;
  for (int i = 0; i < in_spec[1].dims; i++) {
    N *= in_spec[1].shape[i];
  }

  // Call the actual TPU kernel
  int ret = 0;
  if (out_spec[0].dtype == DTYPE_BFP16) {
    ret = a16_gather_bf16(out_spec[0].addr, // output
                          in_spec[0].addr,  // weight
                          in_spec[1].addr,  // indices
                          in_spec[2].addr,  // scale
                          in_spec[3].addr,  // zp
                          vocab_size, dim, N, weight_bits, q_group_size);
  } else if (out_spec[0].dtype == DTYPE_FP16) {
    ret = a16_gather_f16(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr,
                         in_spec[2].addr, in_spec[3].addr, vocab_size, dim, N,
                         weight_bits, q_group_size);
  } else {
    printf("Error: a16_gather only supports BF16 and F16, got dtype=%d\n",
           out_spec[0].dtype);
    exit(-1);
  }

  if (ret != 0) {
    printf("Error: a16_gather kernel returned %d\n", ret);
    exit(-1);
  }
}

// dynamic interface
int api_dyn_a16_gather_global(void *param, void *input, void *output,
                              void *buffer) {
  auto *_param = (a16_gather_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;

  int axis = _param->axis;
  int weight_bits = _param->weight_bits;
  int q_group_size = _param->q_group_size;

  assert(axis == 0 && "A16Gather only supports axis=0");

  // Calculate actual dim from packed dim
  int vocab_size = in_spec[0].shape[0];
  int dim_packed = in_spec[0].shape[1];
  int dim = (weight_bits == 4) ? (dim_packed * 2) : dim_packed;

  // Calculate total number of indices (supports arbitrary shape)
  int N = 1;
  for (int i = 0; i < in_spec[1].dims; i++) {
    N *= in_spec[1].shape[i];
  }

  // For dynamic mode, we need to fill the struct
  auto func = out_spec->dtype == DTYPE_BFP16 ? fill_a16_gather_bf16_struct
                                             : fill_a16_gather_f16_struct;

  if (buffer != nullptr) {
    // If buffer is provided, call the static kernel first
    if (out_spec[0].dtype == DTYPE_BFP16) {
      int ret = a16_gather_bf16(
          out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
          in_spec[3].addr, vocab_size, dim, N, weight_bits, q_group_size);
      if (ret != 0) {
        printf("Error: a16_gather kernel returned %d\n", ret);
        return ret;
      }
    } else {
      int ret = a16_gather_f16(
          out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
          in_spec[3].addr, vocab_size, dim, N, weight_bits, q_group_size);
      if (ret != 0) {
        printf("Error: a16_gather kernel returned %d\n", ret);
        return ret;
      }
    }
  }

  return func(out_spec[0].addr, in_spec[0].addr, in_spec[1].addr,
              in_spec[2].addr, in_spec[3].addr, vocab_size, dim, N, weight_bits,
              q_group_size, buffer);
}

#ifdef __cplusplus
}
#endif
