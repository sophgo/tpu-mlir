//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chunk_gated_delta_rule.h"
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
void api_chunk_gated_delta_rule_global(void *param, size_t param_size,
                                       void *input, void *output) {
  auto *_param = (chunk_gated_delta_rule_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  auto func = in_spec->dtype == DTYPE_BFP16 ? chunk_gated_delta_rule_bf16
                                            : chunk_gated_delta_rule_f16;
  const int core_num = get_core_num();
  int ret =
      func(out_spec[0].addr,
           in_spec[5].addr, // core_attn_out, last_recurrent_state
           in_spec[0].addr, in_spec[1].addr, in_spec[2].addr, in_spec[3].addr,
           in_spec[4].addr, // Q, K, V, g, beta
           in_spec[6].addr, in_spec[7].addr, in_spec[8].addr,
           in_spec[9].addr, // triu_mask, strict_triu_mask, tril_mask, eye
           in_spec[0].shape[0], in_spec[0].shape[1], _param->num_k_heads,
           _param->num_v_heads, _param->d, _param->scale, _param->chunk_size,
           _param->use_qk_l2norm, core_num, _param->chunk_size /*TileS*/);
  if (ret != 0) {
    printf("Error: chunk_gated_delta_rule kernel returned %d\n", ret);
    exit(-1);
  }
}

// dynamic interface
int api_dyn_chunk_gated_delta_rule_global(void *param, void *input,
                                          void *output, void *buffer) {
  auto *_param = (chunk_gated_delta_rule_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const int core_num = get_core_num();
  auto func = in_spec->dtype == DTYPE_BFP16
                  ? fill_chunk_gated_delta_rule_bf16_struct
                  : fill_chunk_gated_delta_rule_f16_struct;
  return func(out_spec[0].addr,
              in_spec[5].addr, // core_attn_out, last_recurrent_state
              in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
              in_spec[3].addr, in_spec[4].addr, // Q, K, V, g, beta
              in_spec[6].addr, in_spec[7].addr, in_spec[8].addr,
              in_spec[9].addr, // triu_mask, strict_triu_mask, tril_mask, eye
              in_spec[0].shape[0], in_spec[0].shape[1], _param->num_k_heads,
              _param->num_v_heads, _param->d, _param->scale, _param->chunk_size,
              _param->use_qk_l2norm, core_num, _param->chunk_size /*TileS*/,
              buffer);
}

#ifdef __cplusplus
}
#endif
