//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "recurrent_gated_delta_rule.h"
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
void recurrent_gated_delta_rule_tiling(
    gaddr_t ptr_out, gaddr_t ptr_Q, gaddr_t ptr_K, gaddr_t ptr_V, gaddr_t ptr_g,
    gaddr_t ptr_beta, gaddr_t ptr_last_recurrent_state, int B, float scale,
    int core_num, int num_k_heads, int num_v_heads, int d, int use_qk_l2norm,
    int32_t dtype, int &block_h) {
  auto func = dtype == DTYPE_BFP16 ? recurrent_gated_delta_rule_bf16
                                   : recurrent_gated_delta_rule_f16;
  block_h = num_k_heads / core_num / 2;
  if (block_h < 1) {
    block_h = 1;
  }
  int ret = 0;
  while (block_h > 0) {
    printf("RecurrentGatedDeltaRule try block_h:%d/%d\n", block_h, num_k_heads);
    ret = func(ptr_out, ptr_last_recurrent_state, ptr_Q, ptr_K, ptr_V, ptr_g,
               ptr_beta, B, scale, core_num, num_k_heads, num_v_heads, d,
               use_qk_l2norm, block_h);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      block_h = block_h / 2;
    } else {
      break;
    }
  }
  if (block_h == 0 || ret != 0) {
    printf("Error: recurrent_gated_delta_rule kernel failed due to address "
           "assignment failure\n");
    exit(-1);
  }
}

// static interface
void api_recurrent_gated_delta_rule_global(void *param, size_t param_size,
                                           void *input, void *output) {
  auto *_param = (recurrent_gated_delta_rule_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  auto func = in_spec[0].dtype == DTYPE_BFP16 ? recurrent_gated_delta_rule_bf16
                                              : recurrent_gated_delta_rule_f16;
  const int core_num = get_core_num();
  int block_h = 0;
  recurrent_gated_delta_rule_tiling(
      out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
      in_spec[3].addr, in_spec[4].addr, in_spec[5].addr, in_spec[0].shape[0],
      _param->scale, core_num, _param->num_k_heads, _param->num_v_heads,
      _param->d, _param->use_qk_l2norm ? 1 : 0, in_spec[0].dtype, block_h);
}

// dynamic interface
int api_dyn_recurrent_gated_delta_rule_global(void *param, void *input,
                                              void *output, void *buffer) {
  auto *_param = (recurrent_gated_delta_rule_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const int core_num = get_core_num();
  auto func = in_spec->dtype == DTYPE_BFP16
                  ? fill_recurrent_gated_delta_rule_bf16_struct
                  : fill_recurrent_gated_delta_rule_f16_struct;
  int block_h = _param->num_k_heads;
  if (buffer != nullptr) {
    recurrent_gated_delta_rule_tiling(
        out_spec[0].addr, in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
        in_spec[3].addr, in_spec[4].addr, in_spec[5].addr, in_spec[0].shape[0],
        _param->scale, core_num, _param->num_k_heads, _param->num_v_heads,
        _param->d, _param->use_qk_l2norm ? 1 : 0, in_spec[0].dtype, block_h);
  }
  return func(out_spec[0].addr,
              in_spec[5].addr, // core_attn_out, last_recurrent_state
              in_spec[0].addr, in_spec[1].addr, in_spec[2].addr,
              in_spec[3].addr, in_spec[4].addr, // Q, K, V, g, beta
              in_spec[0].shape[0], _param->scale, core_num, _param->num_k_heads,
              _param->num_v_heads, _param->d, _param->use_qk_l2norm ? 1 : 0,
              block_h, buffer);
}

#ifdef __cplusplus
}
#endif
