#include "fattention_bf16.h"
#include "fattention_fp16.h"
#include "fattention_v2.h"
#include "helper.h"
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
extern int fattention_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                             gaddr_t ptr_v, gaddr_t ptr_mask, int b, int qm,
                             int kvm, int d, int q_head, int kv_head,
                             float sqrt_d, int has_mask, int core_num,
                             int dtype, bool high_precision, int &block_m,
                             int &block_k, int &block_h);
// static interface
void api_fattention_global(void *param, size_t param_size, void *input_spec,
                           void *output_spec) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto q_spec = in_spec;
  auto k_spec = in_spec + 1;
  auto v_spec = in_spec + 2;
  auto mask_spec = in_spec + 3;
  const int core_num = get_core_num();
  int block_m, block_k, block_h;

  fattention_tiling(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                    _param->common.hasmask ? mask_spec->addr : 0,
                    _param->common.batch, _param->common.mq, _param->common.mk,
                    _param->common.dim, _param->common.q_head,
                    _param->common.kv_head, _param->common.scale,
                    _param->common.hasmask, core_num, in_spec[0].dtype,
                    _param->common.high_precision, block_m, block_k, block_h);
}

// dynamic interface
using DYN_ATTENTION = std::function<int(
    unsigned long long v1, unsigned long long v2, unsigned long long v3,
    unsigned long long v4, unsigned long long v5, int32_t v6, int32_t v7,
    int32_t v8, int32_t v9, int32_t v10, int32_t v11, float v12, int32_t v13,
    int32_t v14, int32_t v15, int32_t v16, int32_t v17, int32_t v18,
    void *buffer)>;
// fill_${OP_NAME}_struct gen automatic by ppl, the differ between ppl kernel
// func are with extra buffer param and return type
static DYN_ATTENTION get_dyn_attention_func(bool is_fp16, bool is_mha,
                                            bool high_precision) {
  if (is_mha && is_fp16) {
    return high_precision ? fill_flash_attention_mha_f16_high_precision_struct
                          : fill_flash_attention_mha_f16_struct;
  }
  if (is_mha && !is_fp16) {
    return high_precision ? fill_flash_attention_mha_bf16_high_precision_struct
                          : fill_flash_attention_mha_bf16_struct;
  }
  if (!is_mha && is_fp16) {
    return high_precision ? fill_flash_attention_gqa_f16_high_precision_struct
                          : fill_flash_attention_gqa_f16_struct;
  }
  if (!is_mha && !is_fp16) {
    return high_precision ? fill_flash_attention_gqa_bf16_high_precision_struct
                          : fill_flash_attention_gqa_bf16_struct;
  }
  // never go here
  return nullptr;
}
// dynamic interface
int api_dyn_fattention_global(void *param, void *input_spec, void *output_spec,
                              void *buffer) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto q_spec = in_spec;
  auto k_spec = in_spec + 1;
  auto v_spec = in_spec + 2;
  auto mask_spec = in_spec + 3;
  const int core_num = get_core_num();
  auto dtype = in_spec[0].dtype;
  auto q_head = _param->common.q_head;
  auto kv_head = _param->common.kv_head;
  auto high_precision = _param->common.high_precision;
  int block_m, block_k, block_h;
  if (buffer) {
    // get tile info
    fattention_tiling(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                      _param->common.hasmask ? mask_spec->addr : 0,
                      _param->common.batch, _param->common.mq,
                      _param->common.mk, _param->common.dim, q_head, kv_head,
                      _param->common.scale, _param->common.hasmask, core_num,
                      dtype, high_precision, block_m, block_k, block_h);
  }
  // If buffer is not null writre param info to buffer according to tile info,
  // return param struct lens.
  DYN_ATTENTION func = get_dyn_attention_func(
      dtype == DTYPE_FP16, q_head == kv_head, high_precision);
  return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
              _param->common.hasmask ? mask_spec->addr : 0,
              _param->common.batch, _param->common.mq, _param->common.mk,
              _param->common.dim, q_head, kv_head, _param->common.scale,
              _param->common.hasmask, core_num,
              align_up(_param->common.dim, 32 /*eu num*/), block_m, block_k,
              block_h, buffer);
}

#ifdef __cplusplus
}
#endif
