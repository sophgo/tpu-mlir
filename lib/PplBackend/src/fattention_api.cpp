#include "fattention_bf16.h"
#include "fattention_fp16.h"
#include "fattention_v2.h"
#include "helper.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
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
                             int dtype, bool high_precision);

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
  fattention_tiling(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                    _param->common.hasmask ? mask_spec->addr : 0,
                    _param->common.batch, _param->common.mq, _param->common.mk,
                    _param->common.dim, _param->common.q_head,
                    _param->common.kv_head, _param->common.scale,
                    _param->common.hasmask, core_num, in_spec[0].dtype,
                    _param->common.high_precision);
}

#ifdef __cplusplus
}
#endif
