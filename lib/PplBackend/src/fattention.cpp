#include "flash_attention_gqa_bf16.h"
#include "flash_attention_gqa_f16.h"
#include "helper.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
#include <assert.h>
#include <cstdio>
#include <stddef.h>
#include <stdint.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void api_fattention_global(void *param, size_t param_size, void *input_spec,
                           void *output_spec, const char *chip, void *cmdid) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto q_spec = in_spec;
  auto k_spec = in_spec + 1;
  auto v_spec = in_spec + 2;
  auto mask_spec = in_spec + 3;

  // tiling
  int ret = 0;
  int dmax = align_up(_param->common.dim, 32 /*eu num*/);
  int block_m, block_k, block_h;
  std::string chip_str(chip);
  if (chip_str == PPL_BM1688) {
    if (_param->common.mq == 1) {
      block_m = 32;
      block_k = 352;
      block_h = 8;
    } else {
      block_m = 224; // 128;
        block_k = 80; // 128;
      if (in_spec[0].dtype == DTYPE_FP16) {
        block_k = 96; // 128;
      }
      block_h = 8; // 32;
    }
  } else {
    if (_param->common.mq == 1) {
      block_m = 64;
      block_k = 192;
      block_h = 32;
    } else {
      block_m = 256; // 128;
      block_k = 256; // 128;
      block_h = 32;  // 32;
    }
  }

  while (block_m > 0 && block_k > 0) {
    if (in_spec[0].dtype == DTYPE_FP16) {
      ret = flash_attention_gqa_f16(
          chip, cmdid, out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
          _param->common.hasmask ? mask_spec->addr : 0, _param->common.batch,
          _param->common.mq, _param->common.mk, _param->common.dim,
          _param->common.q_head, _param->common.kv_head, _param->common.scale,
          _param->common.hasmask, dmax, block_m, block_k, block_h);
    } else if (in_spec[0].dtype == DTYPE_BFP16) {
      ret = flash_attention_gqa_bf16(
          chip, cmdid, out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
          _param->common.hasmask ? mask_spec->addr : 0, _param->common.batch,
          _param->common.mq, _param->common.mk, _param->common.dim,
          _param->common.q_head, _param->common.kv_head, _param->common.scale,
          _param->common.hasmask, dmax, block_m, block_k, block_h);
    } else {
      assert(0);
    }
    CHECK_PPL_RET(ret);
    if (ret == PplAddressAssignErr) {
      block_m -= 2;
      block_k -= 2;
      continue;
    }
    break;
  }
}

#ifdef __cplusplus
}
#endif
