#include "attention_GQA_mlir_bf16.h"
#include "attention_GQA_mlir_fp16.h"
#include "attention_MHA_mlir_bf16.h"
#include "attention_MHA_mlir_fp16.h"
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
static int bm1688_mha_bf16_decode[3] = {32, 288, 8};
static int bm1688_mha_bf16_encode[3] = {160, 96, 8};
static int bm1688_mha_fp16_decode[3] = {32, 352, 8};
static int bm1688_mha_fp16_encode[3] = {224, 96, 8};

static int bm1688_gqa_bf16_decode[3] = {32, 352, 8};
static int bm1688_gqa_bf16_encode[3] = {160, 96, 8};
static int bm1688_gqa_fp16_decode[3] = {32, 352, 8};
static int bm1688_gqa_fp16_encode[3] = {224, 96, 8};

static int bm1684x_mha_bf16_decode[3] = {64, 384, 16};
static int bm1684x_mha_bf16_encode[3] = {320, 96, 16};
static int bm1684x_mha_fp16_decode[3] = {64, 512, 16};
static int bm1684x_mha_fp16_encode[3] = {384, 128, 16};

static int bm1684x_gqa_bf16_decode[3] = {64, 512, 16};
static int bm1684x_gqa_bf16_encode[3] = {320, 96, 16};
static int bm1684x_gqa_fp16_decode[3] = {64, 768, 16};
static int bm1684x_gqa_fp16_encode[3] = {384, 128, 16};

using ATTENTION = std::function<int(
    const char *chip, void *pid_node, unsigned long long v1,
    unsigned long long v2, unsigned long long v3, unsigned long long v4,
    unsigned long long v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9,
    int32_t v10, int32_t v11, float v12, int32_t v13, int32_t v14, int32_t v15,
    int32_t v16, int32_t v17)>;

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
  bool is_mha = _param->common.q_head == _param->common.kv_head;
  bool is_decode = _param->common.mq == 1;
  bool is_fp16 = in_spec[0].dtype == DTYPE_FP16;
  int *split = nullptr;
  ATTENTION func = nullptr;
  if (chip_str == PPL_BM1688) {
    if (is_mha) {
      func = is_fp16 ? flash_attention_mha_f16 : flash_attention_mha_bf16;
      if (is_decode) {
        split = is_fp16 ? bm1688_mha_fp16_decode : bm1688_mha_bf16_decode;
      } else {
        split = is_fp16 ? bm1688_mha_fp16_encode : bm1688_mha_bf16_encode;
      }
    } else {
      func = is_fp16 ? flash_attention_gqa_f16 : flash_attention_gqa_bf16;
      if (is_decode) {
        split = is_fp16 ? bm1688_gqa_fp16_decode : bm1688_gqa_bf16_decode;
      } else {
        split = is_fp16 ? bm1688_gqa_fp16_encode : bm1688_gqa_bf16_encode;
      }
    }
  } else {
    if (is_mha) {
      func = is_fp16 ? flash_attention_mha_f16 : flash_attention_mha_bf16;
      if (is_decode) {
        split = is_fp16 ? bm1684x_mha_fp16_decode : bm1684x_mha_bf16_decode;
      } else {
        split = is_fp16 ? bm1684x_mha_fp16_encode : bm1684x_mha_bf16_encode;
      }
    } else {
      func = is_fp16 ? flash_attention_gqa_f16 : flash_attention_gqa_bf16;
      if (is_decode) {
        split = is_fp16 ? bm1684x_gqa_fp16_decode : bm1684x_gqa_bf16_decode;
      } else {
        split = is_fp16 ? bm1684x_gqa_fp16_encode : bm1684x_gqa_bf16_encode;
      }
    }
  }

  block_m = split[0];
  block_k = split[1];
  block_h = split[2];

  while (block_m > 0 && block_k > 0) {
    ret = func(chip, cmdid, out_spec->addr, q_spec->addr, k_spec->addr,
               v_spec->addr, _param->common.hasmask ? mask_spec->addr : 0,
               _param->common.batch, _param->common.mq, _param->common.mk,
               _param->common.dim, _param->common.q_head,
               _param->common.kv_head, _param->common.scale,
               _param->common.hasmask, dmax, block_m, block_k, block_h);
    CHECK_PPL_RET(ret);
    if (ret == PplLocalAddrAssignErr) {
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
