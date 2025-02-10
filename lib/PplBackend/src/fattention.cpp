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
typedef struct {
  int block[3]; // block_m,block_k,block_h
} block_t;

static block_t v1_configs[2][2][2][2] = {
    {{
         // bm1688 MHA
         {{32, 320, 8}, {192, 128, 8}}, // bf16
         {{32, 512, 8}, {256, 256, 8}}  // fp16
     },
     {
         // bm1688 GQA
         {{32, 160, 16}, {96, 64, 16}}, // bf16
         {{32, 256, 16}, {160, 96, 16}} // fp16
     }},
    {{
         // bm1684x MHA
         {{64, 1024, 8}, {384, 384, 8}}, // bf16
         {{64, 1024, 8}, {384, 384, 8}}  // fp16
     },
     {
         // bm1684x GQA
         {{64, 512, 16}, {352, 128, 16}}, // bf16
         {{64, 896, 16}, {448, 192, 16}}  // fp16
     }}};

// v2
static block_t v2_configs[2][2][2][2] = {
    {{
         // bm1688 MHA
         {{32, 320, 8}, {192, 256, 8}}, // bf16
         {{32, 320, 8}, {192, 256, 8}}  // fp16
     },
     {
         // bm1688 GQA
         {{32, 256, 16}, {96, 64, 16}}, // bf16
         {{32, 256, 16}, {96, 64, 16}}  // fp16
     }},
    {{
         // bm1684x MHA
         {{64, 384, 16}, {256, 320, 8}}, // bf16
         {{64, 384, 16}, {256, 320, 8}}  // fp16
     },
     {
         // bm1684x GQA
         {{64, 512, 16}, {256, 128, 16}}, // bf16
         {{64, 512, 16}, {256, 128, 16}}  // fp16
     }}};

using ATTENTION = std::function<int(
    const char *chip, void *pid_node, unsigned long long v1,
    unsigned long long v2, unsigned long long v3, unsigned long long v4,
    unsigned long long v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9,
    int32_t v10, int32_t v11, float v12, int32_t v13, int32_t v14, int32_t v15,
    int32_t v16, int32_t v17, int32_t v18)>;

static ATTENTION get_attention_func(bool is_fp16, bool is_mha,
                                    bool high_precision) {
  if (is_mha && is_fp16) {
    return high_precision ? flash_attention_mha_f16_high_precision
                          : flash_attention_mha_f16;
  }
  if (is_mha && !is_fp16) {
    return high_precision ? flash_attention_mha_bf16_high_precision
                          : flash_attention_mha_bf16;
  }
  if (!is_mha && is_fp16) {
    return high_precision ? flash_attention_gqa_f16_high_precision
                          : flash_attention_gqa_f16;
  }
  if (!is_mha && !is_fp16) {
    return high_precision ? flash_attention_gqa_bf16_high_precision
                          : flash_attention_gqa_bf16;
  }
  // never go here
  return nullptr;
}

static int *get_block_split(bool is_fp16, bool is_mha, bool is_decode,
                            bool is_v2, const std::string &chip) {
  int chip_idx = (chip == PPL_BM1688) ? 0 : 1; // 0 for bm1688, 1 for bm1684x
  int mha_idx = is_mha ? 0 : 1;                // 0 for MHA, 1 for GQA
  int f16_idx = is_fp16 ? 1 : 0;               // 0 for bf16, 1 for fp16
  int dc_idx = is_decode ? 0 : 1;              // 0 for decode, 1 for encode
  return is_v2 ? v2_configs[chip_idx][mha_idx][f16_idx][dc_idx].block
               : v1_configs[chip_idx][mha_idx][f16_idx][dc_idx].block;
}

void api_fattention_global(void *param, size_t param_size, void *input_spec,
                           void *output_spec, const int core_num,
                           const char *chip, void *cmdid) {
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
  bool is_v2 = _param->common.high_precision;
  int *split = get_block_split(is_fp16, is_mha, is_decode, is_v2, chip_str);
  ATTENTION func = get_attention_func(is_fp16, is_mha, is_v2);
  block_m = split[0];
  block_k = split[1];
  block_h = split[2];

  while (block_m > 0 && block_k > 0) {
    printf("fattention block_m:%d, block_k:%d, block_h:%d\n", block_m, block_k,
           block_h);
    ret = func(
        chip, cmdid, out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
        _param->common.hasmask ? mask_spec->addr : 0, _param->common.batch,
        _param->common.mq, _param->common.mk, _param->common.dim,
        _param->common.q_head, _param->common.kv_head, _param->common.scale,
        _param->common.hasmask, core_num, dmax, block_m, block_k, block_h);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr ||
        ret == -1) {
      printf("block is not suitable, have another try !!!\n");
      if (!is_decode) {
        block_m -= 16;
      }
      block_k -= 16;
      continue;
    }
    break;
  }
  if (block_m <= 0 || block_k <= 0) {
    printf("Error: block split failed!!!\n");
    exit(-1);
  }
  printf("fattention success!!\n");
}

#ifdef __cplusplus
}
#endif
