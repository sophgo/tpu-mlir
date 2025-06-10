#include "fattention_bf16.h"
#include "fattention_fp16.h"
#include "fattention_v2.h"
#include "helper.h"
#include "tpu_mlir/Backend/BM168x/Param.h"

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
    unsigned long long v1, unsigned long long v2, unsigned long long v3,
    unsigned long long v4, unsigned long long v5, int32_t v6, int32_t v7,
    int32_t v8, int32_t v9, int32_t v10, int32_t v11, float v12, int32_t v13,
    int32_t v14, int32_t v15, int32_t v16, int32_t v17, int32_t v18)>;

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

int fattention_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                      gaddr_t ptr_v, gaddr_t ptr_mask, int b, int qm, int kvm,
                      int d, int q_head, int kv_head, float sqrt_d,
                      int has_mask, int core_num, int dtype,
                      bool high_precision) {
  int ret = 0;
  int dmax = align_up(d, 32 /*eu num*/);
  int block_m, block_k, block_h;
  std::string chip_str = get_chip_str();
  bool is_mha = q_head == kv_head;
  bool is_decode = qm == 1;
  bool is_fp16 = dtype == DTYPE_FP16;
  bool is_v2 = high_precision;
  int *split = get_block_split(is_fp16, is_mha, is_decode, is_v2, chip_str);
  ATTENTION func = get_attention_func(is_fp16, is_mha, is_v2);
  block_m = split[0];
  block_k = split[1];
  block_h = split[2];
  if (is_mha) {
    int safe_core_num = std::max(1, core_num);
    int max_block_h = (q_head + safe_core_num - 1) / safe_core_num;
    block_h = std::min(block_h, max_block_h);
  }

  while (block_m > 0 && block_k > 0) {
    printf("fattention block_m:%d, block_k:%d, block_h:%d\n", block_m, block_k,
           block_h);
    ret = func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm, d, q_head,
               kv_head, sqrt_d, has_mask, core_num, dmax, block_m, block_k,
               block_h);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
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
    return ret;
  }
  printf("fattention success!!\n");
  return ret;
}

#ifdef __cplusplus
}
#endif
