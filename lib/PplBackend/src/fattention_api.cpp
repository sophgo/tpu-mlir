//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "fattention_prefill.h"
#include "fattention_v1.h"
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
// fattention v1/v2 tiling
static int align_2n(int x, int limit = 512) {
  int p = 1;
  if (x >= limit) {
    return limit;
  }
  while (p * 2 <= x) {
    p *= 2;
  }
  return p;
}

void fattention_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                       gaddr_t ptr_v, gaddr_t ptr_mask, int b, int qm, int kvm,
                       int d, int q_head, int kv_head, float sqrt_d,
                       int has_mask, int core_num, int dtype,
                       bool high_precision, int &block_m, int &block_k,
                       int &block_qh, int &block_kh) {
  int ret = 0;
  int keep_dim = 0;
  bool is_mha = q_head == kv_head;
  bool is_decode = qm == 1;
  bool is_fp16 = dtype == DTYPE_FP16;
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);
  auto func = high_precision
                  ? (is_fp16 ? fattention_v2_f16 : fattention_v2_bf16)
                  : (is_fp16 ? fattention_v1_f16 : fattention_v1_bf16);
  int safe_core_num = std::max(1, core_num);
  int head_rep = std::max(1, q_head / kv_head);
  if (is_decode) {
    block_m = 1;
    // For decode we want at least 2 iterations of the `_k` loop so that
    // `ppl::enable_pipeline()` can overlap the K/V loads of iter N+1 with
    // the QK/PV matmul + softmax compute of iter N. Capping the initial
    // `block_k` to roughly half of `kvm` (still aligned to a power of two
    // and >= npu_num for healthy NPU utilization) forces multiple iterations
    // while keeping each tile large enough for good DMA bandwidth.
    int half = kvm / 2;
    if (half < npu_num) {
      half = npu_num;
    }
    block_k = align_2n(half, 2048);
  } else {
    int val = std::min(qm, kvm);
    block_m = align_2n(val);
    block_k = block_m;
  }
  block_kh = kv_head / safe_core_num;
  if (block_kh == 0) {
    block_kh = 1;
  }
  block_qh = block_kh * head_rep;
  while (block_m > 0 && block_k > 0) {
    printf("fattention block_m:%d, block_k:%d, block_qh:%d\n", block_m, block_k,
           block_qh);
    ret = func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm, q_head,
               kv_head, sqrt_d, has_mask, core_num, d, keep_dim, block_m,
               block_k, block_qh, block_kh);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      if (is_decode) {
        // For decode (block_m==1) prefer shrinking block_k before block_kh
        // so that head-level parallelism is preserved as long as possible.
        if (block_k > npu_num) {
          block_k /= 2;
        } else if (block_kh > 1) {
          block_kh /= 2;
          block_qh = block_kh * head_rep;
        } else {
          break;
        }
      } else {
        if (block_kh > 1) {
          block_kh /= 2;
          block_qh = block_kh * head_rep;
        } else if (block_m > npu_num) {
          block_m /= 2;
          block_k /= 2;
        } else if (block_k > npu_num) {
          block_k /= 2;
        } else {
          break;
        }
      }
      continue;
    }
    break;
  }
  if (ret != 0) {
    printf("Error: block split failed!!!\n");
    exit(-1);
  }
  printf("fattention success!!\n");
}

// fattention_prefill tiling
void fattention_prefill_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                               gaddr_t ptr_v, gaddr_t ptr_mask, int b, int qm,
                               int kvm, int d, int q_head, int kv_head,
                               int mask_size, float sqrt_d, int core_num,
                               int dtype, int &block_m, int &block_h) {
  int ret = 0;
  int keep_dim = 0;
  auto func =
      dtype == DTYPE_FP16 ? fattention_prefill_f16 : fattention_prefill_bf16;
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);

  // `block_h` is the number of KV heads per block; the kernel internally
  // expands it to `block_h * head_rep` Q heads. Start from the full per-core
  // KV-head footprint and shrink by halving on LMEM/L2 pressure.
  int head_rep = q_head / kv_head;
  if (head_rep < 1) {
    head_rep = 1;
  }
  int kv_head_per_core = (kv_head + core_num - 1) / core_num;

  // NOTE: starting `block_m` at `mask_size/2` (rather than `mask_size`).
  // With block_m == mask_size and block_k_iter == block_m, on bm1684x the
  // local tensors live on C = block_m, which on a 64-NPU chip means the
  // 256-wide C dim is sharded as 4 rows per NPU; in that regime several
  // ops (matmul / pooling / fadd of the fp32 mask) silently produce
  // garbage for the upper EU rows (qi in [block_m/2, block_m)). Halving
  // `block_m` keeps C <= 2 * NPU_NUM and matches the working v2 prefill
  // tiling (see fattention_tile.cpp). The tiling fallback below can still
  // shrink `block_m` further on LMEM pressure.
  block_m = mask_size / 2;
  block_h = kv_head_per_core;

  while (block_m >= npu_num && block_h >= 1) {
    printf("fattention prefill block_m:%d, block_h:%d\n", block_m, block_h);
    ret = func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm, sqrt_d,
               keep_dim, core_num, q_head, kv_head, d, block_m, block_h,
               mask_size);
    if (ret == 0) {
      break;
    }
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      if (block_h > 1) {
        block_h = block_h / 2;
        if (block_h < 1) {
          block_h = 1;
        }
      } else {
        block_m = block_m / 2;
      }
      continue;
    }
    break;
  }
  if (ret != 0) {
    printf("Error: fattention_prefill split failed!!!\n");
    exit(-1);
  }
}

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
  int block_m, block_k, block_qh, block_kh;
  // The mask-free prefill kernel synthesises the causal mask in-kernel and
  // does NOT consume an external mask tensor. If the user actually supplied a
  // mask we must honour it -- fall through to the v2 path which adds the
  // user-provided mask elementwise (matches the CPU reference behaviour where
  // `has_mask` trumps `attn_type`).
  if (_param->common.mask_size == 0) {
    fattention_tiling(
        out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
        _param->common.hasmask ? mask_spec->addr : 0, _param->common.batch,
        _param->common.mq, _param->common.mk, _param->common.dim,
        _param->common.q_head, _param->common.kv_head, _param->common.scale,
        _param->common.hasmask, core_num, in_spec[0].dtype,
        _param->common.high_precision, block_m, block_k, block_qh, block_kh);
  } else {
    fattention_prefill_tiling(
        out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
        mask_spec->addr, _param->common.batch, _param->common.mq,
        _param->common.mk, _param->common.dim, _param->common.q_head,
        _param->common.kv_head, _param->common.mask_size, _param->common.scale,
        core_num, in_spec[0].dtype, block_m, block_kh);
  }
}

// dynamic interface
using DYN_ATTENTION = std::function<int(
    unsigned long long v1, unsigned long long v2, unsigned long long v3,
    unsigned long long v4, unsigned long long v5, int32_t v6, int32_t v7,
    int32_t v8, int32_t v9, int32_t v10, float v11, int32_t v12, int32_t v13,
    int32_t v14, int32_t v15, int32_t v16, int32_t v17, int32_t v18,
    int32_t v19, void *buffer)>;
// fill_${OP_NAME}_struct gen automatic by ppl, the differ between ppl kernel
// func are with extra buffer param and return type
static DYN_ATTENTION get_dyn_attention_func(bool is_fp16, bool high_precision) {
  if (is_fp16) {
    return high_precision ? fill_fattention_v2_f16_struct
                          : fill_fattention_v1_f16_struct;
  } else {
    return high_precision ? fill_fattention_v2_bf16_struct
                          : fill_fattention_v1_bf16_struct;
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
  const int core_num = get_core_num();
  if (_param->common.mask_size == 0 || false == _param->common.hasmask) {
    auto q_spec = in_spec;
    auto k_spec = in_spec + 1;
    auto v_spec = in_spec + 2;
    auto mask_spec = in_spec + 3;
    auto dtype = in_spec[0].dtype;
    auto q_head = _param->common.q_head;
    auto kv_head = _param->common.kv_head;
    auto high_precision = _param->common.high_precision;
    int keep_dim = _param->common.keep_dim ? 1 : 0;
    int block_m, block_k, block_qh, block_kh;
    if (buffer) {
      // get tile info
      fattention_tiling(
          out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
          _param->common.hasmask ? mask_spec->addr : 0, _param->common.batch,
          _param->common.mq, _param->common.mk, _param->common.dim, q_head,
          kv_head, _param->common.scale, _param->common.hasmask, core_num,
          dtype, high_precision, block_m, block_k, block_qh, block_kh);
    }
    // If buffer is not null writre param info to buffer according to tile info,
    // return param struct lens.
    DYN_ATTENTION func =
        get_dyn_attention_func(dtype == DTYPE_FP16, high_precision);
    return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                _param->common.hasmask ? mask_spec->addr : 0,
                _param->common.batch, _param->common.mq, _param->common.mk,
                q_head, kv_head, _param->common.scale, _param->common.hasmask,
                core_num, _param->common.dim, keep_dim, block_m, block_k,
                block_qh, block_kh, buffer);
  } else {
    auto q_spec = in_spec;
    auto k_spec = in_spec + 1;
    auto v_spec = in_spec + 2;
    auto mask_spec = in_spec + 3;
    auto dtype = in_spec[0].dtype;
    auto q_head = _param->common.q_head;
    auto kv_head = _param->common.kv_head;
    int keep_dim = _param->common.keep_dim ? 1 : 0;
    int block_m, block_h;
    if (buffer) {
      fattention_prefill_tiling(
          out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
          mask_spec->addr, _param->common.batch, _param->common.mq,
          _param->common.mk, _param->common.dim, q_head, kv_head,
          _param->common.mask_size, _param->common.scale, core_num, dtype,
          block_m, block_h);
    }
    // If buffer is not null writre param info to buffer according to tile info,
    // return param struct lens.
    auto func = dtype == DTYPE_FP16 ? fill_fattention_prefill_f16_struct
                                    : fill_fattention_prefill_bf16_struct;
    return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                mask_spec->addr, _param->common.batch, _param->common.mq,
                _param->common.mk, _param->common.scale, keep_dim, core_num,
                q_head, kv_head, _param->common.dim, block_m, block_h,
                _param->common.mask_size, buffer);
  }
}

#ifdef __cplusplus
}
#endif
