//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "fattention_flex.h"
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
    // On chips with fewer NPUs (e.g., BM1688), a 512-wide tile can
    // overflow local memory and cause silent data corruption in the
    // fattention v1/v2 kernels.  Cap block_m / block_k at 256 on such
    // chips to match the same mitigation used in the prefill path
    // (see fattention_prefill_tiling block_m = 2 * npu_num).
    int tiling_limit = (npu_num <= 32) ? 256 : 512;
    block_m = align_2n(val, tiling_limit);
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

// LSE-emitting tiling: same tiling logic as fattention_tiling but always uses
// the v2 _lse kernels (which additionally write fp32 logsumexp to ptr_lse).
// ptr_lse goes right after ptr_mask to match the _lse kernel signatures.
void fattention_tiling_lse(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                           gaddr_t ptr_v, gaddr_t ptr_mask, gaddr_t ptr_lse,
                           int b, int qm, int kvm, int d, int q_head,
                           int kv_head, float sqrt_d, int has_mask,
                           int core_num, int dtype, int &block_m, int &block_k,
                           int &block_qh, int &block_kh) {
  int ret = 0;
  int keep_dim = 0;
  bool is_fp16 = dtype == DTYPE_FP16;
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);
  auto func = is_fp16 ? fattention_v2_f16_lse : fattention_v2_bf16_lse;
  int safe_core_num = std::max(1, core_num);
  int head_rep = std::max(1, q_head / kv_head);
  bool is_decode = qm == 1;
  if (is_decode) {
    block_m = 1;
    int half = kvm / 2;
    if (half < npu_num) {
      half = npu_num;
    }
    block_k = align_2n(half, 2048);
  } else {
    int val = std::min(qm, kvm);
    int tiling_limit = (npu_num <= 32) ? 256 : 512;
    block_m = align_2n(val, tiling_limit);
    block_k = block_m;
  }
  block_kh = kv_head / safe_core_num;
  if (block_kh == 0) {
    block_kh = 1;
  }
  block_qh = block_kh * head_rep;
  while (block_m > 0 && block_k > 0) {
    printf("fattention_lse block_m:%d, block_k:%d, block_qh:%d, block_kh:%d\n",
           block_m, block_k, block_qh, block_kh);
    ret = func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, ptr_lse, b, qm, kvm,
               q_head, kv_head, sqrt_d, has_mask, core_num, d, keep_dim,
               block_m, block_k, block_qh, block_kh);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      if (is_decode) {
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
  printf("fattention_lse success!!\n");
}

// fattention_prefill tiling
void fattention_prefill_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                               gaddr_t ptr_v, gaddr_t ptr_mask, int b, int qm,
                               int kvm, int d, int q_head, int kv_head,
                               int mask_size, float sqrt_d, int core_num,
                               int dtype, int &block_m, int &block_qh,
                               int &block_kh) {
  int ret = 0;
  int keep_dim = 0;
  auto func =
      dtype == DTYPE_FP16 ? fattention_prefill_f16 : fattention_prefill_bf16;
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);

  block_m = mask_size / 2;
  assert(block_m >= npu_num);
  int head_rep = std::max(1, q_head / kv_head);
  // Start with all heads in one block (block_qh == q_head, i.e. no head
  // slicing) so the common case keeps the contiguous store_transpose_nc
  // destination. Only when that overflows local/L2 memory do we halve the head
  // block -- the same block_qh/block_kh fallback fattention_tiling uses for
  // fattention_v1/v2. block_m is fixed by mask_size here, so head slicing is
  // the only available knob.
  block_kh = kv_head;
  block_qh = block_kh * head_rep;
  while (block_m >= npu_num && block_kh > 0) {
    printf("fattention_prefill block_m:%d, block_qh:%d, block_kh:%d\n", block_m,
           block_qh, block_kh);
    ret = func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm, sqrt_d,
               keep_dim, core_num, q_head, kv_head, d, block_m, block_qh,
               block_kh, mask_size);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      if (block_m > npu_num) {
        block_m /= 2;
      } else {
        block_kh /= 2;
        block_qh = block_kh * head_rep;
      }
      continue;
    }
    break;
  }
  if (ret != 0) {
    printf("Error: fattention_prefill split failed!!!\n");
    exit(-1);
  }
  printf("fattention_prefill success!!\n");
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
        core_num, in_spec[0].dtype, block_m, block_qh, block_kh);
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
    int block_m, block_qh, block_kh;
    if (buffer) {
      fattention_prefill_tiling(
          out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
          mask_spec->addr, _param->common.batch, _param->common.mq,
          _param->common.mk, _param->common.dim, q_head, kv_head,
          _param->common.mask_size, _param->common.scale, core_num, dtype,
          block_m, block_qh, block_kh);
    }
    // If buffer is not null writre param info to buffer according to tile info,
    // return param struct lens.
    auto func = dtype == DTYPE_FP16 ? fill_fattention_prefill_f16_struct
                                    : fill_fattention_prefill_bf16_struct;
    return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
                mask_spec->addr, _param->common.batch, _param->common.mq,
                _param->common.mk, _param->common.scale, keep_dim, core_num,
                q_head, kv_head, _param->common.dim, block_m, block_qh,
                block_kh, _param->common.mask_size, buffer);
  }
}

// LSE-emitting global entry. Identical to api_fattention_global except:
//  - output_spec has 2 entries: [0]=attention output, [1]=fp32 lse
//  [b,mq,q_head,1]
//  - dispatches to fattention_tiling_lse (v2 _lse kernels).
//  - only the full-mask (mask_size==0) v2 path is supported (prefill kernel has
//    no _lse variant); FAttentionLseOp is expected to use a full runtime mask.
void api_fattention_lse_global(void *param, size_t param_size, void *input_spec,
                               void *output_spec) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto q_spec = in_spec;
  auto k_spec = in_spec + 1;
  auto v_spec = in_spec + 2;
  auto mask_spec = in_spec + 3;
  auto lse_spec = out_spec + 1;
  const int core_num = get_core_num();
  int block_m, block_k, block_qh, block_kh;
  // _lse kernels only exist for the v2 (full-mask) path.
  assert(_param->common.mask_size == 0);
  fattention_tiling_lse(
      out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
      _param->common.hasmask ? mask_spec->addr : 0, lse_spec->addr,
      _param->common.batch, _param->common.mq, _param->common.mk,
      _param->common.dim, _param->common.q_head, _param->common.kv_head,
      _param->common.scale, _param->common.hasmask, core_num, in_spec[0].dtype,
      block_m, block_k, block_qh, block_kh);
}

// dynamic LSE interface. The fill_*_lse_struct helpers are auto-generated by
// ppl to match the _lse kernel signatures (ptr_lse after ptr_mask).
using DYN_ATTENTION_LSE = std::function<int(
    unsigned long long v1, unsigned long long v2, unsigned long long v3,
    unsigned long long v4, unsigned long long v5, unsigned long long v6,
    int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, float v12,
    int32_t v13, int32_t v14, int32_t v15, int32_t v16, int32_t v17,
    int32_t v18, int32_t v19, int32_t v20, void *buffer)>;
static DYN_ATTENTION_LSE get_dyn_attention_lse_func(bool is_fp16) {
  return is_fp16 ? fill_fattention_v2_f16_lse_struct
                 : fill_fattention_v2_bf16_lse_struct;
}
int api_dyn_fattention_lse_global(void *param, void *input_spec,
                                  void *output_spec, void *buffer) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  const int core_num = get_core_num();
  // _lse only supports the full-mask v2 path.
  assert(_param->common.mask_size == 0 || false == _param->common.hasmask);
  auto q_spec = in_spec;
  auto k_spec = in_spec + 1;
  auto v_spec = in_spec + 2;
  auto mask_spec = in_spec + 3;
  auto lse_spec = out_spec + 1;
  auto dtype = in_spec[0].dtype;
  auto q_head = _param->common.q_head;
  auto kv_head = _param->common.kv_head;
  int keep_dim = _param->common.keep_dim ? 1 : 0;
  int block_m, block_k, block_qh, block_kh;
  if (buffer) {
    fattention_tiling_lse(
        out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
        _param->common.hasmask ? mask_spec->addr : 0, lse_spec->addr,
        _param->common.batch, _param->common.mq, _param->common.mk,
        _param->common.dim, q_head, kv_head, _param->common.scale,
        _param->common.hasmask, core_num, dtype, block_m, block_k, block_qh,
        block_kh);
  }
  DYN_ATTENTION_LSE func = get_dyn_attention_lse_func(dtype == DTYPE_FP16);
  return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
              _param->common.hasmask ? mask_spec->addr : 0, lse_spec->addr,
              _param->common.batch, _param->common.mq, _param->common.mk,
              q_head, kv_head, _param->common.scale, _param->common.hasmask,
              core_num, _param->common.dim, keep_dim, block_m, block_k,
              block_qh, block_kh, buffer);
}

// FlexAttention tiling: same structure as fattention_tiling_lse but dispatches
// the v2 _flex kernels (block-sparse bitmap guard + qk_d/v_d split + optional
// lse). When has_bitmap, block_m/block_k are capped at flex_block so each tile
// maps to exactly one bitmap cell (flex_block % block_m == 0). The kernel
// derives num_q_blocks/num_kv_blocks internally (kept out of the signature to
// stay under the PPL kernel param-count limit).
void fattention_flex_tiling(gaddr_t ptr_dst, gaddr_t ptr_q, gaddr_t ptr_k,
                            gaddr_t ptr_v, gaddr_t ptr_mask, gaddr_t ptr_lse,
                            gaddr_t ptr_bitmap, int b, int qm, int kvm,
                            int qk_d, int v_d, int q_head, int kv_head,
                            float sqrt_d, int has_mask, int has_lse,
                            int has_bitmap, int flex_block, int core_num,
                            int dtype, int &block_m, int &block_k,
                            int &block_qh, int &block_kh) {
  int ret = 0;
  bool is_fp16 = dtype == DTYPE_FP16;
  int npu_num, npu_size;
  get_chip_info(&npu_num, &npu_size);
  auto func = is_fp16 ? fattention_v2_f16_flex : fattention_v2_bf16_flex;
  int safe_core_num = std::max(1, core_num);
  int head_rep = std::max(1, q_head / kv_head);
  bool is_decode = qm == 1;
  if (is_decode) {
    block_m = 1;
    int half = kvm / 2;
    if (half < npu_num) {
      half = npu_num;
    }
    block_k = align_2n(half, 2048);
  } else {
    int val = std::min(qm, kvm);
    int tiling_limit = (npu_num <= 32) ? 256 : 512;
    // block-sparse: tile must be <= flex_block so each tile maps to exactly
    // one bitmap cell (flex_block % block_m == 0). Halving below stays <=.
    if (has_bitmap && tiling_limit > flex_block) {
      tiling_limit = flex_block;
    }
    block_m = align_2n(val, tiling_limit);
    block_k = block_m;
  }
  block_kh = kv_head / safe_core_num;
  if (block_kh == 0) {
    block_kh = 1;
  }
  block_qh = block_kh * head_rep;
  // The PPL compile-time local-addr assigner now runs correctly (kernel shape
  // params are `const int`), so the retry loop below handles local-mem overflow
  // by halving block_m/block_kh and retrying until it fits. No upfront cap.
  while (block_m > 0 && block_k > 0) {
    printf("fattention_flex block_m:%d, block_k:%d, block_qh:%d, block_kh:%d\n",
           block_m, block_k, block_qh, block_kh);
    ret =
        func(ptr_dst, ptr_q, ptr_k, ptr_v, ptr_mask, ptr_lse, b, qm, kvm,
             q_head, kv_head, sqrt_d, has_mask, has_lse, has_bitmap, flex_block,
             core_num, qk_d, v_d, block_m, block_k, block_qh, block_kh);
    CHECK_PPL_RET(ret);
    if (ret == PplL2AddrAssignErr || ret == PplLocalAddrAssignErr) {
      printf("block is not suitable, have another try !!!\n");
      if (is_decode) {
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
    printf("Error: fattention_flex split failed!!!\n");
    exit(-1);
  }
  printf("fattention_flex success!!\n");
}

// FlexAttention global entry. get_input_spec skips None operands, so input
// indices are walked dynamically: q,k,v are always present; mask only when
// hasmask; block_bitmap only when has_bitmap; buffer is None (skipped).
// Outputs: [0]attention output, [1]fp32 lse (only meaningful when has_lse).
void api_fattention_flex_global(void *param, size_t param_size,
                                void *input_spec, void *output_spec) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  int idx = 0;
  tensor_spec_t *q_spec = in_spec + idx++;
  tensor_spec_t *k_spec = in_spec + idx++;
  tensor_spec_t *v_spec = in_spec + idx++;
  tensor_spec_t *mask_spec = _param->common.hasmask ? in_spec + idx++ : nullptr;
  tensor_spec_t *bitmap_spec =
      _param->common.has_bitmap ? in_spec + idx++ : nullptr;
  tensor_spec_t *lse_spec = out_spec + 1;
  gaddr_t mask_addr = mask_spec ? mask_spec->addr : 0;
  gaddr_t bitmap_addr = bitmap_spec ? bitmap_spec->addr : 0;
  gaddr_t lse_addr = _param->common.has_lse ? lse_spec->addr : 0;
  const int core_num = get_core_num();
  int qk_d = _param->common.qk_d;
  int v_d = _param->common.v_d;
  int flex_block = _param->common.flex_block ? _param->common.flex_block : 128;
  int block_m, block_k, block_qh, block_kh;
  fattention_flex_tiling(
      out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr, mask_addr,
      lse_addr, bitmap_addr, _param->common.batch, _param->common.mq,
      _param->common.mk, qk_d, v_d, _param->common.q_head,
      _param->common.kv_head, _param->common.scale, _param->common.hasmask,
      _param->common.has_lse, _param->common.has_bitmap, flex_block, core_num,
      in_spec[0].dtype, block_m, block_k, block_qh, block_kh);
}

// dynamic FlexAttention interface. fill_*_flex_struct is auto-generated by ppl
// to match the _flex kernel signature (block_bitmap param removed: block-sparse
// deferred; signature is now out,q,k,v,mask,lse,b,qm,kvm,q_head,kv_head,sqrt_d,
// has_mask,has_lse,has_bitmap,flex_block,core,qk_d,v_d,block_m,block_k,block_qh,
// block_kh = 23 params + buffer).
using DYN_ATTENTION_FLEX = std::function<int(
    unsigned long long v1, unsigned long long v2, unsigned long long v3,
    unsigned long long v4, unsigned long long v5, unsigned long long v6,
    int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, float v12,
    int32_t v13, int32_t v14, int32_t v15, int32_t v16, int32_t v17,
    int32_t v18, int32_t v19, int32_t v20, int32_t v21, int32_t v22,
    int32_t v23, void *buffer)>;
static DYN_ATTENTION_FLEX get_dyn_attention_flex_func(bool is_fp16) {
  return is_fp16 ? fill_fattention_v2_f16_flex_struct
                 : fill_fattention_v2_bf16_flex_struct;
}
int api_dyn_fattention_flex_global(void *param, void *input_spec,
                                   void *output_spec, void *buffer) {
  flash_attention_global_spec_t *_param =
      (flash_attention_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  const int core_num = get_core_num();
  // walk input_spec dynamically (None operands skipped by get_input_spec)
  int idx = 0;
  tensor_spec_t *q_spec = in_spec + idx++;
  tensor_spec_t *k_spec = in_spec + idx++;
  tensor_spec_t *v_spec = in_spec + idx++;
  tensor_spec_t *mask_spec = _param->common.hasmask ? in_spec + idx++ : nullptr;
  tensor_spec_t *bitmap_spec =
      _param->common.has_bitmap ? in_spec + idx++ : nullptr;
  tensor_spec_t *lse_spec = out_spec + 1;
  gaddr_t mask_addr = mask_spec ? mask_spec->addr : 0;
  gaddr_t bitmap_addr = bitmap_spec ? bitmap_spec->addr : 0;
  gaddr_t lse_addr = _param->common.has_lse ? lse_spec->addr : 0;
  auto dtype = in_spec[0].dtype;
  int qk_d = _param->common.qk_d;
  int v_d = _param->common.v_d;
  int flex_block = _param->common.flex_block ? _param->common.flex_block : 128;
  int block_m, block_k, block_qh, block_kh;
  if (buffer) {
    fattention_flex_tiling(
        out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr, mask_addr,
        lse_addr, bitmap_addr, _param->common.batch, _param->common.mq,
        _param->common.mk, qk_d, v_d, _param->common.q_head,
        _param->common.kv_head, _param->common.scale, _param->common.hasmask,
        _param->common.has_lse, _param->common.has_bitmap, flex_block, core_num,
        dtype, block_m, block_k, block_qh, block_kh);
  }
  DYN_ATTENTION_FLEX func = get_dyn_attention_flex_func(dtype == DTYPE_FP16);
  return func(out_spec->addr, q_spec->addr, k_spec->addr, v_spec->addr,
              mask_addr, lse_addr, _param->common.batch, _param->common.mq,
              _param->common.mk, _param->common.q_head, _param->common.kv_head,
              _param->common.scale, _param->common.hasmask,
              _param->common.has_lse, _param->common.has_bitmap, flex_block,
              core_num, qk_d, v_d, block_m, block_k, block_qh, block_kh,
              buffer);
}

#ifdef __cplusplus
}
#endif
