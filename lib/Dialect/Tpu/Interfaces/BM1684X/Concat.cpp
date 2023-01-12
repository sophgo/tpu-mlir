//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct concat_common_spec {
  int input_num;
  int concat_axis;
} concat_common_spec_t;

typedef struct concat_global_spec {
  concat_common_spec_t common;
  int *is_st_concat_way;
} concat_global_spec_t;

typedef struct concat_local_spec {
  concat_common_spec_t common;
  int *is_st_concat_way;
} concat_local_spec_t;

typedef struct concat_local_param {
  concat_local_spec_t spec;
} concat_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ConcatOp::codegen_global_bm1684x() {
  auto op = getOperation();
  int num_input = getInputs().size();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  concat_global_spec_t spec = {0};
  spec.common.input_num = num_input;
  spec.common.concat_axis = getAxis();
  SmallVector<int> is_st_concat_way(num_input, 0);
  spec.is_st_concat_way = is_st_concat_way.data();

  BM168x::call_global_func("backend_api_concat_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ConcatOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                    void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  sec_info->n_slice = gi.n_slice;
  sec_info->h_slice = in0_gi.h_slice;
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info->h_idx = in0_gi.h_idx;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->is_w_split = false;
  sec_info->out_w_slice = w;
}

void tpu::ConcatOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                          void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  concat_local_spec_t spec = {0};
  int num_input = getInputs().size();
  SmallVector<int> is_st_concat_way(num_input, 0);
  spec.is_st_concat_way = is_st_concat_way.data();
  spec.common.input_num = num_input;
  spec.common.concat_axis = getAxis();

  BM168x::call_local_func("backend_api_concat_local", &spec, sizeof(spec),
                          sec_info_, input_spec->data(), output_spec->data());
}

//dynamic codegen
int64_t tpu::ConcatOp::dyn_codegen_local_bm1684x(void *buffer) {
  int input_num = getInputs().size();
  if (buffer) {
      concat_common_spec_t common;
      memset(&common, 0, sizeof(common));
      common.input_num = input_num;
      common.concat_axis = getAxis();
      auto p = static_cast<char *>(buffer);
      memcpy(p, &common, sizeof(common));
      p += sizeof(common);
      int size = p - static_cast<char *>(buffer);
      buffer = (char*)buffer + size;
      SmallVector<int> is_st_concat_way(input_num, 0);
      p = static_cast<char *>(buffer);
      memcpy(p,is_st_concat_way.data(), sizeof(is_st_concat_way[0]) * input_num);
      p += sizeof(is_st_concat_way[0]) * input_num;
  }
  return sizeof(concat_common_spec_t) + input_num * sizeof(int);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ConcatOp::dyn_codegen_global_bm1684x(void *buffer) {
  int input_num = getInputs().size();
  if (buffer) {
      concat_common_spec_t common;
      memset(&common, 0, sizeof(common));
      common.input_num = input_num;
      common.concat_axis = getAxis();
      auto p = static_cast<char *>(buffer);
      memcpy(p, &common, sizeof(common));
      p += sizeof(common);
      int size = p - static_cast<char *>(buffer);
      buffer = (char*)buffer + size;
      SmallVector<int> is_st_concat_way(input_num, 0);
      p = static_cast<char *>(buffer);
      memcpy(p,is_st_concat_way.data(), sizeof(is_st_concat_way[0]) * input_num);
      p += sizeof(is_st_concat_way[0]) * input_num;
  }
  return sizeof(concat_common_spec_t) + input_num * sizeof(int);
}
