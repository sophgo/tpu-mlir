//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
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
  int num_input = inputs().size();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  concat_global_spec_t spec = {0};
  spec.common.input_num = num_input;
  spec.common.concat_axis = axis();
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

void tpu::ConcatOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  auto in0_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  Module::getNCHW(output(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  int num_input = inputs().size();
  SmallVector<int> is_st_concat_way(num_input, 0);
  concat_local_spec_t spec = {0};
  spec.is_st_concat_way = is_st_concat_way.data();
  spec.common.input_num = num_input;
  spec.common.concat_axis = axis();
  local_sec_info_t sec_info{0};
  sec_info.n_slice = gi.n_slice;
  sec_info.h_slice = in0_gi.h_slice;
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info.h_idx = in0_gi.h_idx;

  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.is_w_split = false;
  sec_info.out_w_slice = w;
  BM168x::call_local_func("backend_api_concat_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}
