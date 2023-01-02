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

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::WhereOp::codegen_global_bm1684x() {
  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = false;
  spec.sel1_is_const = false;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_select_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::WhereOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                            int64_t out_lmem_bytes,
                                            int64_t in_nslice, int64_t in_hslice,
                                            int64_t out_nslice,
                                            int64_t out_hslice) {
  return 0;
}

void tpu::WhereOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                    void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  sec_info->n_slice = in_gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = in_gi.h_slice;
  sec_info->h_idx = in_gi.h_idx;
  sec_info->is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = w;
}

void tpu::WhereOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = false;
  spec.sel1_is_const = false;

  BM168x::call_local_func("backend_api_select_local", &spec, sizeof(spec),
                          sec_info_, input_spec->data(), output_spec->data());
}
