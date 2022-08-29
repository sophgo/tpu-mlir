//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::SiLUOp::codegen_global_bm1684x() {
  active_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.common.active_type = ACTIVE_SILU;
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  BM1684x::instance().call_global_func("backend_api_active_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SiLUOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  auto stype = Module::getStorageType(input());
  // |    work1    |    work0    | exp coeff  | exp_table |
  // | tensor_size | tensor_size |     32     |    192    |
  int64_t bytes = in_lmem_bytes / in_nslice;
  int64_t buffer_size = 2 * align_up(bytes, 64l);
  int64_t dtype_len = stype.getIntOrFloatBitWidth() / 8;
  buffer_size += align_up(32 * dtype_len, 64l) + align_up(192 * dtype_len, 64l);
  return buffer_size;
}

void tpu::SiLUOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);

  active_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.common.active_type = ACTIVE_SILU;
  spec.buffer_addr = gi.buffer_addr;

  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = 1;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = w;
  BM1684x::instance().call_local_func("backend_api_active_local", &spec,
                                      sizeof(spec), &sec_info,
                                      input_spec->data(), output_spec->data());
}
