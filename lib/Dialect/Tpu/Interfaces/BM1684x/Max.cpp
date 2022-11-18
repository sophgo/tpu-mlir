//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Binary_param.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
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

void tpu::MaxOp::codegen_global_bm1684x() {
  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_MAX;
  param.if_relu = 0;
  param.relu_upper_limit = -1.0f;
  param.rshift_A = 0;
  param.rshift_B = 0;
  param.scale_A = 1;
  param.scale_B = 1;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param,
                                       sizeof(param), input_spec->data(),
                                       output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MaxOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t buffer_size = 0;
  return buffer_size;
}

void tpu::MaxOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in0_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(inputs()[1], n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  uint32_t input_offset[] = {(uint32_t)in0_gi.out_addr,
                             (uint32_t)in1_gi.out_addr};
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto out_type = Module::getStorageType(output());
  auto in_type = Module::getStorageType(inputs()[0]);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
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

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_MAX;
  param.spec.common.if_relu = 0;
  param.spec.common.relu_upper_limit = -1.0f;
  param.spec.common.rshift_A = 0;
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;
  BM168x::call_local_func("backend_api_bcbinary_local", &param,
                                      sizeof(param), &sec_info,
                                      input_spec->data(), output_spec->data());
}
