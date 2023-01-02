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

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::AddOp::codegen_global_bm1684x() {
  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);

  if (module::isUniformQuantized(getInputs()[0], getOutput())) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_ADD;
  param.if_relu = getDoRelu();
  param.relu_upper_limit = getReluLimit().convertToDouble();
  param.rshift_A = rshift_v[0];
  param.rshift_B = rshift_v[1];
  param.scale_A = multi_v[0];
  param.scale_B = multi_v[1];
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AddOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::AddOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                 void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(getInputs()[1], n_step, h_step);
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

void tpu::AddOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                       void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);

  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);
  if (module::isUniformQuantized(getInputs()[0], getOutput())) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_ADD;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = rshift_v[0];
  param.spec.common.rshift_B = rshift_v[1];
  param.spec.common.scale_A = multi_v[0];
  param.spec.common.scale_B = multi_v[1];
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;

  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          sec_info_, input_spec->data(), output_spec->data());
}
