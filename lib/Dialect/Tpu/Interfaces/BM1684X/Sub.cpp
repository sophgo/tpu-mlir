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
void tpu::SubOp::codegen_global_bm1684x() {
  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);

  if (module::isUniformQuantized(getInputs()[0], getOutput())) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_SUB;
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

int64_t tpu::SubOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice,
                                          group_type_t group_type) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::SubOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                       group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
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
  param.spec.common.binary_type = BINARY_SUB;
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
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::SubOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SubOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }
