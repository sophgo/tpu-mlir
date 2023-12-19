//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::ModOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  bcbinary_common_spec_t spec;
  memset(&spec, 0, sizeof(bcbinary_common_spec_t));
  spec.binary_type = BINARY_FLOOR_MOD;
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.rshift_A = 0;
  spec.rshift_B = 0;
  BM168x::call_global_func("backend_api_eltbinary_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}
// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::ModOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::ModOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_FLOOR_MOD;
  param.spec.common.rshift_A = 0;
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;

  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ModOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_common_spec_t);
  bcbinary_common_spec_t spec = {0};
  spec.binary_type = BINARY_FLOOR_MOD;
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.rshift_A = 0;
  spec.rshift_B = 0;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ModOp::get_fw_type_bm1684x() { return FW_BMNET_ELTWISE_BINARY; }
