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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================

void tpu::CompareOp::codegen_global_bm1684x() {
  bcbinary_common_spec_t spec = {0};
  spec.binary_type = BM168x::compare_mode(getMode());
  spec.if_relu = 0;
  spec.scale_A = 1;
  spec.scale_B = 1;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CompareOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  return 0;
}

void tpu::CompareOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  bcbinary_local_spec_t spec = {0};
  spec.common.binary_type = BM168x::compare_mode(getMode());
  spec.common.if_relu = 0;
  spec.common.scale_A = 1;
  spec.common.scale_B = 1;

  BM168x::call_local_func("backend_api_bcbinary_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::CompareOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(bcbinary_local_param_t);
  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BM168x::compare_mode(getMode());
  param.spec.common.if_relu = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CompareOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(bcbinary_common_spec_t);
  bcbinary_common_spec_t spec = {0};
  spec.binary_type = BM168x::compare_mode(getMode());
  spec.if_relu = 0;
  spec.scale_A = 1;
  spec.scale_B = 1;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::CompareOp::get_fw_type_bm1684x() {
  return FW_BMNET_BROADCAST_BINARY;
}
