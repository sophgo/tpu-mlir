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

void tpu::TriluOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  triangularize_common_spec_t spec = {0};
  spec.is_upper = getUpper();
  spec.diagonal = getDiagonal();
  BM168x::call_global_func("backend_api_triangularize_global", &spec,
                           sizeof(triangularize_common_spec_t),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::TriluOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {

  int64_t idx_dtype_size;
  if (module::isMARS3() || module::isSGTPUV8()) {
    idx_dtype_size = sizeof(int16_t);
  } else {
    idx_dtype_size = sizeof(int32_t);
  }

  int64_t max_H_W = in_hslice > in_wslice ? in_hslice : in_wslice;
  return in_nslice * in_cslice * max_H_W * (max_H_W + 1) * idx_dtype_size;
}

void tpu::TriluOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  triangularize_common_spec_t spec = {0};
  spec.is_upper = getUpper();
  spec.diagonal = getDiagonal();
  BM168x::call_local_func("backend_api_triangularize_local", &spec,
                          sizeof(spec), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::TriluOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(triangularize_common_spec_t);
  triangularize_common_spec_t spec = {0};
  spec.is_upper = getUpper();
  spec.diagonal = getDiagonal();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TriluOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(triangularize_common_spec_t);
  triangularize_common_spec_t spec = {0};
  spec.is_upper = getUpper();
  spec.diagonal = getDiagonal();
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::TriluOp::get_fw_type_bm1684x() { return FW_BMNET_TRIANGULARIZE; }
