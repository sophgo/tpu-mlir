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

void tpu::ClipOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  clip_spec_t spec = {0};
  spec.min = static_cast<float>(getMin().convertToDouble());
  spec.max = static_cast<float>(getMax().convertToDouble());
  spec.if_relu = 0;
  BM168x::call_global_func("backend_api_clip_global", &spec,
                           sizeof(clip_spec_t), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ClipOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::ClipOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                        int64_t h_step, int64_t d_step,
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  clip_spec_t spec = {0};
  spec.min = static_cast<float>(getMin().convertToDouble());
  spec.max = static_cast<float>(getMax().convertToDouble());
  spec.if_relu = 0;
  BM168x::call_local_func("backend_api_clip_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::ClipOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(clip_spec_t);
  clip_spec_t spec = {0};
  spec.min = static_cast<float>(getMin().convertToDouble());
  spec.max = static_cast<float>(getMax().convertToDouble());
  spec.if_relu = 0;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ClipOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(clip_spec_t);
  clip_spec_t spec = {0};
  spec.min = static_cast<float>(getMin().convertToDouble());
  spec.max = static_cast<float>(getMax().convertToDouble());
  spec.if_relu = 0;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ClipOp::get_fw_type_bm1684x() { return FW_BMNET_CLIP; }
