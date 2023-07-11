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
void tpu::UpsampleOp::codegen_global_bm1684x() {
  assert(getScaleH() == getScaleW());
  auto op = getOperation();
  // int64_t n, c, h, w;
  // module::getNCHW(getInput(), n, c, h, w);

  upsample_spec_t spec = {0};
  spec.size = getScaleH();
  spec.if_relu = getDoRelu();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_upsample_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::UpsampleOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                            int64_t h_step, int64_t d_step,
                                            int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  assert(getScaleH() == getScaleW());
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  upsample_spec_t spec = {0};
  spec.size = getScaleH();
  spec.if_relu = getDoRelu();

  BM168x::call_local_func("backend_api_upsample_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::UpsampleOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(upsample_spec_t);
  upsample_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.size = getScaleH();
  spec.if_relu = getDoRelu();

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::UpsampleOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(upsample_spec_t);
  upsample_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.size = getScaleH();
  spec.if_relu = getDoRelu();

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::UpsampleOp::get_fw_type_bm1684x() { return FW_BMNET_UPSAMPLE; }
