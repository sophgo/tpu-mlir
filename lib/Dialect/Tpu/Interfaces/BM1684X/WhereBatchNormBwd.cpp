//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::WhereBnbwdOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  where_batchnorm_backward_param_t param = {0};
  param.do_recompute = getDoRecompute();
  BM168x::call_global_func("backend_api_where_batchnorm_backward_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}


void tpu::WhereBnbwdOp::codegen_global_cv18xx(int64_t layer_id) {}

// // dynamic codegen
// int64_t tpu::WhereBnbwdOp::dyn_codegen_local_bm1684x(void *buffer) {
//     return 0;
// }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::WhereBnbwdOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

// =========================================
// LocalGenInterface
// =========================================

// int64_t tpu::WhereBnbwdOp::getBufferSize_bm1684x(
//     int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
//     int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
//     int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
//     int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
//     // int tile = BM168x::eu_num(sizeof(float));
//     // int64_t in_slice_fp32_size = in_nslice * ceiling_func(in_cslice, BM168x::NPU_NUM) *
//     //       align_up( ceiling_func (in_hslice * in_wslice, tile) * tile, tile) * sizeof(float);
//     // auto stype = module::getStorageType(getInput());
//     // int64_t buffer_size = in_slice_fp32_size;
//     // if( !stype.isF32() ) {
//     //   int64_t channel_fp32_size = ceiling_func(in_cslice, BM168x::NPU_NUM) *
//     //         align_up( 1, tile) * sizeof(float);
//     //   buffer_size = in_slice_fp32_size * 3 + channel_fp32_size * 5;
//     // }
//     // return buffer_size;
//     return 0;
// }

// int64_t tpu::WhereBnbwdOp::getBufferSize_bm1684(
//     int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
//     int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
//   return 0;
// }

// int64_t tpu::WhereBnbwdOp::getBufferSize_cv18xx(
//     int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
//     int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
//   return 0;
// }
int64_t tpu::WhereBnbwdOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }

