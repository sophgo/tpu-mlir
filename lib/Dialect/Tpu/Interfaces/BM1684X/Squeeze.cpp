//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================
void tpu::SqueezeOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  tile_1d_global_param_t param = {0};
  param.tile_axis = 0;
  param.tile_num = 1;
  param.type = 0;
  BM168x::call_global_func("backend_api_tile_1d_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::SqueezeOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::SqueezeOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                           int64_t h_step, int64_t d_step,
                                           int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  if (input_spec->at(0).addr == output_spec->at(0).addr) {
    return;
  }
  auto shape = module::getShape(getOutput());
  reshape_spec_t spec = {0};
  spec.dims = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    spec.shape[i] = shape[i];
  }

  BM168x::call_local_func("backend_api_reshape_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SqueezeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(squeeze_dims_common_spec_t);
  squeeze_dims_common_spec_t param = {0};
  const auto axes = module::getI64Array(getAxes());
  param.axis_num = axes->size();
  for (int i = 0; i < param.axis_num; i++) {
    param.axis_list[i] = axes->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::SqueezeOp::get_fw_type_bm1684x() { return FW_BMNET_SQUEEZE_DIM; }
