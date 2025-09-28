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

void tpu::CompareOp::codegen_global_bm1684x() {
  bcbinary_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_type = BM168x::compare_mode(getMode());
  spec.if_relu = 0;
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.f8_scale_A = 1.0;
  spec.f8_scale_B = 1.0;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CompareOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::CompareOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  bcbinary_local_param_t param = {0};
  auto &spec = param.spec;
  spec.common.binary_type = BM168x::compare_mode(getMode());
  spec.common.if_relu = 0;
  spec.common.scale_A = 1;
  spec.common.scale_B = 1;
  spec.common.f8_scale_A = 1.0;
  spec.common.f8_scale_B = 1.0;
  // inception setting
  auto gi = out_group_infos[0];
  auto in0_gi = in_group_infos[0];
  auto in1_gi = in_group_infos[1];
  setHWMargins(input_spec->at(0).hw_margins, in0_gi, gi);
  setHWMargins(input_spec->at(1).hw_margins, in1_gi, gi);
  BM168x::call_local_func("backend_api_bcbinary_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::CompareOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_local_param_t);
  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BM168x::compare_mode(getMode());
  param.spec.common.if_relu = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  param.spec.common.f8_scale_A = 1.0;
  param.spec.common.f8_scale_B = 1.0;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CompareOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_global_param_t);
  bcbinary_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_type = BM168x::compare_mode(getMode());
  spec.if_relu = 0;
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.f8_scale_A = 1.0;
  spec.f8_scale_B = 1.0;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::CompareOp::get_fw_type_bm1684x() {
  return FW_BMNET_BROADCAST_BINARY;
}
